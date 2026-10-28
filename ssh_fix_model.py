"""
Kill old process, probe available models, fix model name, relaunch.
"""
import paramiko, sys, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

HOST='61.1.175.170'; PORT=2222; USER='piyush'; PASS='2£K40d#N'
GEMINI_KEY = 'AIzaSyB0vfPqn05ZtNt2lEOgSPdvI38hyBIrW-Q'
WORK_DIR   = '/home/piyush/fer_eval'
CONDA_PY   = '/home/piyush/miniconda3/bin/python3'
LOG        = f'{WORK_DIR}/eval_log.txt'

EVAL_SCRIPT = '''
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from google import genai
from google.genai import types
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    sys.exit("ERROR: GEMINI_API_KEY not set")

BASE_DIR  = Path(__file__).parent
TEST_DIR  = BASE_DIR / "test"
CKPT_FILE = BASE_DIR / "fer2013_gemini_results.json"
REPORT    = BASE_DIR / "fer2013_gemini_report.txt"

FOLDER_TO_INT = {
    "angry":0, "disgust":1, "fear":2, "happy":3,
    "sad":4,   "surprise":5, "neutral":6,
}
INT_TO_NAME = {
    0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy",
    4:"Sad",   5:"Surprise", 6:"Neutral",
}
WORD_TO_INT = {
    "angry":0,"anger":0,
    "disgusted":1,"disgust":1,
    "fearful":2,"fear":2,"scared":2,
    "happy":3,"happiness":3,"joy":3,
    "sad":4,"sadness":4,"unhappy":4,
    "surprised":5,"surprise":5,
    "neutral":6,
}
PROMPT = (
    "Look at the person's face in this image. "
    "What is their primary emotion? "
    "Respond with ONLY one of the following words: "
    "Happy, Sad, Angry, Fearful, Surprised, Disgusted, Neutral."
)

RPM       = 55
SLEEP_SEC = 60.0 / RPM
CKPT_N    = 50


def collect_samples(test_dir):
    samples = []
    # Only PrivateTest images (filename prefix)
    for folder, label in FOLDER_TO_INT.items():
        p = test_dir / folder
        if not p.exists():
            continue
        for img in sorted(p.glob("PrivateTest_*.jpg")):
            samples.append((str(img), label))
    return samples


def parse_response(text):
    if not text:
        return -1
    c = text.strip().lower().rstrip(".")
    if c in WORD_TO_INT:
        return WORD_TO_INT[c]
    for w, idx in WORD_TO_INT.items():
        if w in c:
            return idx
    return -1


def query(client, model_name, img_path, idx):
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    part = types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=[PROMPT, part],
            )
            return resp.text.strip()
        except Exception as exc:
            wait = 10 * (2 ** attempt)
            print(f"  [img {idx}] attempt {attempt+1}: {exc} retrying {wait}s",
                  flush=True)
            time.sleep(wait)
    return ""


def main():
    if not TEST_DIR.exists():
        sys.exit(f"ERROR: {TEST_DIR} not found")

    samples = collect_samples(TEST_DIR)
    total   = len(samples)
    from collections import Counter
    dist = Counter(lbl for _, lbl in samples)
    print(f"PrivateTest images: {total}", flush=True)
    for i in range(7):
        print(f"  {INT_TO_NAME[i]:10s}: {dist.get(i,0)}", flush=True)

    # Use v1 API (not v1beta) so gemini-1.5-flash is reachable
    client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options={"api_version": "v1"},
    )
    model_name = "gemini-1.5-flash"

    # Quick smoke test
    test_path, _ = samples[0]
    print(f"Smoke test on {test_path}...", flush=True)
    raw = query(client, model_name, test_path, -1)
    if not raw:
        # Fall back to gemini-2.0-flash if 1.5 still fails
        print("gemini-1.5-flash failed, trying gemini-2.0-flash...", flush=True)
        model_name = "gemini-2.0-flash"
        raw = query(client, model_name, test_path, -1)
    print(f"Smoke test response: {raw!r} | Model: {model_name}", flush=True)

    results = []
    start_idx = 0
    if CKPT_FILE.exists():
        with open(CKPT_FILE) as f:
            results = json.load(f)
        start_idx = len(results)
        print(f"Resuming from checkpoint: {start_idx}/{total}", flush=True)

    eta = (total - start_idx) * SLEEP_SEC / 60
    print(f"Model: {model_name} | Rate: {RPM} RPM | ETA: ~{eta:.0f} min",
          flush=True)

    t0 = time.time()
    for global_i, (img_path, true_label) in enumerate(
            samples[start_idx:], start=start_idx):
        raw  = query(client, model_name, img_path, global_i)
        pred = parse_response(raw)
        results.append({"idx":global_i, "path":img_path,
                         "true":true_label, "pred":pred, "raw":raw})

        if (global_i + 1) % CKPT_N == 0 or global_i == total - 1:
            with open(CKPT_FILE, "w") as f:
                json.dump(results, f)
            elapsed = (time.time() - t0) / 60
            acc_so_far = sum(
                1 for r in results if r["pred"] == r["true"]
            ) / max(len(results), 1) * 100
            print(
                f"  [{global_i+1}/{total}] {elapsed:.1f}min | "
                f"acc: {acc_so_far:.1f}% | "
                f"\\"{raw}\\" -> {INT_TO_NAME.get(pred,\'?\')} "
                f"(true: {INT_TO_NAME[true_label]})",
                flush=True,
            )
        time.sleep(SLEEP_SEC)

    # Metrics
    valid  = [r for r in results if r["pred"] != -1]
    bad    = len(results) - len(valid)
    y_true = [r["true"] for r in valid]
    y_pred = [r["pred"] for r in valid]
    lnames = [INT_TO_NAME[i] for i in range(7)]
    acc = accuracy_score(y_true, y_pred) * 100
    mf1 = f1_score(y_true, y_pred, average="macro",    zero_division=0) * 100
    wf1 = f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100
    rep = classification_report(y_true, y_pred, target_names=lnames,
                                zero_division=0, digits=4)
    cm  = confusion_matrix(y_true, y_pred, labels=list(range(7)))

    out = "\\n".join([
        "=" * 64,
        f"  {model_name}  Zero-Shot  FER2013 PrivateTest",
        "=" * 64,
        f"  Images    : {total}",
        f"  Valid     : {len(valid)}  (unparseable: {bad})",
        f"  Accuracy  : {acc:.2f}%",
        f"  Macro-F1  : {mf1:.2f}%",
        f"  Wtd-F1    : {wf1:.2f}%",
        "", rep,
        "Confusion Matrix (rows=true, cols=predicted)",
        f"Labels: {lnames}",
        str(cm),
    ])
    print("\\n" + out, flush=True)
    with open(REPORT, "w") as f:
        f.write(out)
    print(f"\\nSaved -> {REPORT}", flush=True)


if __name__ == "__main__":
    main()
'''


def run(client, cmd, timeout=30):
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors='replace').strip()
    err = stderr.read().decode(errors='replace').strip()
    return out, err


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)
    print("Connected.\n")

    # Kill old process
    out, _ = run(client, 'pkill -f eval_fer2013_gemini; sleep 1; echo done')
    print(f"Killed old process: {out}")

    # Quick API test: list available models
    print("\nChecking available models...")
    probe = f'''
import warnings; warnings.filterwarnings("ignore")
from google import genai
c = genai.Client(api_key="{GEMINI_KEY}", http_options={{"api_version":"v1"}})
for m in list(c.models.list())[:20]:
    if "flash" in m.name.lower() or "pro" in m.name.lower():
        print(m.name)
'''
    out, _ = run(client,
        f'{CONDA_PY} -c \'{probe}\' 2>/dev/null | head -15',
        timeout=30)
    print(f"Models:\n{out}")

    # Upload fixed script
    print("\nUploading fixed script...")
    sftp = client.open_sftp()
    with sftp.open(f'{WORK_DIR}/eval_fer2013_gemini.py', 'w') as f:
        f.write(EVAL_SCRIPT)
    sftp.close()

    # Clear checkpoint to restart fresh
    run(client, f'rm -f {WORK_DIR}/fer2013_gemini_results.json')

    # Launch
    run(client, f'> {LOG}')
    chan = client.get_transport().open_session()
    chan.exec_command(
        f'cd {WORK_DIR} && GEMINI_API_KEY={GEMINI_KEY} '
        f'nohup {CONDA_PY} {WORK_DIR}/eval_fer2013_gemini.py '
        f'>> {LOG} 2>&1 &'
    )
    time.sleep(2)
    chan.close()

    time.sleep(8)
    out, _ = run(client, f'tail -15 {LOG}')
    print(f"\nLog:\n{out}")

    out, _ = run(client, 'ps aux | grep eval_fer2013 | grep -v grep | head -2')
    print(f"\nProcess:\n{out or '(check log)'}")
    client.close()


if __name__ == '__main__':
    main()
