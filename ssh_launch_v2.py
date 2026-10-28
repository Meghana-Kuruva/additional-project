"""
Installs google-genai (new SDK), uploads corrected eval script
that reads from the image-folder layout, and relaunches.
"""
import paramiko, sys, time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

HOST = '61.1.175.170'; PORT = 2222
USER = 'piyush'; PASS = '2£K40d#N'
GEMINI_KEY = 'AIzaSyB0vfPqn05ZtNt2lEOgSPdvI38hyBIrW-Q'
WORK_DIR   = '/home/piyush/fer_eval'
CONDA_PY   = '/home/piyush/miniconda3/bin/python3'
CONDA_PIP  = '/home/piyush/miniconda3/bin/pip'

# ── Corrected eval script ─────────────────────────────────────────────────────
EVAL_SCRIPT = '''
"""
FER2013 Zero-Shot Evaluation — Gemini 1.5 Flash (new google.genai SDK)
Reads images from: /home/piyush/fer_eval/test/<emotion>/*.jpg
"""
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from google import genai
from google.genai import types
from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    sys.exit("ERROR: GEMINI_API_KEY not set")

BASE_DIR   = Path(__file__).parent
TEST_DIR   = BASE_DIR / "test"
CKPT_FILE  = BASE_DIR / "fer2013_gemini_results.json"
REPORT_TXT = BASE_DIR / "fer2013_gemini_report.txt"

# FER2013 folder names -> integer labels (matches original FER2013 ordering)
FOLDER_TO_INT = {
    "angry":    0,
    "disgust":  1,
    "fear":     2,
    "happy":    3,
    "sad":      4,
    "surprise": 5,
    "neutral":  6,
}
INT_TO_NAME = {v: k.capitalize() for k, v in FOLDER_TO_INT.items()}
INT_TO_NAME[2] = "Fear"   # keep consistent with paper labels

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

REQUESTS_PER_MINUTE = 55
SLEEP_SEC = 60.0 / REQUESTS_PER_MINUTE
CHECKPOINT_EVERY = 50


def collect_samples(test_dir):
    samples = []
    for folder, label in FOLDER_TO_INT.items():
        folder_path = test_dir / folder
        if not folder_path.exists():
            print(f"WARNING: folder missing: {folder_path}", flush=True)
            continue
        for img_path in sorted(folder_path.glob("*.jpg")):
            samples.append((str(img_path), label))
    return samples


def parse_response(text):
    if not text:
        return -1
    cleaned = text.strip().lower().rstrip(".")
    if cleaned in WORD_TO_INT:
        return WORD_TO_INT[cleaned]
    for word, idx in WORD_TO_INT.items():
        if word in cleaned:
            return idx
    return -1


def query_gemini(client, model_name, img_path, idx):
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    img_part = types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[PROMPT, img_part],
            )
            return response.text.strip()
        except Exception as exc:
            wait = 10 * (2 ** attempt)
            print(f"  [img {idx}] attempt {attempt+1} error: {exc} "
                  f"retrying in {wait}s", flush=True)
            time.sleep(wait)
    return ""


def main():
    if not TEST_DIR.exists():
        sys.exit(f"ERROR: test directory not found: {TEST_DIR}")

    samples = collect_samples(TEST_DIR)
    total = len(samples)
    print(f"Found {total} test images across {len(FOLDER_TO_INT)} classes",
          flush=True)

    # Class distribution
    from collections import Counter
    dist = Counter(label for _, label in samples)
    for i in range(7):
        print(f"  {INT_TO_NAME[i]:10s}: {dist.get(i, 0)}", flush=True)

    client = genai.Client(api_key=GEMINI_API_KEY)
    model_name = "gemini-1.5-flash"

    # Resume from checkpoint
    results = []
    start_idx = 0
    if CKPT_FILE.exists():
        with open(CKPT_FILE) as f:
            results = json.load(f)
        start_idx = len(results)
        print(f"Resuming from checkpoint: {start_idx}/{total}", flush=True)

    eta_min = (total - start_idx) * SLEEP_SEC / 60
    print(f"Rate: {REQUESTS_PER_MINUTE} RPM | ETA: ~{eta_min:.0f} min",
          flush=True)

    t0 = time.time()
    for global_i, (img_path, true_label) in enumerate(samples[start_idx:],
                                                        start=start_idx):
        raw  = query_gemini(client, model_name, img_path, global_i)
        pred = parse_response(raw)

        results.append({
            "idx":  global_i,
            "path": img_path,
            "true": true_label,
            "pred": pred,
            "raw":  raw,
        })

        if (global_i + 1) % CHECKPOINT_EVERY == 0 or global_i == total - 1:
            with open(CKPT_FILE, "w") as f:
                json.dump(results, f)
            elapsed = (time.time() - t0) / 60
            acc_so_far = sum(
                1 for r in results if r["pred"] == r["true"]
            ) / max(len(results), 1) * 100
            print(
                f"  [{global_i+1}/{total}] {elapsed:.1f}min | "
                f"running acc: {acc_so_far:.1f}% | "
                f"last: \\"{raw}\\" -> {INT_TO_NAME.get(pred, '?')} "
                f"(true: {INT_TO_NAME[true_label]})",
                flush=True
            )

        time.sleep(SLEEP_SEC)

    # ── Final metrics ─────────────────────────────────────────────────────────
    valid  = [r for r in results if r["pred"] != -1]
    bad    = len(results) - len(valid)
    y_true = [r["true"] for r in valid]
    y_pred = [r["pred"] for r in valid]

    label_names = [INT_TO_NAME[i] for i in range(7)]
    acc = accuracy_score(y_true, y_pred) * 100
    mf1 = f1_score(y_true, y_pred, average="macro",    zero_division=0) * 100
    wf1 = f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100
    report = classification_report(
        y_true, y_pred, target_names=label_names,
        zero_division=0, digits=4
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(7)))

    lines = [
        "=" * 64,
        "  Gemini-1.5-Flash  Zero-Shot  FER2013 PrivateTest",
        "=" * 64,
        f"  Images evaluated : {total}",
        f"  Valid predictions: {len(valid)}  (unparseable: {bad})",
        f"  Accuracy         : {acc:.2f}%",
        f"  Macro-F1         : {mf1:.2f}%",
        f"  Weighted-F1      : {wf1:.2f}%",
        "",
        report,
        "Confusion Matrix (rows=true, cols=predicted)",
        f"Labels: {label_names}",
        str(cm),
    ]
    output = "\\n".join(lines)
    print("\\n" + output, flush=True)
    with open(REPORT_TXT, "w") as f:
        f.write(output)
    print(f"\\nReport saved -> {REPORT_TXT}", flush=True)


if __name__ == "__main__":
    main()
'''


def run(client, cmd, timeout=120):
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors='replace').strip()
    err = stderr.read().decode(errors='replace').strip()
    return out, err


def main():
    print("Connecting...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)
    print("Connected.\n")

    # Step 1: Kill any old running eval
    print("Step 1: Stopping any previous eval job...")
    out, _ = run(client,
        'pkill -f eval_fer2013_gemini || echo "none running"')
    print(f"  {out}")

    # Step 2: Install new google-genai SDK
    print("\nStep 2: Installing google-genai (new SDK)...")
    out, err = run(client,
        f'{CONDA_PIP} install --quiet google-genai 2>&1 | tail -3',
        timeout=120)
    print(f"  {out or err}")

    # Verify new SDK import
    out, _ = run(client,
        f'{CONDA_PY} -c "from google import genai; print(\'genai OK\')" 2>&1')
    print(f"  SDK check: {out}")

    # Step 3: Verify image folders exist
    print("\nStep 3: Verifying image folders...")
    out, _ = run(client,
        f'ls {WORK_DIR}/test/ 2>/dev/null && '
        f'for d in {WORK_DIR}/test/*/; do '
        f'echo "$(basename $d): $(ls $d | wc -l) images"; done')
    print(f"  {out}")

    # Step 4: Upload corrected script
    print("\nStep 4: Uploading corrected eval script...")
    remote_script = f'{WORK_DIR}/eval_fer2013_gemini.py'
    sftp = client.open_sftp()
    with sftp.open(remote_script, 'w') as f:
        f.write(EVAL_SCRIPT)
    sftp.close()
    print(f"  Uploaded -> {remote_script}")

    # Step 5: Launch
    print("\nStep 5: Launching evaluation (nohup)...")
    log = f'{WORK_DIR}/eval_log.txt'
    # Clear old log
    run(client, f'> {log}')
    launch = (
        f'cd {WORK_DIR} && '
        f'GEMINI_API_KEY={GEMINI_KEY} '
        f'nohup {CONDA_PY} {remote_script} '
        f'> {log} 2>&1 & echo PID:$!'
    )
    out, _ = run(client, launch)
    print(f"  {out}")

    time.sleep(6)
    out, _ = run(client, f'tail -15 {log}')
    print(f"\nLog:\n{out}")

    out, _ = run(client,
        'ps aux | grep eval_fer2013 | grep -v grep | head -2')
    print(f"\nProcess:\n{out or '(check log — may have exited with error)'}")

    client.close()
    log_path = f'/home/piyush/fer_eval/eval_log.txt'
    print(f"\n{'='*55}")
    print(f"Evaluation is running in the background.")
    print(f"Monitor with:")
    print(f"  ssh piyush@61.1.175.170 -p 2222 'tail -f {log_path}'")


if __name__ == '__main__':
    main()
