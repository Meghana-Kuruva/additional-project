"""
Remote setup and launch script for FER2013 Gemini evaluation.
Uses miniconda3 on the remote server to avoid PEP 668 restrictions.
"""
import paramiko
import time
import sys

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

HOST = '61.1.175.170'
PORT = 2222
USER = 'piyush'
PASS = '2£K40d#N'
GEMINI_KEY = 'AIzaSyB0vfPqn05ZtNt2lEOgSPdvI38hyBIrW-Q'
WORK_DIR = '/home/piyush/fer_eval'
CONDA_PYTHON = '/home/piyush/miniconda3/bin/python3'
CONDA_PIP    = '/home/piyush/miniconda3/bin/pip'

EVAL_SCRIPT = r'''
import os, sys, time, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import google.generativeai as genai
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    sys.exit("ERROR: GEMINI_API_KEY not set")

BASE_DIR   = Path(__file__).parent
FER_CSV    = BASE_DIR / "fer2013.csv"
CKPT_FILE  = BASE_DIR / "fer2013_gemini_results.json"
REPORT_TXT = BASE_DIR / "fer2013_gemini_report.txt"

REQUESTS_PER_MINUTE = 55
SLEEP_SEC = 60.0 / REQUESTS_PER_MINUTE
CHECKPOINT_EVERY = 50

INT_TO_NAME = {
    0:"Angry", 1:"Disgust", 2:"Fear",
    3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"
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

def pixels_to_pil(pixel_str, upscale=128):
    arr = np.array(pixel_str.split(), dtype=np.uint8).reshape(48, 48)
    img = Image.fromarray(arr, mode="L").convert("RGB")
    if upscale != 48:
        img = img.resize((upscale, upscale), Image.LANCZOS)
    return img

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

def query_gemini(model, img, idx):
    for attempt in range(3):
        try:
            response = model.generate_content([PROMPT, img])
            return response.text.strip()
        except Exception as exc:
            wait = 10 * (2 ** attempt)
            print(f"  [img {idx}] attempt {attempt+1} error: {exc} retrying in {wait}s",
                  flush=True)
            time.sleep(wait)
    return ""

def main():
    if not FER_CSV.exists():
        sys.exit(f"ERROR: {FER_CSV} not found")

    print("Loading FER2013...", flush=True)
    df = pd.read_csv(FER_CSV)
    test_df = df[df["Usage"] == "PrivateTest"].reset_index(drop=True)
    total = len(test_df)
    print(f"PrivateTest: {total} images", flush=True)

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

    results = []
    start_idx = 0
    if CKPT_FILE.exists():
        with open(CKPT_FILE) as f:
            results = json.load(f)
        start_idx = len(results)
        print(f"Resuming from checkpoint: {start_idx}/{total}", flush=True)

    print(f"Rate: {REQUESTS_PER_MINUTE} RPM | ETA: ~{(total-start_idx)*SLEEP_SEC/60:.0f} min",
          flush=True)

    t0 = time.time()
    for local_i, row in enumerate(test_df.iloc[start_idx:].itertuples()):
        global_i = start_idx + local_i
        true_label = int(row.emotion)
        img = pixels_to_pil(row.pixels)
        raw = query_gemini(model, img, global_i)
        pred = parse_response(raw)

        results.append({"idx": global_i, "true": true_label,
                        "pred": pred, "raw": raw})

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
                f"last: '{raw}' -> {INT_TO_NAME.get(pred,'?')} "
                f"(true: {INT_TO_NAME[true_label]})",
                flush=True
            )

        time.sleep(SLEEP_SEC)

    # Final metrics
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
    output = "\n".join(lines)
    print("\n" + output, flush=True)

    with open(REPORT_TXT, "w") as f:
        f.write(output)
    print(f"\nReport -> {REPORT_TXT}", flush=True)

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

    # Step 1: Find best Python (prefer miniconda, else venv, else system with flag)
    print("Step 1: Locating Python...")
    python_candidates = [
        '/home/piyush/miniconda3/bin/python3',
        '/home/piyush/venv/bin/python3',
        '/usr/bin/python3',
    ]
    pip_candidates = [
        '/home/piyush/miniconda3/bin/pip',
        '/home/piyush/venv/bin/pip',
        '/usr/bin/pip3',
    ]
    py_bin = pip_bin = None
    for p, pip in zip(python_candidates, pip_candidates):
        out, _ = run(client, f'{p} --version 2>/dev/null')
        if out:
            py_bin, pip_bin = p, pip
            print(f"  Using: {p} ({out})")
            break
    if not py_bin:
        print("  No Python found — aborting.")
        client.close(); sys.exit(1)

    # Step 2: Install packages
    print("\nStep 2: Installing packages...")
    pkgs = "google-generativeai pillow scikit-learn pandas numpy kaggle"
    flag = "--break-system-packages" if "usr/bin" in pip_bin else ""
    out, err = run(client,
        f'{pip_bin} install --quiet {flag} {pkgs} 2>&1 | tail -8',
        timeout=300)
    print(f"  {out or err}")

    # Find kaggle binary
    out, _ = run(client, 'find /home/piyush -name kaggle -type f 2>/dev/null | head -3')
    kaggle_bin = out.split('\n')[0].strip() if out else 'kaggle'
    print(f"  Kaggle binary: {kaggle_bin}")

    # Step 3: Download FER2013
    print("\nStep 3: Downloading FER2013 dataset...")
    out, _ = run(client, f'ls {WORK_DIR}/fer2013.csv 2>/dev/null || echo MISSING')
    if 'MISSING' in out:
        dl_out, dl_err = run(client,
            f'mkdir -p {WORK_DIR} && '
            f'{kaggle_bin} datasets download -d msambare/fer2013 '
            f'--unzip -p {WORK_DIR} 2>&1',
            timeout=300)
        print(f"  {dl_out or dl_err}")
        out, _ = run(client, f'ls -lh {WORK_DIR}/fer2013.csv 2>/dev/null || echo MISSING')
        print(f"  Result: {out}")
        if 'MISSING' in out:
            print("  Kaggle download failed — check Kaggle credentials on server.")
            client.close(); sys.exit(1)
    else:
        print(f"  Already present: {out}")

    # Step 4: Upload eval script
    print("\nStep 4: Uploading evaluation script...")
    remote_script = f'{WORK_DIR}/eval_fer2013_gemini.py'
    sftp = client.open_sftp()
    with sftp.open(remote_script, 'w') as f:
        f.write(EVAL_SCRIPT)
    sftp.close()
    print(f"  Uploaded -> {remote_script}")

    # Step 5: Launch with nohup
    print("\nStep 5: Launching evaluation...")
    log_file = f'{WORK_DIR}/eval_log.txt'
    launch = (
        f'cd {WORK_DIR} && '
        f'GEMINI_API_KEY={GEMINI_KEY} '
        f'nohup {py_bin} {remote_script} '
        f'> {log_file} 2>&1 & echo PID:$!'
    )
    out, err = run(client, launch)
    print(f"  {out or err}")

    time.sleep(5)
    out, _ = run(client, f'tail -10 {log_file} 2>/dev/null')
    print(f"\nLog output:\n{out}")

    out, _ = run(client,
        f'ps aux | grep eval_fer2013_gemini | grep -v grep | head -2')
    print(f"\nProcess check:\n{out or '(not found — see log for error)'}")

    client.close()
    print(f"\n{'='*50}")
    print(f"Monitor progress:")
    print(f"  ssh piyush@61.1.175.170 -p 2222 'tail -f {log_file}'")


if __name__ == '__main__':
    main()
