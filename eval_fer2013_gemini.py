"""
FER2013 Zero-Shot Evaluation — Gemini-1.5-Flash
================================================
Evaluates Gemini-1.5-Flash on the FER2013 PrivateTest partition
(3,589 images, 7 emotion classes) using the constrained prompt
described in Section VII-B of the MACE paper.

Usage
-----
  1. Download fer2013.csv from Kaggle and place it in this directory.
     https://www.kaggle.com/datasets/msambare/fer2013
  2. Set your API key:
       $env:GEMINI_API_KEY = "your-key-here"    (PowerShell)
       export GEMINI_API_KEY="your-key-here"     (bash)
  3. Run:
       .venv/Scripts/python.exe eval_fer2013_gemini.py

Checkpointing
-------------
  Results are saved every 50 images to fer2013_gemini_results.json.
  If interrupted, re-run the script — it resumes automatically.

Rate limits (Gemini-1.5-Flash)
-------------------------------
  Free tier  : 15 RPM, 1,500 req/day  → ~4 hours, needs 3 days
  Paid tier  : up to 1,000 RPM        → ~4 minutes total
  Set REQUESTS_PER_MINUTE below to match your quota.
"""

import os
import sys
import time
import json
import warnings
from io import BytesIO
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

# ── Configuration ──────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    sys.exit(
        "\nERROR: GEMINI_API_KEY environment variable is not set.\n"
        "Run:  $env:GEMINI_API_KEY = 'your-key-here'  (PowerShell)\n"
        "Then re-run this script.\n"
    )

BASE_DIR   = Path(__file__).parent
FER_CSV    = BASE_DIR / "fer2013.csv"
CKPT_FILE  = BASE_DIR / "fer2013_gemini_results.json"
REPORT_TXT = BASE_DIR / "fer2013_gemini_report.txt"

# ── Adjust to your Gemini quota ────────────────────────────────────────────────
# Free tier → 15  |  Paid tier → 200–1000
REQUESTS_PER_MINUTE = 14          # slightly under limit for safety
SLEEP_SEC = 60.0 / REQUESTS_PER_MINUTE
CHECKPOINT_EVERY = 50             # save to disk every N predictions

# ── Label definitions ──────────────────────────────────────────────────────────
INT_TO_NAME = {
    0: "Angry", 1: "Disgust", 2: "Fear",
    3: "Happy", 4: "Sad",     5: "Surprise", 6: "Neutral"
}
# Map every plausible Gemini response word → FER2013 integer
WORD_TO_INT = {
    "angry":     0, "anger":     0,
    "disgusted": 1, "disgust":   1,
    "fearful":   2, "fear":      2, "scared":    2,
    "happy":     3, "happiness": 3, "joy":        3,
    "sad":       4, "sadness":   4, "unhappy":    4,
    "surprised": 5, "surprise":  5,
    "neutral":   6,
}

# ── Exact prompt from MACE paper §VII-B ───────────────────────────────────────
PROMPT = (
    "Look at the person's face in this image. "
    "What is their primary emotion? "
    "Respond with ONLY one of the following words: "
    "Happy, Sad, Angry, Fearful, Surprised, Disgusted, Neutral."
)

# ── Image helpers ──────────────────────────────────────────────────────────────
def pixels_to_pil(pixel_str: str, upscale: int = 128) -> Image.Image:
    """
    Convert FER2013 space-separated pixel string (48×48 greyscale)
    to an RGB PIL Image, upscaled for better VLM visibility.
    """
    arr = np.array(pixel_str.split(), dtype=np.uint8).reshape(48, 48)
    img = Image.fromarray(arr, mode="L").convert("RGB")
    if upscale and upscale != 48:
        img = img.resize((upscale, upscale), Image.LANCZOS)
    return img


def parse_response(text: str) -> int:
    """Map Gemini response to FER2013 integer; returns -1 if unparseable."""
    if not text:
        return -1
    cleaned = text.strip().lower().rstrip(".")
    # exact match first
    if cleaned in WORD_TO_INT:
        return WORD_TO_INT[cleaned]
    # partial match (e.g. "Fearful." or "Happy face")
    for word, idx in WORD_TO_INT.items():
        if word in cleaned:
            return idx
    return -1


# ── Gemini query with retry ────────────────────────────────────────────────────
def query_gemini(model, img: Image.Image, idx: int) -> str:
    """Send image to Gemini; retry up to 3 times with exponential back-off."""
    for attempt in range(3):
        try:
            response = model.generate_content([PROMPT, img])
            return response.text.strip()
        except Exception as exc:
            wait = 10 * (2 ** attempt)
            print(f"  [img {idx}] attempt {attempt+1} error: {exc} — "
                  f"retrying in {wait}s")
            time.sleep(wait)
    return ""


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Validate FER2013 CSV exists
    if not FER_CSV.exists():
        sys.exit(
            f"\nERROR: {FER_CSV} not found.\n"
            "Download fer2013.csv from:\n"
            "  https://www.kaggle.com/datasets/msambare/fer2013\n"
            f"and place it in: {BASE_DIR}\n"
        )

    print("Loading FER2013 CSV...")
    df = pd.read_csv(FER_CSV)

    # Use PrivateTest (the held-out test partition, 3,589 images)
    test_df = df[df["Usage"] == "PrivateTest"].reset_index(drop=True)
    total = len(test_df)
    print(f"PrivateTest partition: {total} images")

    # Configure Gemini
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Resume from checkpoint
    results: list[dict] = []
    start_idx = 0
    if CKPT_FILE.exists():
        with open(CKPT_FILE) as f:
            results = json.load(f)
        start_idx = len(results)
        print(f"Resuming from checkpoint — {start_idx}/{total} done")

    # ── Evaluation loop ────────────────────────────────────────────────────────
    print(f"\nRate limit: {REQUESTS_PER_MINUTE} RPM "
          f"({SLEEP_SEC:.1f}s between requests)")
    eta_min = (total - start_idx) * SLEEP_SEC / 60
    print(f"ETA for remaining {total - start_idx} images: "
          f"~{eta_min:.0f} min\n")

    t0 = time.time()
    for local_i, row in enumerate(test_df.iloc[start_idx:].itertuples()):
        global_i = start_idx + local_i
        true_label = int(row.emotion)

        img = pixels_to_pil(row.pixels)
        raw = query_gemini(model, img, global_i)
        pred = parse_response(raw)

        results.append({
            "idx":  global_i,
            "true": true_label,
            "pred": pred,
            "raw":  raw,
        })

        # Progress + checkpoint
        if (global_i + 1) % CHECKPOINT_EVERY == 0 or global_i == total - 1:
            with open(CKPT_FILE, "w") as f:
                json.dump(results, f)
            elapsed = (time.time() - t0) / 60
            print(f"  [{global_i+1}/{total}] "
                  f"elapsed {elapsed:.1f} min | "
                  f"last raw: '{raw}' → pred: "
                  f"{INT_TO_NAME.get(pred, 'UNKNOWN')} | "
                  f"true: {INT_TO_NAME[true_label]}")

        time.sleep(SLEEP_SEC)

    # ── Metrics ───────────────────────────────────────────────────────────────
    valid   = [r for r in results if r["pred"] != -1]
    bad     = len(results) - len(valid)
    y_true  = [r["true"] for r in valid]
    y_pred  = [r["pred"] for r in valid]

    label_names = [INT_TO_NAME[i] for i in range(7)]
    acc = accuracy_score(y_true, y_pred) * 100
    mf1 = f1_score(y_true, y_pred, average="macro",     zero_division=0) * 100
    wf1 = f1_score(y_true, y_pred, average="weighted",  zero_division=0) * 100
    report = classification_report(
        y_true, y_pred,
        target_names=label_names,
        zero_division=0, digits=4
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(7)))

    lines = [
        "=" * 64,
        "  Gemini-1.5-Flash  ·  Zero-Shot  ·  FER2013 PrivateTest",
        "=" * 64,
        f"  Images evaluated : {total}",
        f"  Valid predictions: {len(valid)}   "
        f"(unparseable: {bad})",
        f"  Accuracy         : {acc:.2f}%",
        f"  Macro-F1         : {mf1:.2f}%",
        f"  Weighted-F1      : {wf1:.2f}%",
        "",
        report,
        "Confusion Matrix  (rows = true, cols = predicted)",
        f"Labels : {label_names}",
        str(cm),
        "",
    ]
    output = "\n".join(lines)
    print("\n" + output)

    with open(REPORT_TXT, "w") as f:
        f.write(output)
    print(f"Report saved → {REPORT_TXT}")
    print(f"Raw results  → {CKPT_FILE}")


if __name__ == "__main__":
    main()
