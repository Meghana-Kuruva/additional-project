"""
Extended experiments for MACE paper:
  A) Class-weighted classifiers (imbalance fix)
  B) SMOTE + LR
  C) Automated ablation proxy:
       Ablation tests whether removing linguistic context
       (operationalised as n-gram scope) degrades per-class F1,
       especially on minority classes.
       Variants:
         V1 - Unigram only (baseline, simulates text-only)
         V2 - Unigram + Bigram (richer context, simulates +history)
         V3 - Unigram + Bigram + Trigram (fullest surface context)
       Best classifier (LR) used throughout for ablation.
"""

import warnings, json
warnings.filterwarnings("ignore")

import pandas as pd
import neattext.functions as nfx
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report
)

# ── Load & preprocess ──────────────────────────────────────────────────────
df = pd.read_csv(
    r"c:\Users\megha\OneDrive\Desktop\Academics\Additional_project"
    r"\Text-Emotion-Detection\Text Emotion Detection\data\emotion_dataset_raw.csv"
)
df["Clean_Text"] = df["Text"].apply(nfx.remove_userhandles)
df["Clean_Text"] = df["Clean_Text"].apply(nfx.remove_stopwords)
df = df.dropna(subset=["Clean_Text"])
df = df[df["Clean_Text"].str.strip() != ""]

x = df["Clean_Text"]
y = df["Emotion"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42, stratify=y
)
CLASSES = sorted(y.unique())

def print_report(name, y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred) * 100
    mf1  = f1_score(y_true, y_pred, average="macro") * 100
    wf1  = f1_score(y_true, y_pred, average="weighted") * 100
    rep  = classification_report(y_true, y_pred, digits=4)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Accuracy   : {acc:.2f}%")
    print(f"  Macro-F1   : {mf1:.2f}%")
    print(f"  Weighted-F1: {wf1:.2f}%")
    print(rep)

# ════════════════════════════════════════════════════════
# PART A — Class-weighted classifiers
# ════════════════════════════════════════════════════════
print("\n\n" + "="*60)
print("  PART A: CLASS-WEIGHTED CLASSIFIERS")
print("="*60)

pipe_lr_w = Pipeline([
    ("cv", CountVectorizer()),
    ("lr", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
pipe_lr_w.fit(x_train, y_train)
print_report("LR (balanced)", y_test, pipe_lr_w.predict(x_test))

pipe_svm_w = Pipeline([
    ("cv", CountVectorizer()),
    ("svm", LinearSVC(max_iter=2000, class_weight="balanced"))
])
pipe_svm_w.fit(x_train, y_train)
print_report("LinearSVC (balanced)", y_test, pipe_svm_w.predict(x_test))

pipe_rf_w = Pipeline([
    ("cv", CountVectorizer()),
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42,
                                   class_weight="balanced"))
])
pipe_rf_w.fit(x_train, y_train)
print_report("RF-100 (balanced)", y_test, pipe_rf_w.predict(x_test))

# ════════════════════════════════════════════════════════
# PART B — SMOTE + LR
# ════════════════════════════════════════════════════════
print("\n\n" + "="*60)
print("  PART B: SMOTE + LR")
print("="*60)

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    smote_lr = ImbPipeline([
        ("cv",    CountVectorizer()),
        ("smote", SMOTE(random_state=42)),
        ("lr",    LogisticRegression(max_iter=1000))
    ])
    smote_lr.fit(x_train, y_train)
    print_report("SMOTE + LR", y_test, smote_lr.predict(x_test))
except Exception as e:
    print(f"  SMOTE failed: {e}")

# ════════════════════════════════════════════════════════
# PART C — Automated ablation (n-gram context proxy)
# ════════════════════════════════════════════════════════
print("\n\n" + "="*60)
print("  PART C: ABLATION — N-GRAM CONTEXT PROXY (LR)")
print("  Simulates impact of removing linguistic context:")
print("  V1=unigram (text-only), V2=+bigram (+history proxy),")
print("  V3=+trigram (full surface context)")
print("="*60)

for label, ngram in [
    ("V1 — Unigram only  [text-only]",            (1, 1)),
    ("V2 — Unigram+Bigram [+history proxy]",       (1, 2)),
    ("V3 — Unigram+Bigram+Trigram [full context]", (1, 3)),
]:
    pipe = Pipeline([
        ("cv", CountVectorizer(ngram_range=ngram)),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    pipe.fit(x_train, y_train)
    print_report(label, y_test, pipe.predict(x_test))

print("\n\nDONE")
