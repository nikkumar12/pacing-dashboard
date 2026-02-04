# pacing_final_pipeline.py
"""
Final end-to-end pacing pipeline (corrected).
- verifies the file,
- computes CTR/CPM/CVR/Pacing_Rate,
- writes them back to the same Excel file (creates a backup),
- trains a classifier (LightGBM preferred, RandomForest fallback),
- evaluates robustly and saves outputs.
"""

import os
import sys
from pathlib import Path
import time
import hashlib
import warnings
warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
TARGET_FILE = Path(r"C:\Users\nikkumar12\OneDrive - Publicis Groupe\2025\Python pacing script test\sample_pacing_data.xlsx")
OUTPUT_DIR = TARGET_FILE.parent / "ml_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_SUFFIX = ".backup.xlsx"
# label thresholds
OVERSHOOT = 1.15
UNDERSHOOT = 0.85
# --------------------------------

def fatal(msg):
    print("❌", msg)
    sys.exit(1)

def info(msg):
    print("ℹ️", msg)

# 1) Verify file exists and show basic identity info
print("\n=== File verification ===")
if not TARGET_FILE.exists():
    fatal(f"Target file not found: {TARGET_FILE}")

stat = TARGET_FILE.stat()
print("File:", TARGET_FILE.resolve())
print("Size:", f"{stat.st_size:,} bytes")
print("Last modified:", time.ctime(stat.st_mtime))
# short sha
try:
    h = hashlib.sha256()
    with TARGET_FILE.open("rb") as fh:
        chunk = fh.read(8192)
        while chunk:
            h.update(chunk)
            chunk = fh.read(8192)
    print("SHA256 (first 12 chars):", h.hexdigest()[:12])
except Exception:
    pass

# 2) Force working dir to the file folder to avoid relative-path confusion
os.chdir(TARGET_FILE.parent)
print("Working directory set to:", os.getcwd())

# 3) Check pandas/excel engine availability
try:
    import pandas as pd
    import numpy as np
except Exception as e:
    fatal("pandas/numpy not installed in this interpreter. Install via pip install pandas numpy")

# pandas uses openpyxl to read/write xlsx — check
try:
    import openpyxl  # noqa: F401
except Exception:
    fatal("Missing dependency 'openpyxl'. Install in your venv: pip install openpyxl")

# 4) Load the Excel file (first sheet)
print("\n=== Loading Excel ===")
try:
    xls = pd.ExcelFile(TARGET_FILE)
    sheet = xls.sheet_names[0]
    df = pd.read_excel(TARGET_FILE, sheet_name=sheet, header=0)
except Exception as e:
    fatal(f"Failed to read Excel: {e}")

print("Loaded sheet:", sheet, "shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nSample (first 6 rows):")
with pd.option_context('display.max_columns', None, 'display.width', 200):
    print(df.head(6).to_string(index=False))

# 5) Ensure required columns exist (allow forgiving name matching)
required = {"campaign_id","date","planned_spend","actual_spend","impressions","clicks","conversions"}
# map existing columns to normalized keys
col_map = {c: c.strip() for c in df.columns}
norm_map = {c.strip().lower(): c for c in df.columns}

missing = [r for r in required if r not in norm_map]
if missing:
    fatal(f"Missing required columns (after normalizing): {missing}\nFound columns (normalized): {list(norm_map.keys())}")

# resolve original column names
C_CAM = norm_map["campaign_id"]
C_DATE = norm_map["date"]
C_PLANNED = norm_map["planned_spend"]
C_ACTUAL = norm_map["actual_spend"]
C_IMPS = norm_map["impressions"]
C_CLICKS = norm_map["clicks"]
C_CONV = norm_map["conversions"]

# 6) Coerce numeric columns robustly (remove commas, trim, convert)
def coerce_numeric(col_name):
    s = df[col_name].astype(str).str.replace(",","").str.strip()
    coerced = pd.to_numeric(s, errors="coerce").fillna(0)
    return coerced

print("\n=== Coercing numeric columns ===")
df[C_PLANNED] = coerce_numeric(C_PLANNED)
df[C_ACTUAL]  = coerce_numeric(C_ACTUAL)
df[C_IMPS]    = coerce_numeric(C_IMPS)
df[C_CLICKS]  = coerce_numeric(C_CLICKS)
df[C_CONV]    = coerce_numeric(C_CONV)

# 7) Compute CTR, CPM, CVR, Pacing_Rate (safe against zero divisions)
print("\n=== Calculating metrics (CTR, CPM, CVR, Pacing_Rate) ===")
df["CTR"] = (df[C_CLICKS] / df[C_IMPS].replace({0: np.nan})).fillna(0) * 100
df["CPM"] = (df[C_ACTUAL] / df[C_IMPS].replace({0: np.nan})).fillna(0) * 1000
df["CVR"] = (df[C_CONV] / df[C_CLICKS].replace({0: np.nan})).fillna(0) * 100
df["Pacing_Rate"] = (df[C_ACTUAL] / df[C_PLANNED].replace({0: np.nan})).fillna(0)

df["CTR"] = df["CTR"].round(4)
df["CPM"] = df["CPM"].round(4)
df["CVR"] = df["CVR"].round(4)
df["Pacing_Rate"] = df["Pacing_Rate"].round(4)

print("Sample computed metrics (first 6 rows):")
with pd.option_context('display.max_columns', None):
    print(df[[C_CAM, C_DATE, C_PLANNED, C_ACTUAL, C_IMPS, C_CLICKS, C_CONV, "CTR", "CPM", "CVR", "Pacing_Rate"]].head(6).to_string(index=False))

# 8) Save a backup and then write updated values back to the same Excel file
backup = TARGET_FILE.with_name(TARGET_FILE.stem + BACKUP_SUFFIX)
try:
    df.to_excel(backup, index=False)
    info(f"Backup written to: {backup}")
except Exception as e:
    print("⚠️ Could not write backup:", e)

try:
    df.to_excel(TARGET_FILE, index=False)
    info(f"Updated file written (metrics added) to: {TARGET_FILE}")
except Exception as e:
    fatal(f"Failed to write updated Excel file: {e}")

# 9) Create label column
def label_from_rate(r):
    if r > OVERSHOOT:
        return "Overspend"
    elif r < UNDERSHOOT:
        return "Underspend"
    else:
        return "On Track"

df["Status"] = df["Pacing_Rate"].apply(label_from_rate)

print("\nLabel distribution:")
print(df["Status"].value_counts())

# 10) Prepare features & target for ML
features = ["CTR", "CPM", "CVR", "Pacing_Rate"]
X = df[features].fillna(0)
y = df["Status"].astype(str)

# encode labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_enc = le.fit_transform(y)
print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# 11) Check there is more than one class
import numpy as np
if len(le.classes_) == 1:
    info("Only one label present in data; skipping ML training. Add more historical rows to train.")
    sys.exit(0)

# 12) Train/test split (use stratify only if each class has >=2 members)
from sklearn.model_selection import train_test_split
class_counts = np.bincount(y_enc)
can_stratify = (len(class_counts) > 1) and (class_counts.min() >= 2)
stratify_arg = y_enc if can_stratify else None
if not can_stratify:
    info("Not enough examples per class for safe stratified split; doing random split without stratify.")
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.25, random_state=42, stratify=stratify_arg)

# 13) Try LightGBM, fallback to RandomForest
model = None
use_lgb = False
try:
    import lightgbm as lgb
    params = {
        "objective": "multiclass",
        "num_class": len(le.classes_),
        "learning_rate": 0.1,
        "num_leaves": 31,
        "verbosity": -1,
        "seed": 42
    }
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_test, label=y_test, reference=lgb_train)
    model = lgb.train(params, lgb_train, valid_sets=[lgb_val], num_boost_round=200, early_stopping_rounds=20, verbose_eval=False)
    use_lgb = True
    info("Trained LightGBM model.")
except Exception as e:
    info(f"LightGBM not available or failed ({e}). Falling back to RandomForest.")
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    model = rf

# 14) Predictions & evaluation (robust)
if use_lgb:
    y_pred = np.argmax(model.predict(X_test), axis=1)
else:
    y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print("\nConfusion matrix (test set):")
print(confusion_matrix(y_test, y_pred))

# classification_report: only for labels present in y_test -> avoid mismatch error
labels_present = np.unique(y_test)
target_names_present = le.inverse_transform(labels_present)
print("\nClassification report (only labels present in test set):")
print(classification_report(y_test, y_pred, labels=labels_present, target_names=target_names_present, zero_division=0))

# 15) Save predictions for full dataset
print("\nSaving predictions and probabilities (full dataset)...")
if use_lgb:
    probs_full = model.predict(X)  # shape: (n_rows, n_classes)
    pred_idx_full = np.argmax(probs_full, axis=1)
else:
    try:
        probs_full = model.predict_proba(X)
        pred_idx_full = np.argmax(probs_full, axis=1)
    except Exception:
        probs_full = None
        pred_idx_full = model.predict(X)

df["Predicted_Status"] = le.inverse_transform(pred_idx_full)
if probs_full is not None:
    for i, cls in enumerate(le.classes_):
        df[f"prob_{cls}"] = probs_full[:, i]

out_path = OUTPUT_DIR / "pacing_predictions_final.xlsx"
try:
    df.to_excel(out_path, index=False)
    info(f"Predictions written to: {out_path}")
except Exception as e:
    print("⚠️ Failed to write predictions:", e)

# 16) Save model object if joblib available
try:
    import joblib
    model_save = OUTPUT_DIR / "pacing_model.joblib"
    joblib.dump({"model": model, "label_encoder": le, "features": features, "use_lgb": use_lgb}, model_save)
    info(f"Model saved to: {model_save}")
except Exception as e:
    print("⚠️ Could not save model (joblib missing or other error):", e)

print("\n✅ Pipeline complete. Open the Excel and the ml_outputs folder to review results.")
