import os
import glob
import math
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------- USER SETTINGS ----------
BASE_DIR = r"C:\Users\nikkumar12\OneDrive - Publicis Groupe\2025\Python pacing script test"
BASENAME = "FY25 Verizon Value Brands Pacing Doc"  # no extension needed
SHEET = "Pacing Summary Overview"
OUTPUT_SHEET = "PaceSmart Output"
RANDOM_STATE = 42
# -----------------------------------

def find_excel_file(base_dir, basename):
    patterns = [
        os.path.join(base_dir, f"{basename}.xlsx"),
        os.path.join(base_dir, f"{basename}.xlsm"),
        os.path.join(base_dir, f"{basename}*.xlsx"),
        os.path.join(base_dir, f"{basename}*.xlsm"),
    ]
    for p in patterns:
        matches = glob.glob(p)
        if matches:
            # pick the most recently modified match
            matches.sort(key=os.path.getmtime, reverse=True)
            return matches[0]
    return None

def normalize_cols(df):
    # keep a mapping by lowering and stripping non-alnum
    def norm(c):
        c = str(c).strip()
        c = c.replace("\u00A0", " ")  # non-breaking spaces
        return " ".join(c.split()).lower()
    colmap = {c: norm(c) for c in df.columns}

    # expected keys (normalized)
    expects = {
        "campaign name (dsp naming)": "campaign_name_dsp",
        "cid": "cid",
        "client": "client",
        "dsp": "dsp",
        "campaign name (clean)": "campaign_name_clean",
        "channel": "channel",
        "start date": "start_date",
        "end date": "end_date",
        "budget": "budget",
        "total days": "total_days",
        "days remaining": "days_remaining",
        "total spend": "total_spend",
        "remaining budget": "remaining_budget",
        "previous flight spend": "prev_flight_spend",
        "required daily spend": "required_daily_spend",
        "yesterday's spend": "yesterday_spend",
        "% of required spend": "pct_of_required_spend",
        "pacing %": "pacing_pct",
        "3 day avg spend": "avg3_spend",
        "3 day avg pacing": "avg3_pacing",
    }

    new_cols = {}
    for orig, normed in colmap.items():
        if normed in expects:
            new_cols[orig] = expects[normed]
        else:
            new_cols[orig] = normed  # keep something sensible

    df = df.rename(columns=new_cols)
    return df

def to_num(series):
    # handles %, commas, blanks
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False),
        errors="coerce"
    )

def safe_div(a, b):
    return np.where((b is not None) & (np.array(b, dtype=float) != 0), np.array(a, dtype=float) / np.array(b, dtype=float), np.nan)

def label_from_pacing(pacing_pct):
    # pacing_pct is 0-100 scale usually; tolerate raw decimals
    pc = np.array(pacing_pct, dtype=float)
    # If values look like 0-1, convert to 0-100
    if np.nanmax(pc) <= 2.0:
        pc = pc * 100.0
    labels = []
    for v in pc:
        if np.isnan(v):
            labels.append(np.nan)
        elif v < 90:
            labels.append("Underspend")
        elif v > 110:
            labels.append("Overspend")
        else:
            labels.append("On Track")
    return pd.Series(labels)

def project_total_spend(row):
    # Use the best available daily spend signal
    req = row.get("required_daily_spend", np.nan)
    yday = row.get("yesterday_spend", np.nan)
    avg3 = row.get("avg3_spend", np.nan)
    days_rem = row.get("days_remaining", np.nan)
    total_spend = row.get("total_spend", np.nan)

    # pick driver: prefer avg3, then yesterday, else required
    daily_driver = np.nan
    for v in [avg3, yday, req]:
        if not pd.isna(v) and v > 0:
            daily_driver = v
            break

    if pd.isna(daily_driver) or pd.isna(days_rem) or pd.isna(total_spend):
        return np.nan
    return total_spend + days_rem * daily_driver

def rule_based_prediction(df):
    # Use projected_vs_budget thresholds
    pred = []
    for v in df["projected_vs_budget"]:
        if pd.isna(v):
            pred.append("On Track")
        elif v > 1.10:
            pred.append("Overspend")
        elif v < 0.90:
            pred.append("Underspend")
        else:
            pred.append("On Track")
    return pd.Series(pred, index=df.index)

def main():
    path = find_excel_file(BASE_DIR, BASENAME)
    if not path:
        print(f"❌ Could not find file in:\n  {BASE_DIR}\nLooking for base name: {BASENAME} (xlsx/xlsm)")
        return
    print(f"📄 Using file: {path}")

    # Read the pacing sheet
    try:
        df = pd.read_excel(path, sheet_name=SHEET)
    except Exception as e:
        print(f"❌ Failed to read sheet '{SHEET}': {e}")
        return

    print(f"✅ Loaded rows: {len(df)}; columns: {list(df.columns)}")

    df = normalize_cols(df)

    # Coerce numeric/date columns where present
    for col in ["budget","total_days","days_remaining","total_spend","remaining_budget",
                "prev_flight_spend","required_daily_spend","yesterday_spend",
                "pct_of_required_spend","pacing_pct","avg3_spend","avg3_pacing"]:
        if col in df.columns:
            df[col] = to_num(df[col])

    # Dates
    for col in ["start_date","end_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Derived metrics
    df["daily_pacing_variance"] = safe_div(df.get("yesterday_spend"), df.get("required_daily_spend")) - 1
    df["avg3_pacing_variance"]  = safe_div(df.get("avg3_spend"), df.get("required_daily_spend")) - 1
    df["days_remaining_pct"]    = safe_div(df.get("days_remaining"), df.get("total_days"))
    df["spend_utilization_pct"] = safe_div(df.get("total_spend"), df.get("budget")) * 100.0
    # Avg daily so far = total_spend / days_run (avoid div by zero)
    days_run = (df.get("total_days") - df.get("days_remaining"))
    df["avg_daily_spend_so_far"] = safe_div(df.get("total_spend"), days_run)

    # Projection & ratio
    df["projected_total_spend"] = df.apply(project_total_spend, axis=1)
    df["projected_vs_budget"] = safe_div(df["projected_total_spend"], df.get("budget"))

    # Current label (from pacing % if available; else from burn vs required)
    if "pacing_pct" in df.columns and df["pacing_pct"].notna().any():
        df["current_status"] = label_from_pacing(df["pacing_pct"])
    else:
        # fallback: compute pacing from avg_daily_spend_so_far vs required
        pacing_est = safe_div(df["avg_daily_spend_so_far"], df.get("required_daily_spend")) * 100.0
        df["current_status"] = label_from_pacing(pacing_est)

    # ----- ML block (only if enough rows and >=2 classes) -----
    feature_cols = [
        c for c in [
            "required_daily_spend","yesterday_spend","avg3_spend",
            "days_remaining","total_days","budget","total_spend",
            "daily_pacing_variance","avg3_pacing_variance",
            "days_remaining_pct","spend_utilization_pct",
            "avg_daily_spend_so_far","projected_vs_budget"
        ] if c in df.columns
    ]

    model_used = "Rule-based"
    preds = None

    # Filter rows with complete features and valid labels
    usable = df[feature_cols + ["current_status"]].dropna()
    class_counts = usable["current_status"].value_counts() if not usable.empty else pd.Series(dtype=int)

    can_ml = (len(usable) >= 40) and (class_counts.nunique() >= 2) and (class_counts.min() >= 5)
    # The above thresholds keep it realistic and stable. Lower if you want earlier ML.
    if can_ml:
        try:
            X = usable[feature_cols].values
            le = LabelEncoder()
            y = le.fit_transform(usable["current_status"])
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
            )
            clf = RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=4,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
            clf.fit(X_train, y_train)
            # Predict for all rows (not just usable)
            X_all = df[feature_cols].fillna(df[feature_cols].median()).values
            preds_idx = clf.predict(X_all)
            preds = pd.Series(le.inverse_transform(preds_idx), index=df.index)
            model_used = "RandomForest (ML)"
        except Exception as e:
            print(f"⚠️ ML training skipped due to: {e}")

    # Fallback rule-based if ML not used / failed
    if preds is None:
        preds = rule_based_prediction(df)

    df["predicted_status"] = preds

    # Prepare a neat output subset + keep identifiers
    id_cols = []
    for c in ["campaign_name_clean","campaign_name_dsp","cid","client","dsp","channel","start_date","end_date"]:
        if c in df.columns:
            id_cols.append(c)

    out_cols = id_cols + [
        c for c in [
            "budget","total_days","days_remaining","total_spend","remaining_budget",
            "required_daily_spend","yesterday_spend","avg3_spend",
            "pacing_pct","avg3_pacing","pct_of_required_spend",
            "daily_pacing_variance","avg3_pacing_variance","days_remaining_pct",
            "spend_utilization_pct","avg_daily_spend_so_far",
            "projected_total_spend","projected_vs_b udget".replace(" ",""), # guard
            "current_status","predicted_status"
        ] if c in df.columns
    ]

    # Fix accidental space key
    if "projected_vs_budget" in df.columns and "projected_vs_b udget" in out_cols:
        out_cols[out_cols.index("projected_vs_b udget")] = "projected_vs_budget"

    output = df[out_cols].copy()

    # Pretty rounding for numeric columns
    for c in output.columns:
        if pd.api.types.is_numeric_dtype(output[c]):
            output[c] = output[c].round(4)

    # Save to the same workbook (replace or create OUTPUT_SHEET)
    print(f"💾 Writing results to sheet: '{OUTPUT_SHEET}' in the same workbook...")
    with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        output.to_excel(writer, sheet_name=OUTPUT_SHEET, index=False)

    print("\n──────── Summary ────────")
    print(f"Rows processed: {len(df)}")
    print(f"Model used: {model_used}")
    print("Predicted Status counts:")
    print(output["predicted_status"].value_counts(dropna=False))
    print("✅ Done.")

if __name__ == "__main__":
    main()
