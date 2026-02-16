import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==================================================
# PATHS
# ==================================================
BASE_PATH = r"C:\Users\nikkumar12\PyCharmMiscProject"

RAW_FILE = f"{BASE_PATH}\\Realistic_Today_Context_Pacing_Data.xlsx"
EXCEL_FILE = f"{BASE_PATH}\\Excel_Pacing_Output.xlsx"
OUTPUT_FILE = f"{BASE_PATH}\\PaceSmart_ML_vs_Excel_Output.xlsx"

# ==================================================
# LOAD DATA
# ==================================================
daily = pd.read_excel(RAW_FILE, sheet_name="Daily_Raw_Data")
meta = pd.read_excel(RAW_FILE, sheet_name="Campaign_Metadata")
excel_pacing = pd.read_excel(EXCEL_FILE, sheet_name="Pacing")

# ==================================================
# DATE NORMALIZATION (NO FORMAT CHANGE)
# ==================================================
daily["Date"] = pd.to_datetime(daily["Date"]).dt.normalize()
meta["Flight_Start_Date"] = pd.to_datetime(meta["Flight_Start_Date"]).dt.normalize()
meta["Flight_End_Date"] = pd.to_datetime(meta["Flight_End_Date"]).dt.normalize()

YESTERDAY = (
    datetime.now(timezone.utc)
    .replace(hour=0, minute=0, second=0, microsecond=0)
    - timedelta(days=1)
).replace(tzinfo=None)

daily = daily[daily["Date"] <= YESTERDAY]

# ==================================================
# STEP 1: BUILD ML TRAINING DATA (ENDED CAMPAIGNS)
# ==================================================
rows = []

ended = meta[meta["Flight_End_Date"] < YESTERDAY]

for _, c in ended.iterrows():
    cid = c["Campaign_ID"]
    dsp = c["DSP"]
    budget = float(c["Total_Budget"])
    start = c["Flight_Start_Date"]
    end = c["Flight_End_Date"]

    d = daily[daily["Campaign_ID"] == cid]
    if d.empty:
        continue

    total_spend = d["Spend"].sum()
    flight_days = (end - start).days + 1

    mid_point = start + timedelta(days=int(flight_days * 0.5))
    d_mid = d[d["Date"] <= mid_point]

    spend_mid = d_mid["Spend"].sum()
    days_elapsed = (mid_point - start).days + 1
    expected_mid = budget * (days_elapsed / flight_days)

    pace_ratio = spend_mid / expected_mid if expected_mid > 0 else 0
    velocity_7d = d_mid.sort_values("Date").tail(7)["Spend"].mean()

    final_ratio = total_spend / budget

    if final_ratio > 1.05:
        label = "Overdelivered"
    elif final_ratio < 0.95:
        label = "Underdelivered"
    else:
        label = "On Track"

    rows.append([
        cid, dsp, budget, flight_days,
        days_elapsed, spend_mid,
        pace_ratio, velocity_7d, label
    ])

ml_df = pd.DataFrame(rows, columns=[
    "Campaign_ID","DSP","Total_Budget","Flight_Days",
    "Days_Elapsed","Spend_to_Date",
    "Pace_Ratio","Spend_Velocity","Final_Label"
])

# ==================================================
# STEP 2: TRAIN ML MODEL
# ==================================================
le_dsp = LabelEncoder()
le_label = LabelEncoder()

ml_df["DSP_enc"] = le_dsp.fit_transform(ml_df["DSP"])
ml_df["Label_enc"] = le_label.fit_transform(ml_df["Final_Label"])

features = [
    "DSP_enc","Total_Budget","Flight_Days",
    "Days_Elapsed","Spend_to_Date",
    "Pace_Ratio","Spend_Velocity"
]

X = ml_df[features]
y = ml_df["Label_enc"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    random_state=42
)
model.fit(X_train, y_train)

ml_accuracy = accuracy_score(y_test, model.predict(X_test))

# ==================================================
# STEP 3: APPLY ML TO LIVE CAMPAIGNS
# ==================================================
live = meta[meta["Flight_End_Date"] >= YESTERDAY]
pred_rows = []

for _, c in live.iterrows():
    cid = c["Campaign_ID"]
    dsp = c["DSP"]
    budget = float(c["Total_Budget"])
    start = c["Flight_Start_Date"]
    end = c["Flight_End_Date"]

    d = daily[daily["Campaign_ID"] == cid]
    if d.empty:
        continue

    spend = d["Spend"].sum()
    days_elapsed = (YESTERDAY - start).days + 1
    flight_days = (end - start).days + 1
    expected = budget * (days_elapsed / flight_days)

    pace_ratio = spend / expected if expected > 0 else 0
    velocity_7d = d.sort_values("Date").tail(7)["Spend"].mean()

    X_live = pd.DataFrame([[
        le_dsp.transform([dsp])[0],
        budget, flight_days,
        days_elapsed, spend,
        pace_ratio, velocity_7d
    ]], columns=features)

    pred = model.predict(X_live)[0]
    pred_label = le_label.inverse_transform([pred])[0]

    pred_rows.append([cid, pred_label])

ml_preds = pd.DataFrame(pred_rows, columns=["Campaign_ID","ML_Prediction"])

# ==================================================
# STEP 4: SAFE MERGE (FIX FOR Flight_Start_Date ERROR)
# ==================================================
comparison = (
    excel_pacing
    .merge(
        meta[[
            "Campaign_ID",
            "Flight_Start_Date",
            "Flight_End_Date",
            "Total_Budget",
            "DSP"
        ]],
        on="Campaign_ID",
        how="left",
        suffixes=("", "_meta")
    )
    .merge(ml_preds, on="Campaign_ID", how="left")
)

# Force correct column names if suffixed
if "Flight_Start_Date_meta" in comparison.columns:
    comparison["Flight_Start_Date"] = comparison["Flight_Start_Date_meta"]

if "Flight_End_Date_meta" in comparison.columns:
    comparison["Flight_End_Date"] = comparison["Flight_End_Date_meta"]

# Ensure datetime
comparison["Flight_Start_Date"] = pd.to_datetime(comparison["Flight_Start_Date"]).dt.normalize()
comparison["Flight_End_Date"] = pd.to_datetime(comparison["Flight_End_Date"]).dt.normalize()

# ==================================================
# STEP 5: RISK CALCULATION
# ==================================================
comparison["Budget_At_Risk"] = np.where(
    comparison["ML_Prediction"] == "Overdelivered",
    comparison["Total_Budget"] - comparison["Spend_to_Date"],
    0
)

comparison["ML_Early_Warning"] = np.where(
    (comparison["ML_Prediction"].isin(["Overdelivered","Underdelivered"])) &
    (~comparison["Pacing_Status"].isin(["Overpacing","Underpacing"])),
    "YES",
    "NO"
)

# ==================================================
# STEP 6: FEATURE IMPORTANCE
# ==================================================
feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

# ==================================================
# STEP 7: WRITE OUTPUT
# ==================================================
with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    comparison.to_excel(writer, sheet_name="Excel_vs_ML", index=False)
    feature_importance.to_excel(writer, sheet_name="Feature_Importance", index=False)
    pd.DataFrame({
        "Metric": [
            "ML Validation Accuracy",
            "Total Budget At Risk",
            "LAST_REFRESH_UTC"
        ],
        "Value": [
            round(ml_accuracy, 3),
            round(comparison["Budget_At_Risk"].sum(), 2),
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        ]
    }).to_excel(writer, sheet_name="Exec_Summary", index=False)

print("✅ PaceSmart pipeline executed successfully")
print(f"📂 Output written to: {OUTPUT_FILE}")
