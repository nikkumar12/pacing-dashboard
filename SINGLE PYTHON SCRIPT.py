import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

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
# DATE NORMALIZATION
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
# STEP 1: BUILD TRAINING DATA (ENDED CAMPAIGNS)
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

    # 🎯 REGRESSION TARGET
    final_deviation = (total_spend - budget) / budget

    rows.append([
        cid, dsp, budget, flight_days,
        days_elapsed, spend_mid,
        pace_ratio, velocity_7d, final_deviation
    ])

ml_df = pd.DataFrame(rows, columns=[
    "Campaign_ID","DSP","Total_Budget","Flight_Days",
    "Days_Elapsed","Spend_to_Date",
    "Pace_Ratio","Spend_Velocity","Final_Deviation"
])

# ==================================================
# STEP 2: TRAIN REGRESSION MODEL
# ==================================================
le_dsp = LabelEncoder()
ml_df["DSP_enc"] = le_dsp.fit_transform(ml_df["DSP"])

features = [
    "DSP_enc","Total_Budget","Flight_Days",
    "Days_Elapsed","Spend_to_Date",
    "Pace_Ratio","Spend_Velocity"
]

X = ml_df[features]
y = ml_df["Final_Deviation"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = RandomForestRegressor(
    n_estimators=400,
    max_depth=7,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
ml_mae = mean_absolute_error(y_test, y_pred)

# ==================================================
# STEP 3: APPLY TO LIVE CAMPAIGNS
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

    predicted_deviation = model.predict(X_live)[0]
    predicted_pct = round(predicted_deviation * 100, 2)

    pred_rows.append([cid, predicted_pct])

ml_preds = pd.DataFrame(
    pred_rows,
    columns=["Campaign_ID","Predicted_Final_Deviation_%"]
)

# ==================================================
# STEP 4: SAFE MERGE (FIXES Total_Budget ERROR)
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

# Ensure correct Total_Budget column
if "Total_Budget_meta" in comparison.columns:
    comparison["Total_Budget"] = comparison["Total_Budget_meta"]

# ==================================================
# STEP 5: RISK ENGINE
# ==================================================
comparison["Risk_Score"] = (
    comparison["Predicted_Final_Deviation_%"].abs() / 20
) * 100

comparison["Risk_Score"] = comparison["Risk_Score"].clip(0, 100)

comparison["Predicted_Impact_Amount"] = (
    comparison["Predicted_Final_Deviation_%"] / 100
) * comparison["Total_Budget"]

conditions = [
    comparison["Risk_Score"] >= 70,
    comparison["Risk_Score"].between(40, 69),
    comparison["Risk_Score"] < 40
]

choices = [
    "CRITICAL – Immediate Action",
    "MODERATE – Monitor Closely",
    "LOW – Stable"
]

comparison["Risk_Level"] = np.select(conditions, choices)

comparison["Early_Warning"] = np.where(
    (comparison["Risk_Score"] >= 50) &
    (comparison["Pacing_Status"] == "On Track"),
    "YES",
    "NO"
)

# ==================================================
# FEATURE IMPORTANCE
# ==================================================
feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

# ==================================================
# WRITE OUTPUT
# ==================================================
with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    comparison.to_excel(writer, sheet_name="Excel_vs_ML", index=False)
    feature_importance.to_excel(writer, sheet_name="Feature_Importance", index=False)
    pd.DataFrame({
        "Metric": [
            "ML Validation MAE",
            "Total Predicted Impact",
            "LAST_REFRESH_UTC"
        ],
        "Value": [
            round(ml_mae, 4),
            round(comparison["Predicted_Impact_Amount"].sum(), 2),
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        ]
    }).to_excel(writer, sheet_name="Exec_Summary", index=False)

print("✅ PaceSmart Predictive Risk Engine executed successfully")
print(f"📂 Output written to: {OUTPUT_FILE}")
