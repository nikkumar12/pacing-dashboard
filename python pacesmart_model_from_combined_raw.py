import pandas as pd
import numpy as np
import os

# =====================================================
# 1. READ COMBINED RAW DATA
# =====================================================
BASE_DIR = r"C:\Users\nikkumar12\OneDrive - Publicis Groupe\Desktop\2026"
FILE_NAME = "PaceSmart_Excel_Pacing_With_Formulas.xlsx"
FILE_PATH = os.path.join(BASE_DIR, FILE_NAME)

print("\n📂 Reading Combined_Raw from:")
print(FILE_PATH)

df = pd.read_excel(FILE_PATH, sheet_name="Combined_Raw")
df = df.sort_values(["Campaign ID", "Date"])

print("\n✅ Data loaded for modeling")
print("Rows:", df.shape[0])

# =====================================================
# 2. FEATURE ENGINEERING (MODEL SIGNALS)
# =====================================================

# Trend: average spend last 5 days
df["Avg_Spend_Last_5"] = (
    df.groupby("Campaign ID")["Spend"]
      .transform(lambda x: x.rolling(5, min_periods=3).mean())
)

# Volatility: instability of spend
df["Spend_Volatility"] = (
    df.groupby("Campaign ID")["Spend"]
      .transform(lambda x: x.rolling(5, min_periods=3).std())
)

# Projected end-of-flight spend
df["Projected_End_Spend"] = df["Avg_Spend_Last_5"] * df["Total_Days"]

# =====================================================
# 3. MODEL PREDICTION (FORWARD LOOKING)
# =====================================================
def model_prediction(row):
    if pd.isna(row["Projected_End_Spend"]):
        return "Insufficient Data"
    if row["Projected_End_Spend"] > 1.1 * row["Total_Budget"]:
        return "Overspend"
    elif row["Projected_End_Spend"] < 0.9 * row["Total_Budget"]:
        return "Underspend"
    else:
        return "On Track"

df["Model_Prediction"] = df.apply(model_prediction, axis=1)

# =====================================================
# 4. EXCEL BASELINE
# =====================================================
df["Excel_Status"] = df["Pacing_Status"]

# =====================================================
# 5. WHY MODEL IS DIFFERENT (COMMENTARY)
# =====================================================
def explain_difference(row):
    reasons = []

    if row["Excel_Status"] == "On Track" and row["Model_Prediction"] != "On Track":
        reasons.append("Recent spend trend projects future risk")

    if row["Spend_Volatility"] > 0.25 * row["Spend"]:
        reasons.append("Spend pattern is volatile")

    if row["Day_of_Flight"] <= 0.3 * row["Total_Days"]:
        reasons.append("Early-flight trend likely to continue")

    if row["Day_of_Flight"] >= 0.7 * row["Total_Days"]:
        reasons.append("Late-flight deviation hard to recover")

    return " | ".join(reasons) if reasons else "Excel and model agree"

df["Why_Model_Differs"] = df.apply(explain_difference, axis=1)

# =====================================================
# 6. SAVE OUTPUT
# =====================================================
OUTPUT_FILE = os.path.join(BASE_DIR, "PaceSmart_Model_vs_Excel_FINAL.xlsx")
df.to_excel(OUTPUT_FILE, index=False)

print("\n✅ Model vs Excel comparison file created:")
print(OUTPUT_FILE)
