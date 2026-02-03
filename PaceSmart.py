import pandas as pd
import numpy as np

# =========================
# LOAD EXCEL PACING FILE
# =========================
file_path = r"C:\Users\nikkumar12\OneDrive - Publicis Groupe\Desktop\2026\PaceSmart_Excel_Pacing_With_Formulas.xlsx"
df = pd.read_excel(file_path, sheet_name="Excel_Pacing_With_Formulas")

df = df.sort_values(["Campaign ID", "Date"])

# =========================
# FEATURE ENGINEERING
# =========================

# Avg spend last 5 days
df["Avg_Spend_Last_5"] = (
    df.groupby("Campaign ID")["Spend"]
      .transform(lambda x: x.rolling(5, min_periods=3).mean())
)

# Spend volatility (instability)
df["Spend_Volatility"] = (
    df.groupby("Campaign ID")["Spend"]
      .transform(lambda x: x.rolling(5, min_periods=3).std())
)

# Projected end spend
df["Projected_End_Spend"] = df["Avg_Spend_Last_5"] * df["Total Days"]

# =========================
# MODEL DECISION (RULE-BASED BUT REAL)
# =========================
def model_prediction(row):
    if row["Projected_End_Spend"] > 1.1 * row["Total Budget"]:
        return "Overspend"
    elif row["Projected_End_Spend"] < 0.9 * row["Total Budget"]:
        return "Underspend"
    else:
        return "On Track"

df["Model_Prediction"] = df.apply(model_prediction, axis=1)

# =========================
# EXCEL STATUS (BASELINE)
# =========================
df["Excel_Status"] = df["Pacing Status"]

# =========================
# WHY MODEL IS DIFFERENT (COMMENTS)
# =========================
def reason_comment(row):
    reasons = []

    if row["Excel_Status"] == "On Track" and row["Model_Prediction"] != "On Track":
        reasons.append("Recent spend trend deviates from ideal pace")

    if row["Spend_Volatility"] > row["Spend"] * 0.25:
        reasons.append("Spend pattern is volatile")

    if row["Day of Flight"] <= 0.3 * row["Total Days"]:
        reasons.append("Early flight trend indicates future risk")

    if row["Day of Flight"] >= 0.7 * row["Total Days"]:
        reasons.append("Late flight deviation is hard to recover")

    if not reasons:
        return "Excel and model agree"

    return "; ".join(reasons)

df["Why_Model_Differs"] = df.apply(reason_comment, axis=1)

# =========================
# SAVE OUTPUT
# =========================
output_path = r"C:\Users\nikkumar12\OneDrive - Publicis Groupe\Desktop\2026\PaceSmart_Model_vs_Excel.xlsx"
df.to_excel(output_path, index=False)

print("✅ Model vs Excel comparison file created")
print(output_path)
