import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="PaceSmart | Predictive Pacing Intelligence",
    layout="wide"
)

BASE_PATH = "."
DATA_FILE = f"{BASE_PATH}/PaceSmart_ML_vs_Excel_Output.xlsx"

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    comparison = pd.read_excel(DATA_FILE, sheet_name="Excel_vs_ML")
    features = pd.read_excel(DATA_FILE, sheet_name="Feature_Importance")
    summary = pd.read_excel(DATA_FILE, sheet_name="Exec_Summary")

    # Clean column names (prevents hidden space errors)
    comparison.columns = comparison.columns.str.strip()
    features.columns = features.columns.str.strip()
    summary.columns = summary.columns.str.strip()

    return comparison, features, summary

comparison, features, summary = load_data()

# -------------------------------------------------
# SAFE DERIVED FIELDS
# -------------------------------------------------
today_utc = datetime.now(timezone.utc).date()

if "Campaign_Status" not in comparison.columns:
    comparison["Campaign_Status"] = np.where(
        pd.to_datetime(comparison["Flight_End_Date"]).dt.date >= today_utc,
        "LIVE",
        "ENDED"
    )

comparison["Remaining_Budget"] = (
    comparison["Total_Budget"] - comparison["Spend_to_Date"]
)

# Fill numeric safety
safe_numeric_cols = [
    "Predicted_Final_Deviation_%",
    "Risk_Score",
    "Predicted_Impact_Amount"
]

for col in safe_numeric_cols:
    if col in comparison.columns:
        comparison[col] = comparison[col].fillna(0)
    else:
        comparison[col] = 0

if "Risk_Level" in comparison.columns:
    comparison["Risk_Level"] = comparison["Risk_Level"].fillna("LOW – Stable")
else:
    comparison["Risk_Level"] = "LOW – Stable"

# -------------------------------------------------
# ML EXPLANATION LOGIC
# -------------------------------------------------
def explain_ml(row):
    reasons = []

    if row["Predicted_Final_Deviation_%"] > 5:
        reasons.append("Model predicts overspend risk")

    if row["Predicted_Final_Deviation_%"] < -5:
        reasons.append("Model predicts underdelivery risk")

    if row["Risk_Score"] >= 75:
        reasons.append("High severity deviation expected")

    if row["Campaign_Status"] == "LIVE" and row["Risk_Level"] != "LOW – Stable":
        reasons.append("Early warning before Excel breach")

    return "; ".join(reasons) if reasons else "No abnormal pattern detected"

comparison["Why_ML_Flagged"] = comparison.apply(explain_ml, axis=1)

# -------------------------------------------------
# SAFE METRIC HELPER
# -------------------------------------------------
def safe_sum(df, column):
    return df[column].sum() if column in df.columns else 0

def safe_metric(metric_name):
    result = summary.loc[summary["Metric"] == metric_name, "Value"]
    return result.values[0] if not result.empty else "N/A"

last_refresh = safe_metric("LAST_REFRESH_UTC")
ml_mae = safe_metric("ML Validation MAE")

# -------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------
page = st.sidebar.radio(
    "Navigate",
    [
        "Executive Summary",
        "Excel Pacing (Rule-Based)",
        "Predictive Risk View (ML)",
        "ML Feature Importance",
        "How ML Works"
    ]
)

# =================================================
# PAGE 1: EXECUTIVE SUMMARY
# =================================================
if page == "Executive Summary":
    st.title("📊 PaceSmart – Executive Summary")

    status_filter = st.radio(
        "Campaign Status",
        ["All", "LIVE", "ENDED"],
        horizontal=True
    )

    if status_filter == "LIVE":
        filtered = comparison[comparison["Campaign_Status"] == "LIVE"]
    elif status_filter == "ENDED":
        filtered = comparison[comparison["Campaign_Status"] == "ENDED"]
    else:
        filtered = comparison.copy()

    total_campaigns = len(filtered)
    live_campaigns = (filtered["Campaign_Status"] == "LIVE").sum()
    ended_campaigns = (filtered["Campaign_Status"] == "ENDED").sum()

    total_budget = safe_sum(filtered, "Total_Budget")
    total_spend = safe_sum(filtered, "Spend_to_Date")
    total_remaining = safe_sum(filtered, "Remaining_Budget")
    total_predicted_impact = safe_sum(filtered, "Predicted_Impact_Amount")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Campaigns", total_campaigns)
    c2.metric("Live Campaigns", live_campaigns)
    c3.metric("Ended Campaigns", ended_campaigns)
    c4.metric("ML Accuracy (MAE)", f"{ml_mae}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Total Budget", f"${total_budget:,.0f}")
    c6.metric("Spend to Date", f"${total_spend:,.0f}")
    c7.metric("Remaining Budget", f"${total_remaining:,.0f}")
    c8.metric("Predicted Financial Impact", f"${total_predicted_impact:,.0f}")

    st.info(f"🕒 Last refreshed (UTC): {last_refresh}")

# =================================================
# PAGE 2: EXCEL VIEW
# =================================================
elif page == "Excel Pacing (Rule-Based)":
    st.title("📘 Excel Pacing – Current Status")

    st.dataframe(
        comparison[[
            "Campaign_ID",
            "DSP",
            "Campaign_Status",
            "Flight_Start_Date",
            "Flight_End_Date",
            "Total_Budget",
            "Spend_to_Date",
            "Expected_Spend_Till_Date",
            "Remaining_Budget",
            "Pacing_Status"
        ]],
        use_container_width=True
    )

# =================================================
# PAGE 3: ML VIEW
# =================================================
elif page == "Predictive Risk View (ML)":
    st.title("🔮 Predictive Risk Intelligence")

    st.dataframe(
        comparison[[
            "Campaign_ID",
            "DSP",
            "Campaign_Status",
            "Pacing_Status",
            "Predicted_Final_Deviation_%",
            "Risk_Score",
            "Risk_Level",
            "Predicted_Impact_Amount",
            "Why_ML_Flagged"
        ]].sort_values("Risk_Score", ascending=False),
        use_container_width=True
    )

# =================================================
# PAGE 4: FEATURE IMPORTANCE
# =================================================
elif page == "ML Feature Importance":
    st.title("📈 What the ML Model Looks At")
    st.bar_chart(features.set_index("Feature")["Importance"])

# =================================================
# PAGE 5: ML EXPLANATION
# =================================================
elif page == "How ML Works":
    st.title("🧠 How ML Adds Value Over Excel")

    st.markdown("""
### Excel
- Assumes linear spend
- Flags only after deviation
- Does not learn from historical patterns

### ML (Behavioral Regression Model)
- Learns from ended campaigns
- Uses time %, budget %, acceleration & gap logic
- Predicts final deviation %
- Converts deviation into risk score
- Quantifies financial impact
- Flags early before Excel breach

### Risk Score
- 0–100 severity scale
- Helps prioritize campaigns

### Financial Impact
- Converts predicted deviation into monetary exposure
""")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.caption("PaceSmart augments Excel with predictive intelligence. Final decisions remain human-led.")
