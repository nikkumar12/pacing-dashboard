import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="PaceSmart | Excel vs ML Pacing",
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

# Safe fill if missing predictions
comparison["Predicted_Final_Deviation_%"] = comparison["Predicted_Final_Deviation_%"].fillna(0)
comparison["Risk_Score"] = comparison["Risk_Score"].fillna(0)
comparison["Predicted_Impact_Amount"] = comparison["Predicted_Impact_Amount"].fillna(0)
comparison["Risk_Level"] = comparison["Risk_Level"].fillna("LOW – Stable")

# -------------------------------------------------
# EXPLANATION LOGIC
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
# EXECUTIVE METRICS
# -------------------------------------------------
total_campaigns = len(comparison)
live_campaigns = (comparison["Campaign_Status"] == "LIVE").sum()
ended_campaigns = (comparison["Campaign_Status"] == "ENDED").sum()

total_budget = comparison["Total_Budget"].sum()
total_spend = comparison["Spend_to_Date"].sum()
total_remaining = comparison["Remaining_Budget"].sum()
total_risk = comparison["Predicted_Impact_Amount"].sum()

ml_mae = summary.loc[
    summary["Metric"] == "ML Validation MAE", "Value"
].values[0]

last_refresh = summary.loc[
    summary["Metric"] == "LAST_REFRESH_UTC", "Value"
].values[0]

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
page = st.sidebar.radio(
    "Navigate",
    [
        "Executive Summary",
        "Excel Pacing (No ML)",
        "Excel vs ML Decision View",
        "ML Feature Importance",
        "How ML Works"
    ]
)

# =================================================
# PAGE 1: EXECUTIVE SUMMARY
# =================================================
if page == "Executive Summary":
    st.title("📊 PaceSmart – Executive Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Campaigns", total_campaigns)
    c2.metric("Live Campaigns", live_campaigns)
    c3.metric("Ended Campaigns", ended_campaigns)
    c4.metric("ML Validation MAE", round(float(ml_mae), 4))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Total Budget", f"${total_budget:,.0f}")
    c6.metric("Spend to Date", f"${total_spend:,.0f}")
    c7.metric("Remaining Budget", f"${total_remaining:,.0f}")
    c8.metric("Predicted Impact (ML)", f"${total_risk:,.0f}")

    st.info(f"🕒 Last refreshed (UTC): {last_refresh}")

# =================================================
# PAGE 2: EXCEL ONLY
# =================================================
elif page == "Excel Pacing (No ML)":
    st.title("📘 Excel Pacing (Rule-Based)")

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
# PAGE 3: EXCEL vs ML
# =================================================
elif page == "Excel vs ML Decision View":
    st.title("⚖️ Excel vs ML – Predictive Risk View")

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

    st.bar_chart(
        features.set_index("Feature")["Importance"]
    )

# =================================================
# PAGE 5: ML EXPLANATION
# =================================================
elif page == "How ML Works":
    st.title("🧠 How ML Adds Value Over Excel")

    st.markdown("""
### Excel
- Assumes linear spend
- Flags only after deviation
- Cannot learn from history

### ML (Regression Risk Engine)
- Learns from ended campaigns
- Predicts final deviation %
- Assigns risk score (0–100)
- Quantifies financial impact
- Flags early before Excel breach

### Risk Score
- Based on predicted deviation severity
- Helps prioritize action

### Impact
- Converts % deviation into $ value
- Executive decision ready
""")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.caption("PaceSmart augments Excel with predictive intelligence. Decisions remain human-led.")
