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

if "Risk_Level" not in comparison.columns:
    comparison["Risk_Level"] = "LOW – Stable"
else:
    comparison["Risk_Level"] = comparison["Risk_Level"].fillna("LOW – Stable")

# -------------------------------------------------
# STYLING FUNCTIONS
# -------------------------------------------------
def highlight_risk_level(val):
    if isinstance(val, str):
        if "CRITICAL" in val.upper():
            return "background-color: #c00000; color: white;"
        elif "HIGH" in val.upper():
            return "background-color: #ff6600; color: white;"
        elif "MODERATE" in val.upper():
            return "background-color: #ffd966;"
        elif "LOW" in val.upper():
            return "background-color: #92d050;"
    return ""

def highlight_pacing(val):
    if isinstance(val, str):
        if "OVER" in val.upper():
            return "background-color: #c00000; color: white;"
        elif "UNDER" in val.upper():
            return "background-color: #ff6600; color: white;"
        elif "TRACK" in val.upper():
            return "background-color: #92d050;"
    return ""

def risk_score_gradient(val):
    if isinstance(val, (int, float)):
        if val >= 75:
            return "background-color: #c00000; color: white;"
        elif val >= 50:
            return "background-color: #ff6600; color: white;"
        elif val >= 25:
            return "background-color: #ffd966;"
        else:
            return "background-color: #92d050;"
    return ""

# -------------------------------------------------
# SAFE HELPERS
# -------------------------------------------------
def safe_sum(df, column):
    return df[column].sum() if column in df.columns else 0

def safe_metric(metric_name):
    result = summary.loc[summary["Metric"] == metric_name, "Value"]
    return result.values[0] if not result.empty else "N/A"

last_refresh = safe_metric("LAST_REFRESH_UTC")
ml_mae = safe_metric("ML Validation MAE")

# -------------------------------------------------
# SIDEBAR
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
    total_budget = safe_sum(filtered, "Total_Budget")
    total_spend = safe_sum(filtered, "Spend_to_Date")
    total_remaining = safe_sum(filtered, "Remaining_Budget")
    total_predicted_impact = safe_sum(filtered, "Predicted_Impact_Amount")

    # KPI Severity Indicator
    if total_predicted_impact > 1_000_000:
        impact_color = "🔴"
    elif total_predicted_impact > 500_000:
        impact_color = "🟠"
    else:
        impact_color = "🟢"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Campaigns", total_campaigns)
    c2.metric("Total Budget", f"${total_budget:,.0f}")
    c3.metric("Spend to Date", f"${total_spend:,.0f}")
    c4.metric("Remaining Budget", f"${total_remaining:,.0f}")

    c5, c6 = st.columns(2)
    c5.metric("ML Accuracy (MAE)", f"{ml_mae}")
    c6.metric(
        "Predicted Financial Impact",
        f"{impact_color} ${total_predicted_impact:,.0f}"
    )

    st.info(f"🕒 Last refreshed (UTC): {last_refresh}")

    # Risk Distribution Chart
    st.subheader("📊 Portfolio Risk Distribution")
    risk_dist = filtered["Risk_Level"].value_counts()
    st.bar_chart(risk_dist)

    # Top 10 High Risk
    st.subheader("🔥 Top 10 High Risk Campaigns")
    top10 = filtered.sort_values("Risk_Score", ascending=False).head(10)
    st.dataframe(
        top10[[
            "Campaign_ID",
            "DSP",
            "Risk_Score",
            "Risk_Level",
            "Predicted_Impact_Amount"
        ]],
        use_container_width=True
    )

# =================================================
# PAGE 2: EXCEL VIEW
# =================================================
elif page == "Excel Pacing (Rule-Based)":

    st.title("📘 Excel Pacing – Current Status")

    excel_view = comparison[[
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
    ]]

    styled_excel = excel_view.style.applymap(
        highlight_pacing,
        subset=["Pacing_Status"]
    )

    st.dataframe(styled_excel, use_container_width=True)

# =================================================
# PAGE 3: ML VIEW
# =================================================
elif page == "Predictive Risk View (ML)":

    st.title("🔮 Predictive Risk Intelligence")

    ml_view = comparison[[
        "Campaign_ID",
        "DSP",
        "Campaign_Status",
        "Pacing_Status",
        "Predicted_Final_Deviation_%",
        "Risk_Score",
        "Risk_Level",
        "Predicted_Impact_Amount",
        "Why_ML_Flagged"
    ]].sort_values("Risk_Score", ascending=False)

    styled_ml = ml_view.style \
        .applymap(highlight_risk_level, subset=["Risk_Level"]) \
        .applymap(risk_score_gradient, subset=["Risk_Score"]) \
        .applymap(highlight_pacing, subset=["Pacing_Status"])

    st.dataframe(styled_ml, use_container_width=True)

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
- Linear projection
- Reactive alerts
- No historical learning

### PaceSmart ML
- Learns from ended campaigns
- Detects acceleration & behavioral shifts
- Predicts final deviation %
- Converts deviation to standardized risk score
- Quantifies financial exposure
- Enables proactive intervention
""")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.caption("PaceSmart augments Excel with predictive intelligence. Final decisions remain human-led.")
