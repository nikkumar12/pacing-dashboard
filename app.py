import streamlit as st
import pandas as pd
st.cache_data.clear()
from datetime import datetime, timezone

# ==================================================
# CONFIG
# ==================================================
st.set_page_config(
    page_title="PaceSmart Dashboard",
    layout="wide"
)

DATA_FILE = "PaceSmart_ML_vs_Excel_Output.xlsx"

# ==================================================
# LOAD DATA
# ==================================================
@st.cache_data
def load_data():
    comparison = pd.read_excel(DATA_FILE, sheet_name="Excel_vs_ML")
    features = pd.read_excel(DATA_FILE, sheet_name="Feature_Importance")
    summary = pd.read_excel(DATA_FILE, sheet_name="Exec_Summary")
    return comparison, features, summary

comparison, features, summary = load_data()

# ==================================================
# LAST REFRESH (UTC ONLY)
# ==================================================
last_refresh_utc = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")

# ==================================================
# CORE METRICS (STRICT – NO GUESSING)
# ==================================================
total_campaigns = comparison.shape[0]
live_campaigns = (comparison["Campaign_Status"] == "LIVE").sum()
ended_campaigns = (comparison["Campaign_Status"] == "ENDED").sum()

total_budget = comparison["Total_Budget"].sum()
spend_to_date = comparison["Spend_to_Date"].sum()
remaining_budget = total_budget - spend_to_date

ml_accuracy = summary.loc[
    summary["Metric"] == "ML Validation Accuracy", "Value"
].values[0]

budget_at_risk = summary.loc[
    summary["Metric"] == "Total Budget At Risk", "Value"
].values[0]

# ==================================================
# SIDEBAR NAVIGATION
# ==================================================
st.sidebar.title("Navigate")
page = st.sidebar.radio(
    "",
    [
        "Executive Summary",
        "Excel Pacing (No ML)",
        "Excel vs ML Decision View",
        "How ML Works & Why It Helps"
    ]
)

# ==================================================
# PAGE 1: EXECUTIVE SUMMARY
# ==================================================
if page == "Executive Summary":
    st.title("📊 PaceSmart – Executive Summary")
    st.caption(f"🕒 Last refreshed: {last_refresh_utc}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Campaigns", total_campaigns)
    col2.metric("Live Campaigns", live_campaigns)
    col3.metric("Ended Campaigns", ended_campaigns)
    col4.metric("ML Accuracy", f"{ml_accuracy * 100:.2f}%")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Total Budget", f"${total_budget:,.0f}")
    col6.metric("Spend to Date", f"${spend_to_date:,.0f}")
    col7.metric("Remaining Budget", f"${remaining_budget:,.0f}")
    col8.metric("Budget at Risk (ML)", f"${budget_at_risk:,.0f}")

    st.markdown("### What this means")
    st.markdown("""
- **Excel pacing** shows where campaigns stand **today**
- **ML pacing** predicts where campaigns are likely to **end**
- **ML flags risk earlier**, not after damage is done
- Predictions are based on **historical delivery behavior**
""")

# ==================================================
# PAGE 2: EXCEL PACING (NO ML) — UNTOUCHED
# ==================================================
elif page == "Excel Pacing (No ML)":
    st.title("📄 Excel Pacing (No ML)")
    st.caption("Pure Excel logic. No machine learning applied.")

    excel_only_cols = [
        "Campaign_ID",
        "DSP",
        "Campaign_Status",
        "Spend_to_Date",
        "Total_Budget",
        "Pacing_Status"
    ]

    st.dataframe(
        comparison[excel_only_cols].sort_values(
            ["Campaign_Status", "Spend_to_Date"],
            ascending=[True, False]
        ),
        use_container_width=True
    )

# ==================================================
# PAGE 3: EXCEL vs ML (WITH REASONS)
# ==================================================
elif page == "Excel vs ML Decision View":
    st.title("🤖 Excel vs ML – Decision View")

    def ml_reason(row):
        if row["ML_Prediction"] == "Overdelivered":
            return "Historical overspend patterns + high recent spend velocity"
        elif row["ML_Prediction"] == "Underdelivered":
            return "Historical underspend patterns + weak recent delivery"
        else:
            return "Delivery aligns with historical norms"

    comparison["ML_Reason"] = comparison.apply(ml_reason, axis=1)

    decision_cols = [
        "Campaign_ID",
        "DSP",
        "Campaign_Status",
        "Spend_to_Date",
        "Total_Budget",
        "Pacing_Status",
        "ML_Prediction",
        "ML_Early_Warning",
        "Budget_At_Risk",
        "ML_Reason"
    ]

    st.dataframe(
        comparison[decision_cols].sort_values(
            "Budget_At_Risk", ascending=False
        ),
        use_container_width=True
    )

# ==================================================
# PAGE 4: HOW ML WORKS
# ==================================================
elif page == "How ML Works & Why It Helps":
    st.title("🧠 How ML Works & Why It Helps")

    st.markdown("""
### How the model learns
- Trained on **ended campaigns**
- Learns pacing drift, spend velocity, and delivery shape
- Uses Random Forest to recognize **patterns, not rules**

### Why ML adds value
| Excel | ML |
|------|----|
| Linear pacing | Pattern-based |
| Looks at today | Predicts final outcome |
| Reacts late | Flags risk early |

### When ML disagrees with Excel
- Campaign looks *On Track today*
- Historical twins ended badly
- ML warns **before budget is lost**
""")

    st.markdown("### Top ML Signals")
    st.dataframe(features, use_container_width=True)

# ==================================================
# FOOTER
# ==================================================
st.markdown("---")
st.caption(
    "PaceSmart combines Excel discipline with machine-learning foresight. "
    "Excel shows status. ML predicts outcome. Humans decide."
)
