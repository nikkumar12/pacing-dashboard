import streamlit as st
import pandas as pd
import numpy as np

# ==================================================
# STREAMLIT CONFIG
# ==================================================
st.set_page_config(
    page_title="PaceSmart | Predictive Pacing",
    layout="wide"
)

# ==================================================
# FILE PATH (GitHub / Streamlit Cloud friendly)
# ==================================================
DATA_FILE = "PaceSmart_ML_vs_Excel_Output.xlsx"

# ==================================================
# LOAD DATA
# ==================================================
@st.cache_data(show_spinner=False)
def load_data():
    comparison = pd.read_excel(DATA_FILE, sheet_name="Excel_vs_ML")
    summary = pd.read_excel(DATA_FILE, sheet_name="Exec_Summary")

    # Optional refresh info (if present)
    refresh_time = None
    try:
        refresh_df = pd.read_excel(DATA_FILE, sheet_name="Refresh_Info")
        refresh_time = refresh_df.loc[0, "Value"]
    except Exception:
        refresh_time = "Unknown (UTC)"

    return comparison, summary, refresh_time


df, summary, last_refresh_utc = load_data()

# ==================================================
# DERIVED FIELDS (SAFE)
# ==================================================
df["Campaign_Status"] = np.where(df["Days_Left"] > 0, "LIVE", "ENDED")
df["Remaining_Budget"] = df["Total_Budget"] - df["Spend_to_Date"]

# ==================================================
# EXECUTIVE METRICS
# ==================================================
total_campaigns = len(df)
live_campaigns = (df["Campaign_Status"] == "LIVE").sum()
ended_campaigns = (df["Campaign_Status"] == "ENDED").sum()

total_budget = df["Total_Budget"].sum()
total_spend = df["Spend_to_Date"].sum()
remaining_budget = df["Remaining_Budget"].sum()
budget_at_risk = df["Budget_At_Risk"].sum()

ml_accuracy = summary.loc[
    summary["Metric"] == "ML Validation Accuracy", "Value"
].values[0]

# ==================================================
# SIDEBAR NAVIGATION
# ==================================================
page = st.sidebar.radio(
    "Navigate",
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

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Campaigns", total_campaigns)
    c2.metric("Live Campaigns", live_campaigns)
    c3.metric("Ended Campaigns", ended_campaigns)
    c4.metric("ML Accuracy", f"{ml_accuracy:.2%}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Total Budget", f"${total_budget:,.0f}")
    c6.metric("Spend to Date", f"${total_spend:,.0f}")
    c7.metric("Remaining Budget", f"${remaining_budget:,.0f}")
    c8.metric("Budget at Risk (ML)", f"${budget_at_risk:,.0f}")

    st.markdown("""
### What this means
- **Excel pacing** shows where campaigns stand *today*
- **ML pacing** predicts where campaigns are likely to *end*
- ML flags risk **earlier**, before Excel thresholds are crossed
- Confidence comes from **historical campaign patterns**
""")

# ==================================================
# PAGE 2: EXCEL ONLY VIEW
# ==================================================
elif page == "Excel Pacing (No ML)":
    st.title("📘 Excel Pacing (Rule-Based Only)")

    st.dataframe(
        df[[
            "Campaign_ID",
            "DSP",
            "Campaign_Status",
            "Flight_Start_Date",
            "Flight_End_Date",
            "Total_Budget",
            "Spend_to_Date",
            "Expected_Spend_Till_Date",
            "Remaining_Budget",
            "Pacing_%_vs_Ideal",
            "Pacing_Status"
        ]],
        use_container_width=True
    )

# ==================================================
# PAGE 3: EXCEL vs ML VIEW
# ==================================================
elif page == "Excel vs ML Decision View":
    st.title("⚖️ Excel vs ML – Decision Comparison")

    st.dataframe(
        df[[
            "Campaign_ID",
            "DSP",
            "Campaign_Status",
            "Total_Budget",
            "Spend_to_Date",
            "Remaining_Budget",
            "Pacing_Status",
            "ML_Prediction",
            "Budget_At_Risk",
            "ML_Early_Warning"
        ]].sort_values("Budget_At_Risk", ascending=False),
        use_container_width=True
    )

# ==================================================
# PAGE 4: ML EXPLANATION
# ==================================================
elif page == "How ML Works & Why It Helps":
    st.title("🧠 How ML Works & Why It Helps")

    st.markdown("""
### How the ML model works
- Trained on **ended historical campaigns**
- Learns patterns of:
  - Spend velocity
  - Early pacing behavior
  - Budget exhaustion trends
- Predicts **final outcome**, not just today’s status

### Why Excel misses risk
- Assumes linear spend
- Reacts **after** deviation
- Cannot learn from history

### What ML adds
- Early warnings
- Risk prioritization
- Preventable budget loss detection

### Confidence & accuracy
- Accuracy shown is validation accuracy
- Predictions are **probabilistic**, not rule-based
- ML complements Excel — it does not replace it
""")

# ==================================================
# FOOTER
# ==================================================
st.markdown("---")
st.caption(
    "PaceSmart combines Excel discipline with machine learning foresight. "
    "Final decisions remain human-led."
)
