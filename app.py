import streamlit as st
import pandas as pd

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="PaceSmart | Excel vs ML Pacing",
    layout="wide"
)

BASE_PATH = r"C:\Users\nikkumar12\PyCharmMiscProject"
DATA_FILE = f"{BASE_PATH}\\PaceSmart_ML_vs_Excel_Output.xlsx"

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
# HEADER
# -------------------------------------------------
st.title("📊 PaceSmart: Excel vs ML Pacing Dashboard")
st.caption("Predictive pacing vs rule-based pacing • Data till yesterday")

# -------------------------------------------------
# EXEC SUMMARY
# -------------------------------------------------
st.subheader("🧠 Executive Summary")

col1, col2, col3, col4, col5 = st.columns(5)

metrics = dict(zip(summary["Metric"], summary["Value"]))

col1.metric("ML Accuracy", f"{metrics['ML Validation Accuracy']:.2%}")
col2.metric("Excel Alerts", int(metrics["Excel Alerts"]))
col3.metric("ML Risk Alerts", int(metrics["ML Risk Alerts"]))
col4.metric("Early Warnings", int(metrics["Early Warnings"]))
col5.metric("Budget at Risk", f"${metrics['Total Budget At Risk']:,.0f}")

st.markdown("---")

# -------------------------------------------------
# FILTERS
# -------------------------------------------------
st.subheader("🎛 Filters")

col1, col2, col3 = st.columns(3)

dsp_filter = col1.multiselect(
    "DSP",
    options=sorted(comparison["DSP"].dropna().unique()),
    default=sorted(comparison["DSP"].dropna().unique())
)

excel_filter = col2.multiselect(
    "Excel Pacing Status",
    options=sorted(comparison["Pacing_Status"].dropna().unique()),
    default=sorted(comparison["Pacing_Status"].dropna().unique())
)

ml_filter = col3.multiselect(
    "ML Prediction",
    options=sorted(comparison["ML_Prediction"].dropna().unique()),
    default=sorted(comparison["ML_Prediction"].dropna().unique())
)

filtered = comparison[
    (comparison["DSP"].isin(dsp_filter)) &
    (comparison["Pacing_Status"].isin(excel_filter)) &
    (comparison["ML_Prediction"].isin(ml_filter))
]

# -------------------------------------------------
# EXCEL vs ML TABLE
# -------------------------------------------------
st.subheader("📋 Excel vs ML Campaign Comparison")

st.dataframe(
    filtered[[
        "Campaign_ID",
        "DSP",
        "Pacing_Status",
        "ML_Prediction",
        "ML_Early_Warning",
        "Spend_to_Date",
        "Total_Budget",
        "Budget_At_Risk"
    ]].sort_values("Budget_At_Risk", ascending=False),
    use_container_width=True
)

# -------------------------------------------------
# EARLY WARNING VIEW
# -------------------------------------------------
st.subheader("🚨 ML Early Warnings (Excel Missed)")

early = filtered[filtered["ML_Early_Warning"] == "YES"]

st.dataframe(
    early[[
        "Campaign_ID",
        "DSP",
        "Pacing_Status",
        "ML_Prediction",
        "Budget_At_Risk"
    ]].sort_values("Budget_At_Risk", ascending=False),
    use_container_width=True
)

# -------------------------------------------------
# FEATURE IMPORTANCE
# -------------------------------------------------
st.subheader("🔍 Why ML Flagged These Campaigns")

st.bar_chart(
    features.set_index("Feature")["Importance"]
)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.caption(
    "Excel pacing = reactive | ML pacing = predictive | Built for PaceSmart-style decisioning"
)
