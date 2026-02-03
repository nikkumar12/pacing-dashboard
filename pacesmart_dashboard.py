import streamlit as st
import pandas as pd
import os

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="PaceSmart – Excel vs Model",
    layout="wide"
)

st.title("📊 PaceSmart: Excel Pacing vs Predictive Model")
st.caption("Comparing reactive Excel pacing with forward-looking predictions")

# =====================================================
# CLOUD-SAFE DATA LOADING (IMPORTANT)
# =====================================================
BASE_DIR = os.path.dirname(__file__)           # folder where this file lives
DATA_DIR = os.path.join(BASE_DIR, "sample_data")
FILE_NAME = "PaceSmart_Model_vs_Excel_FINAL.xlsx"
FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)

st.write("📂 Data source (repo relative path):")
st.code(FILE_PATH)

if not os.path.exists(FILE_PATH):
    st.error("❌ Data file not found. Ensure it exists in /sample_data/")
    st.stop()

df = pd.read_excel(FILE_PATH)

# =====================================================
# KPI SECTION
# =====================================================
st.subheader("📌 Overall Snapshot")

col1, col2, col3, col4 = st.columns(4)

total_rows = len(df)
excel_ontrack = (df["Excel_Status"] == "On Track").sum()
model_flagged = (
    (df["Excel_Status"] == "On Track") &
    (df["Model_Prediction"] != "On Track")
).sum()
agree = (df["Excel_Status"] == df["Model_Prediction"]).sum()

col1.metric("Total Campaign Days", total_rows)
col2.metric("Excel On Track", excel_ontrack)
col3.metric("Model Flagged Early Risk", model_flagged)
col4.metric("Excel & Model Agree", agree)

st.divider()

# =====================================================
# FILTERS
# =====================================================
st.subheader("🔍 Filters")

f1, f2, f3 = st.columns(3)

campaign = f1.selectbox(
    "Campaign",
    ["All"] + sorted(df["Campaign ID"].unique())
)

model_status = f2.selectbox(
    "Model Prediction",
    ["All", "Overspend", "Underspend", "On Track"]
)

excel_status = f3.selectbox(
    "Excel Status",
    ["All", "Overpacing", "Underpacing", "On Track"]
)

filtered = df.copy()

if campaign != "All":
    filtered = filtered[filtered["Campaign ID"] == campaign]

if model_status != "All":
    filtered = filtered[filtered["Model_Prediction"] == model_status]

if excel_status != "All":
    filtered = filtered[filtered["Excel_Status"] == excel_status]

# =====================================================
# HIGHLIGHT LOGIC
# =====================================================
def highlight_diff(row):
    if row["Excel_Status"] == "On Track" and row["Model_Prediction"] != "On Track":
        return ["background-color:#ffe6e6"] * len(row)
    return [""] * len(row)

# =====================================================
# MAIN TABLE
# =====================================================
st.subheader("📋 Excel vs Model – With Explanation")

display_cols = [
    "Date",
    "Campaign ID",
    "Excel_Status",
    "Model_Prediction",
    "Why_Model_Differs",
    "Pacing_Rate",
    "Avg_Spend_Last_5",
    "Spend_Volatility",
    "Projected_End_Spend",
    "Total_Budget"
]

st.dataframe(
    filtered[display_cols]
        .style.apply(highlight_diff, axis=1),
    use_container_width=True,
    height=520
)

st.caption("🔴 Highlighted rows = Excel says *On Track* but model flags **future risk**")

# =====================================================
# EXPLANATION: SPEND VOLATILITY
# =====================================================
st.divider()

st.subheader("🧠 What is Spend Volatility & Why the Model Uses It")

st.markdown("""
**Spend Volatility** measures how much a campaign’s daily spend
**fluctuates from day to day**.

### Why this matters
- Two campaigns can look *On Track* in Excel today  
- The one with **unstable daily spend** is far more likely to overspend or underspend later

### Excel vs Model
- **Excel pacing** checks only cumulative spend till today  
- **The model** also checks how *predictable* the spend behavior is

### Simple example
- **Stable spend**: 1000 → 1020 → 980 → 1010 → 995  
  → Low volatility → Lower risk  
- **Unstable spend**: 400 → 1800 → 600 → 2100 → 500  
  → High volatility → Higher risk  

### How it’s calculated
The model looks at the **last 5 days of spend** and measures
how much it varies (standard deviation).

### How to read this dashboard
- **Low volatility + On Track** → Safe  
- **High volatility + Excel On Track** → ⚠️ Model flags early risk
""")

# =====================================================
# FOOTER
# =====================================================
st.divider()
st.caption(
    "📌 Summary: Excel reacts after deviation happens. "
    "PaceSmart anticipates risk earlier using trends and spend stability."
)
