import pandas as pd
from datetime import datetime, timedelta

# --------------------------------------------------
# FILE PATHS (YOUR LOCAL SETUP)
# --------------------------------------------------
BASE_PATH = r"C:\Users\nikkumar12\PyCharmMiscProject"

INPUT_FILE = f"{BASE_PATH}\\Realistic_Today_Context_Pacing_Data.xlsx"
OUTPUT_FILE = f"{BASE_PATH}\\Excel_Pacing_Output.xlsx"

# --------------------------------------------------
# DATE CONTEXT
# --------------------------------------------------
TODAY = datetime.today().date()
LAST_DATA_DATE = TODAY - timedelta(days=1)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
daily = pd.read_excel(INPUT_FILE, sheet_name="Daily_Raw_Data")
meta = pd.read_excel(INPUT_FILE, sheet_name="Campaign_Metadata")

daily["Date"] = pd.to_datetime(daily["Date"]).dt.date
meta["Flight_Start_Date"] = pd.to_datetime(meta["Flight_Start_Date"]).dt.date
meta["Flight_End_Date"] = pd.to_datetime(meta["Flight_End_Date"]).dt.date

# --------------------------------------------------
# BUILD PACING LOGIC
# --------------------------------------------------
rows = []

for _, c in meta.iterrows():
    campaign_id = c["Campaign_ID"]
    dsp = c["DSP"]
    flight_start = c["Flight_Start_Date"]
    flight_end = c["Flight_End_Date"]
    total_budget = c["Total_Budget"]

    # Filter daily data till yesterday
    d = daily[
        (daily["Campaign_ID"] == campaign_id) &
        (daily["Date"] <= LAST_DATA_DATE)
    ]

    spend_to_date = d["Spend"].sum() if not d.empty else 0.0

    total_days = (flight_end - flight_start).days + 1

    if LAST_DATA_DATE < flight_start:
        days_elapsed = 0
        days_left = total_days
    else:
        days_elapsed = min(
            (LAST_DATA_DATE - flight_start).days + 1,
            total_days
        )
        days_left = max((flight_end - LAST_DATA_DATE).days, 0)

    expected_spend = (
        total_budget * (days_elapsed / total_days)
        if total_days > 0 else 0
    )

    pacing_pct = (
        spend_to_date / expected_spend
        if expected_spend > 0 else 0
    )

    remaining_budget = total_budget - spend_to_date

    ideal_daily_spend = (
        remaining_budget / days_left
        if days_left > 0 else 0
    )

    # Pacing status
    if expected_spend == 0:
        pacing_status = "No Data"
    elif pacing_pct < 0.95:
        pacing_status = "Underpacing"
    elif pacing_pct > 1.05:
        pacing_status = "Overpacing"
    else:
        pacing_status = "On Track"

    rows.append([
        campaign_id,
        dsp,
        flight_start,
        flight_end,
        TODAY,
        LAST_DATA_DATE,
        round(total_budget, 2),
        round(spend_to_date, 2),
        days_elapsed,
        days_left,
        round(expected_spend, 2),
        round(pacing_pct * 100, 2),
        round(remaining_budget, 2),
        round(ideal_daily_spend, 2),
        pacing_status
    ])

# --------------------------------------------------
# OUTPUT EXCEL
# --------------------------------------------------
columns = [
    "Campaign_ID",
    "DSP",
    "Flight_Start_Date",
    "Flight_End_Date",
    "Today_Date",
    "Last_Data_Till",
    "Total_Budget",
    "Spend_to_Date",
    "Days_Elapsed",
    "Days_Left",
    "Expected_Spend_Till_Date",
    "Pacing_%_vs_Ideal",
    "Remaining_Budget",
    "Ideal_Daily_Spend",
    "Pacing_Status"
]

pacing_df = pd.DataFrame(rows, columns=columns)

pacing_df.to_excel(OUTPUT_FILE, sheet_name="Pacing", index=False)

print("✅ Excel pacing file created at:")
print(OUTPUT_FILE)
