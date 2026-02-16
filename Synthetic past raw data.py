import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)

# ==================================================
# CONFIG
# ==================================================
num_campaigns = 60
start_base = datetime(2025, 1, 1)

daily_rows = []
meta_rows = []

for i in range(num_campaigns):

    campaign_id = f"CAMP_{i+1}"
    dsp = random.choice(["DV360", "TTD", "META"])
    budget = random.randint(100000, 500000)
    flight_days = random.randint(20, 40)

    start_date = start_base + timedelta(days=random.randint(0, 120))
    end_date = start_date + timedelta(days=flight_days-1)

    pattern_type = random.choice(["linear", "slow_ramp", "front_loaded"])

    daily_spend = []

    for d in range(flight_days):
        progress = d / flight_days

        if pattern_type == "linear":
            spend = budget / flight_days

        elif pattern_type == "slow_ramp":
            spend = (budget / flight_days) * (0.5 + progress)

        elif pattern_type == "front_loaded":
            spend = (budget / flight_days) * (1.5 - progress)

        daily_spend.append(spend)

    # Normalize so total equals budget ± small noise
    daily_spend = np.array(daily_spend)
    daily_spend = daily_spend / daily_spend.sum() * budget
    daily_spend = daily_spend * np.random.uniform(0.9, 1.1)

    # Store daily rows
    for day in range(flight_days):
        daily_rows.append([
            campaign_id,
            start_date + timedelta(days=day),
            round(daily_spend[day], 2)
        ])

    meta_rows.append([
        campaign_id,
        dsp,
        budget,
        start_date,
        end_date
    ])

# ==================================================
# CREATE DATAFRAMES
# ==================================================
daily_df = pd.DataFrame(daily_rows, columns=[
    "Campaign_ID", "Date", "Spend"
])

meta_df = pd.DataFrame(meta_rows, columns=[
    "Campaign_ID", "DSP", "Total_Budget",
    "Flight_Start_Date", "Flight_End_Date"
])

# ==================================================
# SAVE FILE
# ==================================================
with pd.ExcelWriter("Synthetic_Pacing_Data.xlsx") as writer:
    daily_df.to_excel(writer, sheet_name="Daily_Raw_Data", index=False)
    meta_df.to_excel(writer, sheet_name="Campaign_Metadata", index=False)

print("✅ Synthetic dataset created: Synthetic_Pacing_Data.xlsx")
