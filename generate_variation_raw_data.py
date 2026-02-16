import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# ==================================================
# CONFIG
# ==================================================
BASE_PATH = r"C:\Users\nikkumar12\PyCharmMiscProject"

OUTPUT_FILE = f"{BASE_PATH}\\Realistic_Today_Context_Pacing_Data_variation.xlsx"

RANDOM_SEED = np.random.randint(1, 1_000_000)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

print(f"\n🔁 Generating LARGE Realistic Dataset | Seed: {RANDOM_SEED}")

# ==================================================
# GLOBAL SETTINGS
# ==================================================
TOTAL_CAMPAIGNS = 400
TODAY = datetime.now().date()
YESTERDAY = TODAY - timedelta(days=1)

# DSP pool
DSP_LIST = ["DV360", "TTD", "META", "AMAZON", "XANDR"]

meta_rows = []
daily_rows = []

# ==================================================
# GENERATE CAMPAIGNS
# ==================================================
for i in range(TOTAL_CAMPAIGNS):

    campaign_id = f"CAMP_{i+1}"
    dsp = random.choice(DSP_LIST)

    # Budget tiers
    budget = random.choice([
        random.randint(50000, 150000),
        random.randint(200000, 400000),
        random.randint(500000, 1200000)
    ])

    flight_days = random.randint(20, 60)

    # 60% ended
    if random.random() < 0.6:
        end_date = TODAY - timedelta(days=random.randint(5, 30))
        start_date = end_date - timedelta(days=flight_days)
        status = "ENDED"
    else:
        start_date = TODAY - timedelta(days=random.randint(5, 25))
        end_date = TODAY + timedelta(days=random.randint(5, 40))
        status = "LIVE"

    meta_rows.append([
        campaign_id,
        dsp,
        budget,
        start_date,
        end_date,
        status
    ])

    # ==================================================
    # SPEND BEHAVIOR
    # ==================================================
    daily_target = budget / flight_days
    prev_spend = daily_target

    spend_end = min(end_date, YESTERDAY)
    if start_date > spend_end:
        continue

    date_range = pd.date_range(start=start_date, end=spend_end)

    curve_type = random.choice([
        "linear",
        "late_acceleration",
        "early_burst",
        "sigmoid",
        "fade_out",
        "recover_mid",
        "volatile",
        "flat_then_push"
    ])

    dsp_bias = {
        "META": 1.15,
        "DV360": 1.0,
        "TTD": 0.95,
        "AMAZON": 1.05,
        "XANDR": 0.9
    }[dsp]

    for i_day, d in enumerate(date_range):

        progress = i_day / flight_days

        # -------- CORE CURVES --------
        if curve_type == "linear":
            multiplier = 1.0

        elif curve_type == "late_acceleration":
            multiplier = 0.5 + (progress ** 2.5) * 2.2

        elif curve_type == "early_burst":
            multiplier = 1.8 - (progress ** 2.2) * 1.5

        elif curve_type == "sigmoid":
            multiplier = 1 / (1 + np.exp(-12*(progress - 0.6))) + 0.5

        elif curve_type == "fade_out":
            multiplier = 1.6 - progress * 1.2

        elif curve_type == "recover_mid":
            multiplier = 0.7 + np.sin(progress * 3.14) * 0.8

        elif curve_type == "volatile":
            multiplier = np.random.uniform(0.6, 1.6)

        elif curve_type == "flat_then_push":
            multiplier = 0.8 if progress < 0.75 else 2.0

        # Weekend dip
        if d.weekday() >= 5:
            multiplier *= np.random.uniform(0.85, 0.95)

        # Month-end push
        if d.day >= 25:
            multiplier *= 1.1

        # Random shock day
        if random.random() < 0.05:
            multiplier *= random.uniform(0.5, 1.8)

        # DSP bias
        multiplier *= dsp_bias

        # Volatility cluster
        volatility = np.random.normal(1.0, 0.12)

        raw_spend = daily_target * multiplier * volatility

        # Autocorrelation smoothing
        spend = (0.65 * prev_spend) + (0.35 * raw_spend)
        spend = round(max(spend, 0), 2)

        prev_spend = spend

        daily_rows.append([
            campaign_id,
            d,
            spend
        ])

# ==================================================
# FINAL DATAFRAMES
# ==================================================
meta_df = pd.DataFrame(meta_rows, columns=[
    "Campaign_ID",
    "DSP",
    "Total_Budget",
    "Flight_Start_Date",
    "Flight_End_Date",
    "Campaign_Status"
])

daily_df = pd.DataFrame(daily_rows, columns=[
    "Campaign_ID",
    "Date",
    "Spend"
]).sort_values(["Campaign_ID", "Date"])

# ==================================================
# SAVE OUTPUT
# ==================================================
with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    daily_df.to_excel(writer, sheet_name="Daily_Raw_Data", index=False)
    meta_df.to_excel(writer, sheet_name="Campaign_Metadata", index=False)

print("\n✅ LARGE realistic dataset generated successfully")
print(f"📊 Total Campaigns: {TOTAL_CAMPAIGNS}")
print("✔ Non-linear curves")
print("✔ DSP behavior bias")
print("✔ Volatility clusters")
print("✔ Shock days")
print("✔ Month-end push")
print("✔ Mixed live + ended")
print(f"📂 Output: {OUTPUT_FILE}")
