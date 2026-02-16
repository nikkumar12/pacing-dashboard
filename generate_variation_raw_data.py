import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ==================================================
# CONFIG (UNCHANGED)
# ==================================================
BASE_PATH = r"C:\Users\nikkumar12\PyCharmMiscProject"

INPUT_FILE = f"{BASE_PATH}\\Realistic_Today_Context_Pacing_Data.xlsx"
OUTPUT_FILE = f"{BASE_PATH}\\Realistic_Today_Context_Pacing_Data_variation.xlsx"

# New random seed EVERY RUN
RANDOM_SEED = np.random.randint(1, 1_000_000)
np.random.seed(RANDOM_SEED)

print(f"\n🔁 Generating Daily ML Data | Seed: {RANDOM_SEED}")

# ==================================================
# LOAD METADATA
# ==================================================
meta = pd.read_excel(INPUT_FILE, sheet_name="Campaign_Metadata")

meta["Flight_Start_Date"] = pd.to_datetime(meta["Flight_Start_Date"])
meta["Flight_End_Date"] = pd.to_datetime(meta["Flight_End_Date"])

TODAY = datetime.now().date()
YESTERDAY = TODAY - timedelta(days=1)

print(f"🕒 Today: {TODAY}")
print(f"📅 Spend generated till: {YESTERDAY}")

# ==================================================
# REMOVE FUTURE START CAMPAIGNS
# ==================================================
meta = meta[meta["Flight_Start_Date"].dt.date <= TODAY].copy()

# ==================================================
# DISTRIBUTE ENDED + LIVE CAMPAIGNS
# ==================================================
total_campaigns = len(meta)

ended_count = int(total_campaigns * 0.6)
live_count = total_campaigns - ended_count

meta = meta.sample(frac=1).reset_index(drop=True)
meta["Campaign_Status"] = None

# ---------- ENDED CAMPAIGNS ----------
for i in range(ended_count):

    duration = np.random.randint(20, 45)

    end_date = TODAY - timedelta(days=np.random.randint(5, 25))
    start_date = end_date - timedelta(days=duration)

    meta.loc[i, "Flight_Start_Date"] = pd.Timestamp(start_date)
    meta.loc[i, "Flight_End_Date"] = pd.Timestamp(end_date)
    meta.loc[i, "Campaign_Status"] = "ENDED"

# ---------- LIVE CAMPAIGNS ----------
for i in range(ended_count, total_campaigns):

    start_date = TODAY - timedelta(days=np.random.randint(5, 20))
    end_date = TODAY + timedelta(days=np.random.randint(5, 30))  # future window

    meta.loc[i, "Flight_Start_Date"] = pd.Timestamp(start_date)
    meta.loc[i, "Flight_End_Date"] = pd.Timestamp(end_date)
    meta.loc[i, "Campaign_Status"] = "LIVE"

print(f"📊 Live: {live_count} | Ended: {ended_count}")

# ==================================================
# GENERATE SPEND DATA (TILL YESTERDAY ONLY)
# ==================================================
new_rows = []

ml_patterns = [
    "underpace",
    "overpace",
    "ontrack",
    "late_push",
    "early_burst"
]

for _, row in meta.iterrows():

    cid = row["Campaign_ID"]
    start = pd.to_datetime(row["Flight_Start_Date"]).date()
    end = pd.to_datetime(row["Flight_End_Date"]).date()

    total_budget = row.get("Total_Budget", np.random.uniform(200000, 800000))
    flight_days = (end - start).days + 1
    daily_target = total_budget / flight_days

    pattern = np.random.choice(ml_patterns)

    cumulative_spend = 0
    prev_spend = daily_target

    # Generate spend ONLY till yesterday
    spend_end_date = min(end, YESTERDAY)

    if start > spend_end_date:
        continue

    date_range = pd.date_range(start=start, end=spend_end_date)

    for i, d in enumerate(date_range):

        progress = i / flight_days

        # -------- ML PATTERNS --------
        if pattern == "underpace":
            multiplier = 0.75

        elif pattern == "overpace":
            multiplier = 1.25

        elif pattern == "ontrack":
            multiplier = 1.0

        elif pattern == "late_push":
            multiplier = 0.7 if progress < 0.7 else 1.4

        elif pattern == "early_burst":
            multiplier = 1.4 if progress < 0.3 else 0.8

        # Weekend dip
        if d.weekday() >= 5:
            multiplier *= np.random.uniform(0.85, 0.95)

        # Smooth spend (autocorrelation)
        raw_spend = daily_target * multiplier
        spend = (0.6 * prev_spend) + (0.4 * raw_spend)

        spend = round(max(spend, 0), 2)

        cumulative_spend += spend
        prev_spend = spend

        new_rows.append({
            "Campaign_ID": cid,
            "Date": d,
            "Spend": spend
        })

# ==================================================
# FINALIZE OUTPUT
# ==================================================
daily_clean = pd.DataFrame(new_rows)
daily_clean = daily_clean.sort_values(["Campaign_ID", "Date"])

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    daily_clean.to_excel(writer, sheet_name="Daily_Raw_Data", index=False)
    meta.to_excel(writer, sheet_name="Campaign_Metadata", index=False)

print("\n✅ Daily ML dataset generated successfully")
print("✔ No future start campaigns")
print("✔ Live campaigns end within next 30 days")
print("✔ Spend generated only till yesterday")
print("✔ Variations change every run")
print(f"📂 Output: {OUTPUT_FILE}")
