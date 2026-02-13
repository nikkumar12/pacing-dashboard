import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ==================================================
# CONFIG
# ==================================================
BASE_PATH = r"C:\Users\nikkumar12\PyCharmMiscProject"

INPUT_FILE = f"{BASE_PATH}\\Realistic_Today_Context_Pacing_Data.xlsx"
OUTPUT_FILE = f"{BASE_PATH}\\Realistic_Today_Context_Pacing_Data_variation.xlsx"

# New random seed each run
RANDOM_SEED = np.random.randint(1, 1_000_000)
np.random.seed(RANDOM_SEED)

print(f"\n🔁 Generating clean campaign-consistent variation | Seed: {RANDOM_SEED}")

# ==================================================
# LOAD ORIGINAL DATA
# ==================================================
daily = pd.read_excel(INPUT_FILE, sheet_name="Daily_Raw_Data")
meta = pd.read_excel(INPUT_FILE, sheet_name="Campaign_Metadata")

daily["Date"] = pd.to_datetime(daily["Date"], errors="coerce").dt.normalize()
meta["Flight_Start_Date"] = pd.to_datetime(meta["Flight_Start_Date"]).dt.normalize()
meta["Flight_End_Date"] = pd.to_datetime(meta["Flight_End_Date"]).dt.normalize()

TODAY = datetime.utcnow().date()
YESTERDAY = TODAY - timedelta(days=1)

print(f"🕒 Today (UTC): {TODAY}")
print(f"📅 Generating data till: {YESTERDAY}")

# ==================================================
# PATTERN TYPES
# ==================================================
pattern_types = ["stable", "front_loaded", "back_loaded", "volatile"]

# ==================================================
# GENERATE CLEAN DATA
# ==================================================
new_rows = []

for _, row in meta.iterrows():

    cid = row["Campaign_ID"]
    start = row["Flight_Start_Date"].date()
    end = min(row["Flight_End_Date"].date(), YESTERDAY)

    # Skip if campaign hasn't started
    if start > YESTERDAY:
        continue

    # Choose ONE pattern per campaign
    pattern = np.random.choice(pattern_types)

    date_range = pd.date_range(start=start, end=end, freq="D")

    base_spend = np.random.uniform(800, 2000)

    for i, d in enumerate(date_range):

        day_position = i / len(date_range)

        # Pattern logic
        if pattern == "stable":
            multiplier = np.random.uniform(0.9, 1.1)

        elif pattern == "front_loaded":
            multiplier = 1.4 - day_position + np.random.uniform(-0.1, 0.1)

        elif pattern == "back_loaded":
            multiplier = 0.6 + day_position + np.random.uniform(-0.1, 0.1)

        elif pattern == "volatile":
            multiplier = np.random.uniform(0.6, 1.6)

        spend = round(max(base_spend * multiplier, 0), 2)

        new_rows.append({
            "Campaign_ID": cid,
            "Date": d,
            "Spend": spend
        })

# ==================================================
# CREATE CLEAN DATAFRAME
# ==================================================
daily_clean = pd.DataFrame(new_rows)

# Sort properly
daily_clean = daily_clean.sort_values(["Campaign_ID", "Date"])

# ==================================================
# WRITE OUTPUT (SCHEMA UNCHANGED)
# ==================================================
with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    daily_clean.to_excel(writer, sheet_name="Daily_Raw_Data", index=False)
    meta.to_excel(writer, sheet_name="Campaign_Metadata", index=False)

print("\n✅ Clean structured variation file generated")
print("✔ Continuous dates")
print("✔ Campaign-consistent patterns")
print("✔ DSP preserved from metadata")
print("✔ No broken schema")
print(f"📂 Output: {OUTPUT_FILE}")
