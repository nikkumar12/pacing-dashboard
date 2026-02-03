import pandas as pd
import os

# =========================
# CONFIG — SINGLE SOURCE
# =========================
BASE_DIR = r"C:\Users\nikkumar12\OneDrive - Publicis Groupe\Desktop\2026"
FILE_NAME = "PaceSmart_Excel_Pacing_With_Formulas.xlsx"
FILE_PATH = os.path.join(BASE_DIR, FILE_NAME)

SHEET_NAME = "Combined_Raw"

# =========================
# PRINT WHAT WE ARE READING
# =========================
print("\n📂 Attempting to read file:")
print(FILE_PATH)

if not os.path.exists(FILE_PATH):
    raise FileNotFoundError("❌ Excel file NOT found at this location")

print("✅ File exists")

# =========================
# READ COMBINED RAW TAB
# =========================
df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)

print(f"\n📄 Sheet loaded: {SHEET_NAME}")
print("Rows:", df.shape[0])
print("Columns:", list(df.columns))

print("\n📊 Sample rows (first 5):")
print(df.head())

# =========================
# BASIC VALIDATION
# =========================
expected_cols = {"Date", "Campaign ID", "DSP", "Spend"}

missing = expected_cols - set(df.columns)
if missing:
    raise ValueError(f"❌ Missing expected columns: {missing}")

print("\n✅ Combined_Raw looks correct and ready for modeling")
