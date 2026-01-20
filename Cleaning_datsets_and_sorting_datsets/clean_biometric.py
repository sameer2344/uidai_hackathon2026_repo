import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(BASE_DIR, "biometric_all.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "biometric_cleaned.csv")

print("Loading merged CSV...")
df = pd.read_csv(INPUT_FILE)
print("Rows before cleaning:", len(df))

# 1. Remove exact duplicate rows
df = df.drop_duplicates()
print("Rows after removing duplicates:", len(df))

# 2. Clean STATE (do not change column position)
df["state"] = (
    df["state"]
    .astype(str)
    .str.strip()
    .str.title()
)

# Mark clearly wrong states as INVALID (optional but safe)
df.loc[df["state"].str.len() < 3, "state"] = "INVALID"

# 3. Clean DISTRICT
df["district"] = (
    df["district"]
    .astype(str)
    .str.strip()
    .str.title()
)

# 4. Standardize DATE → YYYY-MM-DD
df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce",
    dayfirst=True
).dt.strftime("%Y-%m-%d")

# 5. Fix PINCODE → exactly 6 digits
df["pincode"] = (
    df["pincode"]
    .astype(str)
    .str.extract(r"(\d+)", expand=False)
    .str.zfill(6)
)

# Save cleaned file (same column order preserved)
df.to_csv(OUTPUT_FILE, index=False)

print("✅ Cleaning done")
print("Final rows:", len(df))
print("Saved as:", OUTPUT_FILE)
