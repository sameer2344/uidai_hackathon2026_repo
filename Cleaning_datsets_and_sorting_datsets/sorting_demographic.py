import pandas as pd
import os
import re

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "demographic_cleaned.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "demographic_sorted.csv")

CHUNK_SIZE = 200_000

# ==========================================
# STANDARDIZATION DICTIONARIES
# ==========================================
state_corrections = {
    'Andaman & Nicobar Islands': 'Andaman And Nicobar Islands',
    'Chhatisgarh': 'Chhattisgarh',
    'Dadra & Nagar Haveli': 'Dadra And Nagar Haveli And Daman And Diu',
    'Dadra And Nagar Haveli': 'Dadra And Nagar Haveli And Daman And Diu',
    'Daman & Diu': 'Dadra And Nagar Haveli And Daman And Diu',
    'Daman And Diu': 'Dadra And Nagar Haveli And Daman And Diu',
    'Jammu & Kashmir': 'Jammu And Kashmir',
    'Orissa': 'Odisha',
    'Pondicherry': 'Puducherry',
    'Tamilnadu': 'Tamil Nadu',
    'Uttaranchal': 'Uttarakhand',
    'West  Bengal': 'West Bengal',
    'West Bangal': 'West Bengal',
    'Westbengal': 'West Bengal',
    'Telengana': 'Telangana'
}

district_corrections = {
    'Ahmadabad': 'Ahmedabad',
    'Ahmadnagar': 'Ahmednagar',
    'Ahmed Nagar': 'Ahmednagar',
    'Ahilyanagar': 'Ahmednagar',
    'Allahabad': 'Prayagraj',
    '?': 'Unknown'
}

bad_name_pattern = re.compile(
    r"^\s*(\?|unknown|null|none|nan|test|dummy)\s*$",
    re.IGNORECASE
)

# ==========================================
# SAFETY CHECK
# ==========================================
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError("❌ demographic_cleaned.csv not found")

print("📥 Reading demographic data...")

# ==========================================
# AUTO-DETECT DATE COLUMN
# ==========================================
first_chunk = pd.read_csv(INPUT_FILE, nrows=1)
cols = first_chunk.columns.tolist()

date_col = None
if "date" in cols:
    date_col = "date"
else:
    candidates = [c for c in cols if "date" in c.lower()]
    if candidates:
        date_col = candidates[0]

print(f"📅 Detected date column: {date_col}")

# ==========================================
# CHUNK PROCESSING
# ==========================================
temp_files = []
chunk_no = 0

for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False):
    chunk_no += 1
    print(f"🔄 Processing chunk {chunk_no}", end="\r")

    # -------- STATE CLEANING --------
    if "state" in chunk.columns:
        chunk["state"] = (
            chunk["state"]
            .astype("string")
            .str.strip()
            .replace(state_corrections)
        )
        chunk.loc[chunk["state"].str.len() < 3, "state"] = "INVALID"

    # -------- DISTRICT CLEANING --------
    if "district" in chunk.columns:
        chunk["district"] = (
            chunk["district"]
            .astype("string")
            .str.strip()
            .replace(district_corrections)
        )

        invalid_district_mask = (
            (chunk["district"].str.len() < 3) |
            (chunk["district"].str.match(bad_name_pattern, na=False))
        )

        chunk.loc[invalid_district_mask, "district"] = "Unknown"

    # -------- DATE CLEANING --------
    if date_col and date_col in chunk.columns:
        chunk[date_col] = pd.to_datetime(
            chunk[date_col],
            errors="coerce",
            dayfirst=True
        )

    # -------- SAVE TEMP FILE --------
    temp_file = os.path.join(BASE_DIR, f"_tmp_demo_{chunk_no}.csv")
    chunk.to_csv(temp_file, index=False)
    temp_files.append(temp_file)

print(f"\n✅ Finished processing {chunk_no} chunks")

# ==========================================
# MERGE & FINAL SORT
# ==========================================
print("📊 Merging and sorting (Date → State → District)...")

df = pd.concat(
    (pd.read_csv(f) for f in temp_files),
    ignore_index=True
)

# Ensure correct dtypes
if date_col and date_col in df.columns:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

df["state"] = df["state"].astype(str)
df["district"] = df["district"].astype(str)

# 🔑 NESTED SORT
sort_cols = [date_col, "state", "district"] if date_col else ["state", "district"]

df.sort_values(
    by=sort_cols,
    ascending=[True] * len(sort_cols),
    inplace=True,
    kind="mergesort"   # stable sort
)

df.to_csv(OUTPUT_FILE, index=False)

# ==========================================
# CLEANUP
# ==========================================
for f in temp_files:
    os.remove(f)

print("✅ SUCCESS")
print(f"📁 Output saved as: {OUTPUT_FILE}")
print(f"📈 Total rows: {len(df):,}")
