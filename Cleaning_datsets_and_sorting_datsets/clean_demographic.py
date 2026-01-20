import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(BASE_DIR, "demographic_all.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "demographic_cleaned.csv")

if not os.path.isfile(INPUT_FILE):
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

CHUNK_SIZE = 200_000   # 2 lakh rows per chunk (safe & fast)

print("Starting chunk-wise cleaning...")

# Remove old output if exists
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

chunk_no = 0
total_rows = 0

for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False):
    chunk_no += 1
    print(f"Processing chunk {chunk_no}...")

    # ---------------- CLEANING ----------------

    # 1. Drop duplicates WITHIN chunk
    chunk.drop_duplicates(inplace=True)

    # 2. Normalize text columns
    for col in ("state", "district"):
        if col in chunk.columns:
            chunk[col] = (
                chunk[col]
                .astype("string")
                .str.strip()
                .str.title()
            )

    # 3. Mark invalid states
    if "state" in chunk.columns:
        chunk.loc[chunk["state"].str.len() < 3, "state"] = "INVALID"

    # 4. Standardize date
    if "date" in chunk.columns:
        chunk["date"] = (
            pd.to_datetime(chunk["date"], errors="coerce", dayfirst=True)
            .dt.strftime("%Y-%m-%d")
        )

    # 5. Clean pincode (strict 6-digit)
    if "pincode" in chunk.columns:
        chunk["pincode"] = (
            chunk["pincode"]
            .astype("string")
            .str.extract(r"(\d{6})", expand=False)
        )

    # ---------------- SAVE ----------------
    chunk.to_csv(
        OUTPUT_FILE,
        mode="a",
        header=not os.path.exists(OUTPUT_FILE),
        index=False
    )

    total_rows += len(chunk)

print("✅ Cleaning finished")
print(f"Total rows written: {total_rows:,}")
print(f"Saved as: {OUTPUT_FILE}")
