import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(BASE_DIR, "enrolment_all.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "enrolment_cleaned.csv")

CHUNK_SIZE = 200_000

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

print("Starting enrolment cleaning...")

total_rows = 0
chunk_no = 0

for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False):
    chunk_no += 1
    print(f"Processing chunk {chunk_no}")

    # 1. Drop duplicates
    chunk.drop_duplicates(inplace=True)

    # 2. Clean state / district
    for col in ("state", "district"):
        if col in chunk.columns:
            chunk[col] = (
                chunk[col]
                .astype("string")
                .str.strip()
                .str.title()
            )

    # 3. Enrolment date standardization
    if "enrolment_date" in chunk.columns:
        chunk["enrolment_date"] = (
            pd.to_datetime(chunk["enrolment_date"], errors="coerce", dayfirst=True)
            .dt.strftime("%Y-%m-%d")
        )

    # 4. Pincode cleanup
    if "pincode" in chunk.columns:
        chunk["pincode"] = (
            chunk["pincode"]
            .astype("string")
            .str.extract(r"(\d{6})", expand=False)
        )

    # 5. Gender normalization (common in enrolment data)
    if "gender" in chunk.columns:
        chunk["gender"] = (
            chunk["gender"]
            .astype("string")
            .str.upper()
            .replace({"M": "Male", "F": "Female"})
        )

    chunk.to_csv(
        OUTPUT_FILE,
        mode="a",
        header=not os.path.exists(OUTPUT_FILE),
        index=False
    )

    total_rows += len(chunk)

print("✅ Enrolment cleaning done")
print("Final rows:", total_rows)
print("Saved as:", OUTPUT_FILE)
