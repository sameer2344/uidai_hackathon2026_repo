import pandas as pd

# Load the dataframe
df = pd.read_csv('biometric_cleaned.csv')

# --- 1. Standardization (Same as before) ---
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
    'Westbengal': 'West Bengal'
}

district_corrections = {
    'Ahmadabad': 'Ahmedabad',
    'Ahmadnagar': 'Ahmednagar',
    'Ahmed Nagar': 'Ahmednagar',
    'Ahilyanagar': 'Ahmednagar',
    'Allahabad': 'Prayagraj',
    '?': 'Unknown'
}

# Apply corrections
df['state'] = df['state'].replace(state_corrections)
df['district'] = df['district'].replace(district_corrections)

# Strip whitespace
df['state'] = df['state'].str.strip()
df['district'] = df['district'].str.strip()

# --- 2. Sorting Logic (Extended) ---

# Convert 'date' to datetime objects to ensure chronological sorting
df['date'] = pd.to_datetime(df['date'])

# Sort by Date first, then by State (and District for cleanliness)
# This fulfills the requirement: "keeping all the same state data near on that particular date"
df_sorted = df.sort_values(by=['date', 'state', 'district'])

# Inspect the result
print(df_sorted.head())

# Optional: Save the sorted file
df_sorted.to_csv('biometric_sorted.csv', index=False)