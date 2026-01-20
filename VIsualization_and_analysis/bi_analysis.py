import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- SETUP & DATA LOADING ---
print("Loading data for Bilateral Analysis...")
biometric_df = pd.read_csv('biometric_cleaned.csv')
demographic_df = pd.read_csv('demographic_cleaned.csv')
enrolment_df = pd.read_csv('enrolment_cleaned.csv')

# Convert date columns
biometric_df['date'] = pd.to_datetime(biometric_df['date'])
demographic_df['date'] = pd.to_datetime(demographic_df['date'])
enrolment_df['date'] = pd.to_datetime(enrolment_df['date'])

# Calculate totals
biometric_df['total_updates'] = biometric_df['bio_age_5_17'] + biometric_df['bio_age_17_']
enrolment_df['total_enrolment'] = enrolment_df['age_0_5'] + enrolment_df['age_5_17'] + enrolment_df['age_18_greater']

sns.set_style("whitegrid")

# Prepare Aggregations
bio_by_state = biometric_df.groupby('state')['total_updates'].sum()
demo_by_state = demographic_df.groupby('state')['demo_age_5_17'].sum()
enrol_by_state = enrolment_df.groupby('state')['total_enrolment'].sum()

# --- CRITICAL FIX: ALIGN DATES ---
# We use .add(fill_value=0) to ensure both series cover the exact same date range
daily_bio_total = biometric_df.groupby('date')['total_updates'].sum()
daily_demo_total = demographic_df.groupby('date')['demo_age_5_17'].sum()

# This aligns them. If a date is missing in one, it treats it as 0.
daily_bio_aligned = daily_bio_total.add(daily_demo_total * 0, fill_value=0)
daily_demo_aligned = daily_demo_total.add(daily_bio_total * 0, fill_value=0)

# Repeat for enrolment
daily_enrol_total = enrolment_df.groupby('date')['total_enrolment'].sum()
daily_enrol_aligned_demo = daily_enrol_total.add(daily_demo_total * 0, fill_value=0)
daily_demo_aligned_enrol = daily_demo_total.add(daily_enrol_total * 0, fill_value=0)
daily_enrol_aligned_bio = daily_enrol_total.add(daily_bio_total * 0, fill_value=0)
daily_bio_aligned_enrol = daily_bio_total.add(daily_enrol_total * 0, fill_value=0)


# --- BIOMETRIC VS DEMOGRAPHIC ---
print("Generating Biometric vs Demographic Comparisons...")

# Figure 28: Scatter Plot
plt.figure(figsize=(12, 8))
merged_bio_demo = pd.merge(bio_by_state, demo_by_state, left_index=True, right_index=True, how='inner')
plt.scatter(merged_bio_demo['total_updates'], merged_bio_demo['demo_age_5_17'], 
            alpha=0.7, s=150, c='purple', edgecolors='black')
for state in merged_bio_demo.index[:10]:
    plt.annotate(state, (merged_bio_demo.loc[state, 'total_updates'], 
                         merged_bio_demo.loc[state, 'demo_age_5_17']), fontsize=8)
plt.title('Biometric vs Demographic: State-wise Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Biometric Total Updates')
plt.ylabel('Demographic Updates (Age 5-17)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Figure 29: Side-by-Side Bar (Top 10)
plt.figure(figsize=(14, 6))
top_states_bio_demo = merged_bio_demo.nlargest(10, 'total_updates')
x = np.arange(len(top_states_bio_demo))
width = 0.35
plt.bar(x - width/2, top_states_bio_demo['total_updates'], width, label='Biometric', color='skyblue')
plt.bar(x + width/2, top_states_bio_demo['demo_age_5_17'], width, label='Demographic', color='lightcoral')
plt.title('Biometric vs Demographic: Top 10 States', fontsize=14, fontweight='bold')
plt.xticks(x, top_states_bio_demo.index, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Figure 30: Daily Trend Comparison
plt.figure(figsize=(14, 6))
# Use aligned data
plt.plot(daily_bio_aligned.index, daily_bio_aligned.values, label='Biometric', linewidth=2)
plt.plot(daily_demo_aligned.index, daily_demo_aligned.values, label='Demographic', linewidth=2)
plt.title('Biometric vs Demographic: Daily Updates Comparison', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()

# Figure 31: Stacked Area (FIXED)
plt.figure(figsize=(14, 6))
# Use aligned data so shapes match (89 vs 126 issue resolved)
plt.fill_between(daily_bio_aligned.index, 0, daily_bio_aligned.values, alpha=0.5, label='Biometric')
plt.fill_between(daily_demo_aligned.index, daily_bio_aligned.values, 
                 daily_bio_aligned.values + daily_demo_aligned.values, alpha=0.5, label='Demographic')
plt.title('Biometric vs Demographic: Cumulative Daily Updates', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()

# Figure 32: Distribution Comparison
plt.figure(figsize=(12, 6))
sns.histplot(biometric_df['bio_age_5_17'], bins=30, kde=True, color='blue', label='Biometric', alpha=0.5)
sns.histplot(demographic_df['demo_age_5_17'], bins=30, kde=True, color='green', label='Demographic', alpha=0.5)
plt.title('Biometric vs Demographic: Distribution Comparison', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()

# --- DEMOGRAPHIC VS ENROLMENT ---
print("Generating Demographic vs Enrolment Comparisons...")

# Figure 33: Scatter Plot
plt.figure(figsize=(12, 8))
merged_demo_enrol = pd.merge(demo_by_state, enrol_by_state, left_index=True, right_index=True, how='inner')
plt.scatter(merged_demo_enrol['demo_age_5_17'], merged_demo_enrol['total_enrolment'], 
            alpha=0.7, s=150, c='orange', edgecolors='black')
for state in merged_demo_enrol.index[:10]:
    plt.annotate(state, (merged_demo_enrol.loc[state, 'demo_age_5_17'], 
                         merged_demo_enrol.loc[state, 'total_enrolment']), fontsize=8)
plt.title('Demographic vs Enrolment: State-wise Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Demographic Updates')
plt.ylabel('Total Enrolment')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Figure 34: Side-by-Side Bar
plt.figure(figsize=(14, 6))
top_states_demo_enrol = merged_demo_enrol.nlargest(10, 'total_enrolment')
x = np.arange(len(top_states_demo_enrol))
width = 0.35
plt.bar(x - width/2, top_states_demo_enrol['demo_age_5_17'], width, label='Demographic', color='lightgreen')
plt.bar(x + width/2, top_states_demo_enrol['total_enrolment'], width, label='Enrolment', color='gold')
plt.title('Demographic vs Enrolment: Top 10 States', fontsize=14, fontweight='bold')
plt.xticks(x, top_states_demo_enrol.index, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Figure 35: Daily Trend Comparison
plt.figure(figsize=(14, 6))
plt.plot(daily_demo_aligned_enrol.index, daily_demo_aligned_enrol.values, label='Demographic', color='green')
plt.plot(daily_enrol_aligned_demo.index, daily_enrol_aligned_demo.values, label='Enrolment', color='coral')
plt.title('Demographic vs Enrolment: Daily Trend Comparison', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()

# Figure 36: Stacked Area (FIXED)
plt.figure(figsize=(14, 6))
plt.fill_between(daily_demo_aligned_enrol.index, 0, daily_demo_aligned_enrol.values, alpha=0.5, label='Demographic')
plt.fill_between(daily_enrol_aligned_demo.index, daily_demo_aligned_enrol.values, 
                 daily_demo_aligned_enrol.values + daily_enrol_aligned_demo.values, alpha=0.5, label='Enrolment')
plt.title('Demographic vs Enrolment: Cumulative Comparison', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()

# Figure 37: Age Group Comparison
plt.figure(figsize=(14, 6))
daily_demo_age = demographic_df.groupby('date')['demo_age_5_17'].sum()
daily_enrol_age = enrolment_df.groupby('date')['age_5_17'].sum()
# Aligning these specific series as well
daily_demo_age_aligned = daily_demo_age.add(daily_enrol_age * 0, fill_value=0)
daily_enrol_age_aligned = daily_enrol_age.add(daily_demo_age * 0, fill_value=0)

plt.plot(daily_demo_age_aligned.index, daily_demo_age_aligned.values, label='Demographic (Age 5-17)', color='green')
plt.plot(daily_enrol_age_aligned.index, daily_enrol_age_aligned.values, label='Enrolment (Age 5-17)', color='orange')
plt.title('Demographic vs Enrolment: Age 5-17 Daily Comparison', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()

# --- ENROLMENT VS BIOMETRIC ---
print("Generating Enrolment vs Biometric Comparisons...")

# Figure 38: Scatter Plot
plt.figure(figsize=(12, 8))
merged_enrol_bio = pd.merge(enrol_by_state, bio_by_state, left_index=True, right_index=True, how='inner')
plt.scatter(merged_enrol_bio['total_enrolment'], merged_enrol_bio['total_updates'], 
            alpha=0.7, s=150, c='teal', edgecolors='black')
for state in merged_enrol_bio.index[:10]:
    plt.annotate(state, (merged_enrol_bio.loc[state, 'total_enrolment'], 
                         merged_enrol_bio.loc[state, 'total_updates']), fontsize=8)
plt.title('Enrolment vs Biometric: State-wise Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Total Enrolment')
plt.ylabel('Biometric Total Updates')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Figure 39: Side-by-Side Bar
plt.figure(figsize=(14, 6))
top_states_enrol_bio = merged_enrol_bio.nlargest(10, 'total_enrolment')
x = np.arange(len(top_states_enrol_bio))
width = 0.35
plt.bar(x - width/2, top_states_enrol_bio['total_enrolment'], width, label='Enrolment', color='gold')
plt.bar(x + width/2, top_states_enrol_bio['total_updates'], width, label='Biometric', color='skyblue')
plt.title('Enrolment vs Biometric: Top 10 States', fontsize=14, fontweight='bold')
plt.xticks(x, top_states_enrol_bio.index, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Figure 40: Daily Trend Comparison
plt.figure(figsize=(14, 6))
plt.plot(daily_enrol_aligned_bio.index, daily_enrol_aligned_bio.values, label='Enrolment', color='coral')
plt.plot(daily_bio_aligned_enrol.index, daily_bio_aligned_enrol.values, label='Biometric', color='blue')
plt.title('Enrolment vs Biometric: Daily Trend Comparison', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()

# Figure 41: Stacked Area (FIXED)
plt.figure(figsize=(14, 6))
plt.fill_between(daily_enrol_aligned_bio.index, 0, daily_enrol_aligned_bio.values, alpha=0.5, label='Enrolment')
plt.fill_between(daily_bio_aligned_enrol.index, daily_enrol_aligned_bio.values, 
                 daily_enrol_aligned_bio.values + daily_bio_aligned_enrol.values, alpha=0.5, label='Biometric')
plt.title('Enrolment vs Biometric: Cumulative Comparison', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()

# Figure 42: Age Group Comparison
plt.figure(figsize=(14, 6))
daily_enrol_age_5_17 = enrolment_df.groupby('date')['age_5_17'].sum()
daily_bio_age_5_17 = biometric_df.groupby('date')['bio_age_5_17'].sum()
# Aligning age groups
daily_enrol_age_5_17_aligned = daily_enrol_age_5_17.add(daily_bio_age_5_17 * 0, fill_value=0)
daily_bio_age_5_17_aligned = daily_bio_age_5_17.add(daily_enrol_age_5_17 * 0, fill_value=0)

plt.plot(daily_enrol_age_5_17_aligned.index, daily_enrol_age_5_17_aligned.values, label='Enrolment (Age 5-17)', color='orange')
plt.plot(daily_bio_age_5_17_aligned.index, daily_bio_age_5_17_aligned.values, label='Biometric (Age 5-17)', color='blue')
plt.title('Enrolment vs Biometric: Age 5-17 Daily Comparison', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()

print("Bilateral Analysis Complete.")