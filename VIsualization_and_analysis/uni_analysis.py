import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- SETUP & DATA LOADING ---
print("Loading data for Unilateral Analysis...")
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

# --- BIOMETRIC VISUALIZATIONS ---
print("Generating Biometric Visualizations...")

# Figure 1: Biometric - Top 10 States
plt.figure(figsize=(12, 6))
state_bio = biometric_df.groupby('state')['total_updates'].sum().sort_values(ascending=False).head(10)
sns.barplot(x=state_bio.values, y=state_bio.index, palette='viridis')
plt.title('Biometric: Top 10 States by Total Updates', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Figure 2: Biometric - All States
plt.figure(figsize=(12, 16))
state_bio_all = biometric_df.groupby('state')['total_updates'].sum().sort_values(ascending=False)
sns.barplot(x=state_bio_all.values, y=state_bio_all.index, palette='coolwarm')
plt.title('Biometric: All States by Total Updates', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Figure 3: Biometric - Age Group Distribution
plt.figure(figsize=(10, 6))
bio_age_totals = {
    'Age 5-17': biometric_df['bio_age_5_17'].sum(),
    'Age 17+': biometric_df['bio_age_17_'].sum()
}
plt.pie(bio_age_totals.values(), labels=bio_age_totals.keys(), autopct='%1.1f%%', 
        colors=['#66b3ff', '#ff9999'], startangle=90)
plt.title('Biometric: Updates Distribution by Age Group', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Figure 4: Biometric - Age Group Comparison (Stacked)
plt.figure(figsize=(12, 6))
top_states = biometric_df.groupby('state')['total_updates'].sum().sort_values(ascending=False).head(10).index
bio_state_age = biometric_df[biometric_df['state'].isin(top_states)].groupby('state')[['bio_age_5_17', 'bio_age_17_']].sum()
bio_state_age.plot(kind='bar', stacked=True, color=['#8dd3c7', '#fb8072'])
plt.title('Biometric: Age Group Distribution (Top 10 States)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Figure 5: Biometric - Daily Trend
plt.figure(figsize=(14, 6))
daily_bio = biometric_df.groupby('date')['total_updates'].sum()
plt.plot(daily_bio.index, daily_bio.values, color='blue', linewidth=2, marker='o', markersize=4)
plt.title('Biometric: Daily Updates Trend', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Figure 6: Biometric - Age Group Trends
plt.figure(figsize=(14, 6))
daily_bio_age = biometric_df.groupby('date')[['bio_age_5_17', 'bio_age_17_']].sum()
plt.plot(daily_bio_age.index, daily_bio_age['bio_age_5_17'], label='Age 5-17')
plt.plot(daily_bio_age.index, daily_bio_age['bio_age_17_'], label='Age 17+')
plt.title('Biometric: Daily Updates by Age Group', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()

# Figure 7: Biometric - Cumulative Updates
plt.figure(figsize=(14, 6))
plt.fill_between(daily_bio_age.index, 0, daily_bio_age['bio_age_5_17'], alpha=0.5, label='Age 5-17')
plt.fill_between(daily_bio_age.index, daily_bio_age['bio_age_5_17'], 
                 daily_bio_age['bio_age_5_17'] + daily_bio_age['bio_age_17_'], alpha=0.5, label='Age 17+')
plt.title('Biometric: Cumulative Daily Updates by Age Group', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()

# Figure 8: Biometric - Distribution
plt.figure(figsize=(12, 6))
sns.histplot(biometric_df['bio_age_5_17'], bins=30, kde=True, color='skyblue', label='Age 5-17')
sns.histplot(biometric_df['bio_age_17_'], bins=30, kde=True, color='salmon', label='Age 17+')
plt.title('Biometric: Distribution of Updates by Age Group', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()

# Figure 9: Biometric - KDE Density
plt.figure(figsize=(12, 6))
sns.kdeplot(data=biometric_df['bio_age_5_17'], fill=True, color='blue', alpha=0.5, label='Age 5-17')
sns.kdeplot(data=biometric_df['bio_age_17_'], fill=True, color='red', alpha=0.5, label='Age 17+')
plt.title('Biometric: Density Distribution by Age Group', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()

# Figure 10: Biometric - Records Count
plt.figure(figsize=(12, 16))
state_counts = biometric_df['state'].value_counts().sort_values(ascending=False)
sns.barplot(x=state_counts.values, y=state_counts.index, palette='magma')
plt.title('Biometric: Number of Records by State', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# --- DEMOGRAPHIC VISUALIZATIONS ---
print("Generating Demographic Visualizations...")

# Figure 11: Demographic - Top 10 States
plt.figure(figsize=(12, 6))
state_demo = demographic_df.groupby('state')['demo_age_5_17'].sum().sort_values(ascending=False).head(10)
sns.barplot(x=state_demo.values, y=state_demo.index, palette='plasma')
plt.title('Demographic: Top 10 States by Updates (Age 5-17)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Figure 12: Demographic - All States
plt.figure(figsize=(12, 16))
state_demo_all = demographic_df.groupby('state')['demo_age_5_17'].sum().sort_values(ascending=False)
sns.barplot(x=state_demo_all.values, y=state_demo_all.index, palette='YlOrRd')
plt.title('Demographic: All States by Updates', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Figure 13: Demographic - Daily Trend
plt.figure(figsize=(14, 6))
daily_demo = demographic_df.groupby('date')['demo_age_5_17'].sum()
plt.plot(daily_demo.index, daily_demo.values, color='green', linewidth=2, marker='o')
plt.title('Demographic: Daily Updates Trend', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Figure 14: Demographic - Distribution
plt.figure(figsize=(12, 6))
sns.histplot(demographic_df['demo_age_5_17'], bins=30, kde=True, color='lightgreen')
plt.title('Demographic: Distribution of Updates', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Figure 15: Demographic - KDE Density
plt.figure(figsize=(12, 6))
sns.kdeplot(data=demographic_df['demo_age_5_17'], fill=True, color='green', alpha=0.6)
plt.title('Demographic: Density Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Figure 16: Demographic - Records Count
plt.figure(figsize=(12, 16))
demo_state_counts = demographic_df['state'].value_counts().sort_values(ascending=False)
sns.barplot(x=demo_state_counts.values, y=demo_state_counts.index, palette='Greens')
plt.title('Demographic: Number of Records by State', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Figure 17: Demographic - Area Chart
plt.figure(figsize=(14, 6))
plt.fill_between(daily_demo.index, daily_demo.values, alpha=0.5, color='lightgreen')
plt.plot(daily_demo.index, daily_demo.values, color='darkgreen')
plt.title('Demographic: Cumulative Daily Updates', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# --- ENROLMENT VISUALIZATIONS ---
print("Generating Enrolment Visualizations...")

# Figure 18: Enrolment - Top 10 States
plt.figure(figsize=(12, 6))
state_enrol = enrolment_df.groupby('state')['total_enrolment'].sum().sort_values(ascending=False).head(10)
sns.barplot(x=state_enrol.values, y=state_enrol.index, palette='rocket')
plt.title('Enrolment: Top 10 States', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Figure 19: Enrolment - All States
plt.figure(figsize=(12, 16))
state_enrol_all = enrolment_df.groupby('state')['total_enrolment'].sum().sort_values(ascending=False)
sns.barplot(x=state_enrol_all.values, y=state_enrol_all.index, palette='mako')
plt.title('Enrolment: All States', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Figure 20: Enrolment - Age Group Distribution
plt.figure(figsize=(10, 6))
enrol_age_totals = {
    'Age 0-5': enrolment_df['age_0_5'].sum(),
    'Age 5-17': enrolment_df['age_5_17'].sum(),
    'Age 18+': enrolment_df['age_18_greater'].sum()
}
plt.pie(enrol_age_totals.values(), labels=enrol_age_totals.keys(), autopct='%1.1f%%', 
        colors=['#ff9999', '#66b3ff', '#99ff99'], startangle=90)
plt.title('Enrolment: Distribution by Age Group', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Figure 21: Enrolment - Age Group Comparison
plt.figure(figsize=(12, 6))
top_enrol_states = enrolment_df.groupby('state')['total_enrolment'].sum().sort_values(ascending=False).head(10).index
enrol_state_age = enrolment_df[enrolment_df['state'].isin(top_enrol_states)].groupby('state')[['age_0_5', 'age_5_17', 'age_18_greater']].sum()
enrol_state_age.plot(kind='bar', stacked=True, color=['#ffd700', '#87ceeb', '#98fb98'])
plt.title('Enrolment: Age Group Distribution (Top 10 States)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Figure 22: Enrolment - Daily Trend
plt.figure(figsize=(14, 6))
daily_enrolment = enrolment_df.groupby('date')['total_enrolment'].sum()
plt.plot(daily_enrolment.index, daily_enrolment.values, color='coral', linewidth=2, marker='o')
plt.title('Enrolment: Daily Total Enrolment Trend', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Figure 23: Enrolment - Age Group Trends
plt.figure(figsize=(14, 6))
age_daily = enrolment_df.groupby('date')[['age_0_5', 'age_5_17', 'age_18_greater']].sum()
plt.plot(age_daily.index, age_daily['age_0_5'], label='Age 0-5')
plt.plot(age_daily.index, age_daily['age_5_17'], label='Age 5-17')
plt.plot(age_daily.index, age_daily['age_18_greater'], label='Age 18+')
plt.title('Enrolment: Daily Trends by Age Group', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()

# Figure 24: Enrolment - Cumulative Area Chart
plt.figure(figsize=(14, 6))
plt.fill_between(age_daily.index, 0, age_daily['age_0_5'], alpha=0.5, label='Age 0-5', color='gold')
plt.fill_between(age_daily.index, age_daily['age_0_5'], 
                 age_daily['age_0_5'] + age_daily['age_5_17'], alpha=0.5, label='Age 5-17', color='skyblue')
plt.fill_between(age_daily.index, age_daily['age_0_5'] + age_daily['age_5_17'],
                 age_daily['age_0_5'] + age_daily['age_5_17'] + age_daily['age_18_greater'], alpha=0.5, label='Age 18+', color='lightgreen')
plt.title('Enrolment: Cumulative Daily Enrolment', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()

# Figure 25: Enrolment - Distribution
plt.figure(figsize=(12, 6))
sns.histplot(enrolment_df['age_0_5'], bins=30, kde=True, color='gold', label='Age 0-5', alpha=0.6)
sns.histplot(enrolment_df['age_5_17'], bins=30, kde=True, color='skyblue', label='Age 5-17', alpha=0.6)
sns.histplot(enrolment_df['age_18_greater'], bins=30, kde=True, color='lightgreen', label='Age 18+', alpha=0.6)
plt.title('Enrolment: Distribution by Age Group', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()

# Figure 26: Enrolment - KDE Density
plt.figure(figsize=(12, 6))
sns.kdeplot(data=enrolment_df['age_0_5'], fill=True, color='gold', alpha=0.5, label='Age 0-5')
sns.kdeplot(data=enrolment_df['age_5_17'], fill=True, color='blue', alpha=0.5, label='Age 5-17')
sns.kdeplot(data=enrolment_df['age_18_greater'], fill=True, color='green', alpha=0.5, label='Age 18+')
plt.title('Enrolment: Density Distribution', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()

# Figure 27: Enrolment - Records Count
plt.figure(figsize=(12, 16))
enrol_state_counts = enrolment_df['state'].value_counts().sort_values(ascending=False)
sns.barplot(x=enrol_state_counts.values, y=enrol_state_counts.index, palette='Blues')
plt.title('Enrolment: Number of Records by State', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("Unilateral Analysis Complete.")