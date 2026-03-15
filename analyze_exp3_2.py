import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ----------------------
# Global Configuration
# ----------------------
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
VALID_PR_LABELS = ['Compliance', 'Resistance']

# ----------------------
# Step 1: Load Matched Data from Step 1
# ----------------------
df_matched = pd.read_csv('exp3_eml_pr_matched_data.csv', encoding='utf-8')

# Data cleaning: filter records with empty DA type
df_matched = df_matched.dropna(subset=['eml_da_type'])
# Mark EML presence (all records have EML, retained for 3-dimension framework)
df_matched['EML_Presence'] = 'Yes'

print(f"Number of valid EML-PR matched records for analysis: {len(df_matched)}")

# ----------------------
# Step 2: Three-Dimensional Basic Statistics
# ----------------------
def calculate_basic_metrics(df):
    """Calculate 3-dimensional basic metrics: count/probability/compliance rate"""
    # 1. Count by EML Presence × DA Type × PR Type
    count_metrics = df.groupby(['EML_Presence', 'eml_da_type', 'EML_result']).size().reset_index(name='Count')

    # 2. Calculate probability for each combination (Count / Total for the DA type)
    da_total = df.groupby(['EML_Presence', 'eml_da_type']).size().reset_index(name='DA_Total')
    count_metrics = count_metrics.merge(da_total, on=['EML_Presence', 'eml_da_type'])
    count_metrics['Probability'] = count_metrics['Count'] / count_metrics['DA_Total'] * 100

    # 3. Calculate compliance rate for each DA type (Compliance count / Total valid PRs for the DA type)
    compliance_rate = []
    for (eml_presence, da_type), group in df.groupby(['EML_Presence', 'eml_da_type']):
        total_pr = len(group)
        compliance_count = len(group[group['EML_result'] == 'Compliance'])
        resistance_count = len(group[group['EML_result'] == 'Resistance'])
        compliance_rate.append({
            'EML_Presence': eml_presence,
            'eml_da_type': da_type,
            'Compliance_Rate': (compliance_count / total_pr) * 100 if total_pr > 0 else 0,
            'Resistance_Rate': (resistance_count / total_pr) * 100 if total_pr > 0 else 0,
            'Total_PR': total_pr
        })
    compliance_rate_df = pd.DataFrame(compliance_rate)

    return count_metrics, compliance_rate_df

# Execute basic statistics calculation
count_metrics, compliance_rate_df = calculate_basic_metrics(df_matched)

# ----------------------
# Step 3: Conversion Rate Analysis (PR_Status_before → EML_result)
# ----------------------
def calculate_conversion_rate(df):
    """Calculate PR conversion rate by DA type"""
    # Filter records with PR_Status_before (exclude NaN)
    df_conversion = df[df['PR_Status_before'].isin(VALID_PR_LABELS)].copy()

    # Statistics of conversion matrix by DA type
    conversion_stats = []
    for da_type, group in df_conversion.groupby('eml_da_type'):
        total = len(group)
        # Four conversion types
        cr_cc = len(group[(group['PR_Status_before'] == 'Compliance') & (group['EML_result'] == 'Compliance')])
        cr_cr = len(group[(group['PR_Status_before'] == 'Compliance') & (group['EML_result'] == 'Resistance')])
        cr_rc = len(group[(group['PR_Status_before'] == 'Resistance') & (group['EML_result'] == 'Compliance')])
        cr_rr = len(group[(group['PR_Status_before'] == 'Resistance') & (group['EML_result'] == 'Resistance')])

        conversion_stats.append({
            'DA_Type': da_type,
            'Total_Conversion': total,
            'Compliance→Compliance': cr_cc / total * 100 if total > 0 else 0,
            'Compliance→Resistance': cr_cr / total * 100 if total > 0 else 0,
            'Resistance→Compliance': cr_rc / total * 100 if total > 0 else 0,
            'Resistance→Resistance': cr_rr / total * 100 if total > 0 else 0
        })

    return pd.DataFrame(conversion_stats), df_conversion

# Execute conversion rate analysis
conversion_rate_df, df_conversion = calculate_conversion_rate(df_matched)

# ----------------------
# Step 4: Visualization (Core Results)
# ----------------------
# 4.1 Bar chart of compliance rate by DA type
plt.figure(figsize=(12, 6))
da_types = compliance_rate_df['eml_da_type'].unique()
rates = compliance_rate_df['Compliance_Rate'].values
counts = compliance_rate_df['Total_PR'].values

# Plot compliance rate bars
bars = plt.bar(da_types, rates, color='#1f77b4', alpha=0.8)
# Annotate count and compliance rate on bars
for bar, rate, count in zip(bars, rates, counts):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{rate:.1f}%\n(n={count})', ha='center', va='bottom', fontsize=9)

plt.title('Compliance Rate by DA Type (EML Present)', fontsize=14, pad=20)
plt.xlabel('DA Type', fontsize=12)
plt.ylabel('Compliance Rate (%)', fontsize=12)
plt.ylim(0, 100)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('exp3_compliance_rate_by_da.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------
# 4.2 NEW: Resistance Rate by DA Type
# ----------------------
plt.figure(figsize=(12, 6))
da_types = compliance_rate_df['eml_da_type'].unique()
resist_rates = compliance_rate_df['Resistance_Rate'].values
counts = compliance_rate_df['Total_PR'].values

bars = plt.bar(da_types, resist_rates, color='#ff7f0e', alpha=0.8)
for bar, rate, count in zip(bars, resist_rates, counts):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{rate:.1f}%\n(n={count})', ha='center', va='bottom', fontsize=9)

plt.title('Resistance Rate by DA Type (EML Present)', fontsize=14, pad=20)
plt.xlabel('DA Type', fontsize=12)
plt.ylabel('Resistance Rate (%)', fontsize=12)
plt.ylim(0, 100)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('exp3_resistance_rate_by_da.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------
# 4.3 PR Conversion Rate Heatmap
# ----------------------
if len(conversion_rate_df) > 0:
    conversion_matrix = conversion_rate_df[['Compliance→Compliance', 'Compliance→Resistance',
                                           'Resistance→Compliance', 'Resistance→Resistance']].values
    da_labels = conversion_rate_df['DA_Type'].values
    transition_labels = ['Compliance→Compliance', 'Compliance→Resistance',
                         'Resistance→Compliance', 'Resistance→Resistance']

    plt.figure(figsize=(10, 8))
    im = plt.imshow(conversion_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=100)

    for i in range(len(da_labels)):
        for j in range(len(transition_labels)):
            plt.text(j, i, f'{conversion_matrix[i, j]:.1f}%',
                     ha="center", va="center", color="black", fontsize=9)

    plt.xticks(np.arange(len(transition_labels)), transition_labels, rotation=15, ha='right')
    plt.yticks(np.arange(len(da_labels)), da_labels)
    plt.xlabel('PR Transition Type', fontsize=12)
    plt.ylabel('DA Type', fontsize=12)
    plt.title('PR Conversion Rate by DA Type (Before → After EML)', fontsize=14, pad=20)

    cbar = plt.colorbar(im)
    cbar.set_label('Conversion Rate (%)', fontsize=10)
    plt.tight_layout()
    plt.savefig('exp3_conversion_rate_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

# ----------------------
# Step 5: Output All Statistical Results
# ----------------------
print("=== Experiment 3 Step 2 Three-Dimensional Basic Statistics Results ===")
print("\n1. Count and Probability by Combination (EML Presence × DA Type × PR Type):")
print(count_metrics.round(2))

print("\n2. Compliance & Resistance Rate by DA Type:")
print(compliance_rate_df.round(2))

print("\n=== Experiment 3 Step 2 PR Conversion Rate Analysis Results ===")
print(f"Number of EML records with PR status before and after: {len(df_conversion)}")
print("\nPR Conversion Rate by DA Type (Before EML → After EML):")
print(conversion_rate_df.round(2))

# ----------------------
# Step 6: Save All Results to Excel
# ----------------------
with pd.ExcelWriter('exp3_final_results.xlsx', engine='openpyxl') as writer:
    count_metrics.to_excel(writer, sheet_name='Basic_Count_Metrics', index=False)
    compliance_rate_df.to_excel(writer, sheet_name='Compliance_Resistance_Rate', index=False)
    conversion_rate_df.to_excel(writer, sheet_name='Conversion_Rate', index=False)
    df_matched.to_excel(writer, sheet_name='Matched_EML_PR_Data', index=False)

print("\nAll results saved to: exp3_final_results.xlsx")