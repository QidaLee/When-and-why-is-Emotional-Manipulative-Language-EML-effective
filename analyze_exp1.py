import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# Set plot font (English)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ----------------------
# 1. Data Loading and Preprocessing
# ----------------------
# Load dataset
df = pd.read_excel('output_data/all_turns_data_with_EML_DA_PR.xlsx')

# Check data types and missing values for key columns
print("Basic Data Information:")
print(f"Total records: {len(df)}")
print(f"Missing value statistics:\n{df[['Dialogue_ID', 'Turn', 'EML_label', 'PR_label', 'PR_confidence']].isnull().sum()}")

# Ensure Turn and EML_label are numeric
df['Turn'] = pd.to_numeric(df['Turn'], errors='coerce')
df['EML_label'] = pd.to_numeric(df['EML_label'], errors='coerce')

# ----------------------
# 2. Identify First EML Turn per Dialogue
# ----------------------
# Filter records where EML_label = 1
eml_turns = df[df['EML_label'] == 1].copy()

# Group by Dialogue_ID to find the first occurrence of EML in each dialogue
first_eml_per_dialogue = eml_turns.groupby('Dialogue_ID')['Turn'].min().reset_index()
first_eml_per_dialogue.columns = ['Dialogue_ID', 'first_EML_turn']

# Merge first_EML_turn back to main dataframe
df = df.merge(first_eml_per_dialogue, on='Dialogue_ID', how='left')

# For dialogues without EML, set first_EML_turn to infinity
df['first_EML_turn'] = df['first_EML_turn'].fillna(np.inf)

# ----------------------
# 3. Label PR Timing (Before/After EML)
# ----------------------
# Filter records with valid PR labels
pr_data = df[df['PR_label'].notna() & (df['PR_label'] != '')].copy()

# Label whether PR occurs before or after EML
def mark_pr_timing(row):
    if row['Turn'] < row['first_EML_turn']:
        return 'before_EML'
    elif row['Turn'] > row['first_EML_turn']:
        return 'after_EML'
    else:
        return 'same_as_EML'  # Same turn as EML

pr_data['PR_timing'] = pr_data.apply(mark_pr_timing, axis=1)

# Keep only before/after EML records (exclude same turn)
pr_data = pr_data[pr_data['PR_timing'].isin(['before_EML', 'after_EML'])]

# ----------------------
# 4. Statistical Analysis
# ----------------------
# 4.1 Calculate Compliance/Resistance ratios for each group
pr_label_counts = pr_data.groupby(['PR_timing', 'PR_label']).size().unstack(fill_value=0)
print("\n=== PR Label Distribution (Compliance/Resistance) ===")
print(pr_label_counts)

# Calculate percentage ratios
pr_label_ratio = pr_label_counts.div(pr_label_counts.sum(axis=1), axis=0) * 100
print("\n=== PR Label Ratios (%) ===")
print(pr_label_ratio)

# 4.2 Chi-square test for label distribution differences
chi2, p_chi2, dof, expected = chi2_contingency(pr_label_counts)
print(f"\n=== Chi-square Test Results ===")
print(f"Chi2 statistic: {chi2:.4f}, p-value: {p_chi2:.4f}")
print(f"Conclusion: {'Significant difference exists' if p_chi2 < 0.05 else 'No significant difference'} (α=0.05)")

# 4.3 Statistical comparison of PR_confidence between groups
pr_confidence_before = pr_data[pr_data['PR_timing'] == 'before_EML']['PR_confidence'].dropna()
pr_confidence_after = pr_data[pr_data['PR_timing'] == 'after_EML']['PR_confidence'].dropna()

print(f"\n=== PR_confidence Statistics ===")
print(f"PR_confidence before EML: Mean={pr_confidence_before.mean():.4f}, Std={pr_confidence_before.std():.4f}, Sample size={len(pr_confidence_before)}")
print(f"PR_confidence after EML: Mean={pr_confidence_after.mean():.4f}, Std={pr_confidence_after.std():.4f}, Sample size={len(pr_confidence_after)}")

# Mann-Whitney U test (non-parametric test, better for non-normal distributions)
u_stat, p_mw = mannwhitneyu(pr_confidence_before, pr_confidence_after, alternative='two-sided')
print(f"\n=== Mann-Whitney U Test Results ===")
print(f"U statistic: {u_stat:.4f}, p-value: {p_mw:.4f}")
print(f"Conclusion: {'Significant difference exists' if p_mw < 0.05 else 'No significant difference'} (α=0.05)")

# ----------------------
# 5. Visualization (Split into two separate figures)
# ----------------------
# 5.1 Separate Bar Chart: Compliance Ratio Before vs After EML
plt.figure(figsize=(8, 6))
if 'Compliance' in pr_label_ratio.columns:
    compliance_ratios = pr_label_ratio['Compliance']
    ax = compliance_ratios.plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title('Compliance Ratio Before vs After EML', fontsize=14)
    plt.xlabel('PR Timing Relative to EML', fontsize=12)
    plt.ylabel('Compliance Ratio (%)', fontsize=12)
    plt.xticks(rotation=0)
    # Add value labels
    for i, v in enumerate(compliance_ratios):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
# Save bar chart as separate file
plt.tight_layout()
plt.savefig('compliance_ratio_before_after_eml.png', dpi=300, bbox_inches='tight')
plt.show()

# 5.2 Separate Boxplot: PR_confidence Distribution
plt.figure(figsize=(8, 6))
pr_data_for_boxplot = pr_data[pr_data['PR_confidence'].notna()]
boxplot = pr_data_for_boxplot.boxplot(column='PR_confidence', by='PR_timing',
                                      patch_artist=True,
                                      boxprops=dict(facecolor='lightblue'),
                                      medianprops=dict(color='red'))
plt.title('PR_confidence Distribution Before vs After EML', fontsize=14)
plt.xlabel('PR Timing Relative to EML', fontsize=12)
plt.ylabel('PR_confidence', fontsize=12)
plt.suptitle('')  # Remove auto-generated title
plt.grid(axis='y', linestyle='--', alpha=0.7)
# Save boxplot as separate file
plt.tight_layout()
plt.savefig('pr_confidence_distribution_before_after_eml.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------
# 6. Output Results Table
# ----------------------
result_summary = pd.DataFrame({
    'PR_timing': ['before_EML', 'after_EML'],
    'Compliance_ratio': [pr_label_ratio.loc['before_EML', 'Compliance'] if 'before_EML' in pr_label_ratio.index else 0,
                         pr_label_ratio.loc['after_EML', 'Compliance'] if 'after_EML' in pr_label_ratio.index else 0],
    'Resistance_ratio': [pr_label_ratio.loc['before_EML', 'Resistance'] if 'before_EML' in pr_label_ratio.index else 0,
                         pr_label_ratio.loc['after_EML', 'Resistance'] if 'after_EML' in pr_label_ratio.index else 0],
    'PR_confidence_mean': [pr_confidence_before.mean() if len(pr_confidence_before) > 0 else np.nan,
                           pr_confidence_after.mean() if len(pr_confidence_after) > 0 else np.nan],
    'Sample_size': [len(pr_confidence_before), len(pr_confidence_after)]
})

print("\n=== Experiment Results Summary ===")
print(result_summary)

# Save results to Excel
# result_summary.to_excel('eml_temporal_effect_results.xlsx', index=False)