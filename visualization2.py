import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set English font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

# Load data
file_path = 'output_data/all_turns_data_with_EML_DA_PR.xlsx'
df = pd.read_excel(file_path)

df_clean = df.dropna(subset=['EML_label', 'DA_label', 'PR_label']).copy()

# 4. Cross-Model Analysis: EML vs PR Label Correlation
# Create contingency table
eml_pr_crosstab = pd.crosstab(df_clean['EML_label'], df_clean['PR_label'], margins=True)
print("=== EML vs PR Label Contingency Table ===")
print(eml_pr_crosstab)

# Calculate percentage distribution
eml_pr_pct = pd.crosstab(df_clean['EML_label'], df_clean['PR_label'], normalize='index') * 100
print("\n=== EML vs PR Label Percentage Distribution (Row-wise) ===")
print(round(eml_pr_pct, 2))

# Check unique values in EML_label
unique_eml = df_clean['EML_label'].unique()
print(f"\nUnique EML_label values in data: {sorted(unique_eml)}")

# Visualize EML vs PR correlation
plt.figure(figsize=(12, 7))

# 修复：正确获取所有EML值（排除'All'行）
# 方法1：使用drop方法排除'All'行
eml_pct_no_margin = eml_pr_pct.drop('All', errors='ignore')

# 或者方法2：使用布尔索引
# eml_values = eml_pr_pct.index[eml_pr_pct.index != 'All']
# eml_pct_no_margin = eml_pr_pct.loc[eml_values]

print(f"\nEML values for plotting: {eml_pct_no_margin.index.tolist()}")

x = np.arange(len(eml_pct_no_margin.index))
width = 0.6

# Get PR labels (排除'All'列)
pr_labels = [col for col in eml_pr_pct.columns if col != 'All']
print(f"PR labels for plotting: {pr_labels}")

colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green for Neutral, Orange for Mixed, Red for Polar

# 确保颜色数量足够
if len(pr_labels) > len(colors):
    additional_colors = plt.cm.Set3(np.linspace(0, 1, len(pr_labels) - len(colors)))
    colors.extend(additional_colors)

# Calculate bottom positions for stacking
bottom = np.zeros(len(x))
bars_list = []

for i, pr_label in enumerate(pr_labels):
    values = eml_pct_no_margin[pr_label].values
    bars = plt.bar(x, values, width, bottom=bottom, label=pr_label,
                   color=colors[i % len(colors)], alpha=0.8, edgecolor='black', linewidth=1)
    bars_list.append(bars)
    bottom += values

# Add percentage labels inside bars
for bars in bars_list:
    for bar in bars:
        height = bar.get_height()
        if height > 5:  # Only show labels for segments >5% to avoid clutter
            plt.text(bar.get_x() + bar.get_width()/2.,
                     bar.get_y() + height/2.,
                     f'{height:.1f}%',
                     ha='center', va='center', fontsize=10, fontweight='bold', color='white')

plt.title('EML vs PR Label Correlation (Percentage Distribution)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('EML Label', fontsize=12, fontweight='bold')
plt.ylabel('Percentage (%)', fontsize=12, fontweight='bold')

# 创建x轴标签
x_labels = [f'EML = {int(idx)}' for idx in eml_pct_no_margin.index]
plt.xticks(x, x_labels)

plt.ylim(0, 100)
plt.legend(title='PR Label', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('eml_vs_pr_correlation.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nEML vs PR Correlation Chart saved successfully")

# Key observations with error handling
print("\n=== Key Observations: EML vs PR ===")
if 0 in eml_pr_pct.index:
    eml_0_neutral_pct = eml_pr_pct.loc[0, 'Neutral'] if 'Neutral' in eml_pr_pct.columns else 0
    print(f"Non-Emotional (EML=0) samples are {eml_0_neutral_pct:.2f}% Neutral in PR")
else:
    print("No EML=0 samples found in data")

if 1 in eml_pr_pct.index:
    eml_1_neutral_pct = eml_pr_pct.loc[1, 'Neutral'] if 'Neutral' in eml_pr_pct.columns else 0
    print(f"Emotional (EML=1) samples are {eml_1_neutral_pct:.2f}% Neutral in PR")
else:
    print("No EML=1 samples found in data")

# Calculate difference if both exist
if 0 in eml_pr_pct.index and 1 in eml_pr_pct.index and 'Neutral' in eml_pr_pct.columns:
    print(f"Difference in Neutral PR percentage: {abs(eml_pr_pct.loc[0, 'Neutral'] - eml_pr_pct.loc[1, 'Neutral']):.2f} percentage points")

# 额外信息：显示数据分布
print(f"\n=== Data Distribution ===")
print(f"Total samples: {len(df_clean)}")
print(f"EML=0 samples: {len(df_clean[df_clean['EML_label'] == 0])} ({len(df_clean[df_clean['EML_label'] == 0])/len(df_clean)*100:.2f}%)")
print(f"EML=1 samples: {len(df_clean[df_clean['EML_label'] == 1])} ({len(df_clean[df_clean['EML_label'] == 1])/len(df_clean)*100:.2f}%)")

# 5. Cross-Model Analysis: DA vs PR Label Correlation (Focus on Top 3 DA Labels)
# Identify top 3 DA labels by frequency
top3_da_labels = df_clean['DA_label'].value_counts().nlargest(3).index.tolist()
print(f"Top 3 DA Labels by Frequency: {top3_da_labels}")

# Filter data to only include top 3 DA labels
df_top3_da = df_clean[df_clean['DA_label'].isin(top3_da_labels)].copy()
print(
    f"Number of samples with top 3 DA labels: {len(df_top3_da)} ({len(df_top3_da) / len(df_clean) * 100:.2f}% of total)")

# Create contingency table for top 3 DA vs PR
da_pr_crosstab = pd.crosstab(df_top3_da['DA_label'], df_top3_da['PR_label'], margins = True)
print("\n=== Top 3 DA vs PR Label Contingency Table ===")
print(da_pr_crosstab)

# Calculate percentage distribution (row-wise)
da_pr_pct = pd.crosstab(df_top3_da['DA_label'], df_top3_da['PR_label'], normalize = 'index') * 100
print("\n=== Top 3 DA vs PR Label Percentage Distribution (Row-wise) ===")
print(round(da_pr_pct, 2))

# Visualize Top 3 DA vs PR correlation
plt.figure(figsize = (14, 8))

# 修复：正确排除'All'行和列
# 获取所有DA标签（排除'All'行）
da_labels_for_plot = da_pr_pct.index[da_pr_pct.index != 'All'].tolist()
# 或者使用：da_labels_for_plot = da_pr_pct.drop('All', errors='ignore').index.tolist()

# 获取所有PR标签（排除'All'列）
pr_labels_for_plot = [col for col in da_pr_pct.columns if col != 'All']

print(f"\nDA labels for plotting: {da_labels_for_plot}")
print(f"PR labels for plotting: {pr_labels_for_plot}")

# 创建不包含'All'的数据框用于绘图
da_pct_no_margin = da_pr_pct.loc[da_labels_for_plot, pr_labels_for_plot]

# Prepare data for grouped bar chart
x = np.arange(len(da_labels_for_plot))
width = 0.25
colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green for Neutral, Orange for Mixed, Red for Polar

# 确保颜色数量足够
if len(pr_labels_for_plot) > len(colors):
    additional_colors = plt.cm.Set3(np.linspace(0, 1, len(pr_labels_for_plot) - len(colors)))
    colors.extend(additional_colors)

# Create grouped bars
bars = []
for i, pr_label in enumerate(pr_labels_for_plot):
    values = da_pct_no_margin[pr_label].values
    bar_positions = x + (i - (len(pr_labels_for_plot) - 1) / 2) * width  # 居中对齐
    bar = plt.bar(bar_positions, values, width, label = pr_label,
                  color = colors[i % len(colors)], alpha = 0.8, edgecolor = 'black', linewidth = 1)
    bars.append(bar)

    # Add percentage labels on top of bars
    for j, v in enumerate(values):
        if v > 0:  # 只在非零值上显示标签
            plt.text(bar_positions[j], v + 1,
                     f'{v:.1f}%',
                     ha = 'center', va = 'bottom', fontsize = 9, fontweight = 'bold',
                     rotation = 0 if v < 15 else 0)  # 如果值太小可以考虑旋转，但这里保持水平

plt.title('Top 3 DA Labels vs PR Label Correlation', fontsize = 16, fontweight = 'bold', pad = 20)
plt.xlabel('DA Label (Top 3 by Frequency)', fontsize = 12, fontweight = 'bold')
plt.ylabel('Percentage (%)', fontsize = 12, fontweight = 'bold')
plt.xticks(x, da_labels_for_plot, rotation = 15, ha = 'right')  # 如果标签太长可以旋转
plt.ylim(0, 100)
plt.legend(title = 'PR Label', loc = 'upper right', bbox_to_anchor = (1, 1))
plt.grid(axis = 'y', alpha = 0.3, linestyle = '--')
plt.tight_layout()
plt.savefig('top3_da_vs_pr_correlation.png', dpi = 300, bbox_inches = 'tight')
plt.show()

print("\nTop 3 DA vs PR Correlation Chart saved successfully")

# Calculate PR label distribution differences
print("\n=== Key Observations: Top 3 DA vs PR ===")
for da_label in da_labels_for_plot:
    compliance_pct = da_pct_no_margin.loc[da_label, 'Compliance'] if 'Compliance' in da_pct_no_margin.columns else 0
    neutral_pct = da_pct_no_margin.loc[da_label, 'Neutral'] if 'Neutral' in da_pct_no_margin.columns else 0
    resistance_pct = da_pct_no_margin.loc[da_label, 'Resistance'] if 'Resistance' in da_pct_no_margin.columns else 0
    print(
        f"DA Label '{da_label}': Compliance={compliance_pct:.2f}%, Neutral={neutral_pct:.2f}%, Resistance={resistance_pct:.2f}%")

# 安全检查：确保有数据
if not da_pct_no_margin.empty and len(da_pct_no_margin) > 0:
    # Find DA label with highest compliance and resistance
    if 'Compliance' in da_pct_no_margin.columns:
        highest_compliance_da = da_pct_no_margin['Compliance'].idxmax()
        highest_compliance_pct = da_pct_no_margin['Compliance'].max()
        print(f"\nDA Label with Highest Compliance: '{highest_compliance_da}' ({highest_compliance_pct:.2f}%)")

    if 'Resistance' in da_pct_no_margin.columns:
        highest_resistance_da = da_pct_no_margin['Resistance'].idxmax()
        highest_resistance_pct = da_pct_no_margin['Resistance'].max()
        print(f"DA Label with Highest Resistance: '{highest_resistance_da}' ({highest_resistance_pct:.2f}%)")

    if 'Neutral' in da_pct_no_margin.columns:
        highest_neutral_da = da_pct_no_margin['Neutral'].idxmax()
        highest_neutral_pct = da_pct_no_margin['Neutral'].max()
        print(f"DA Label with Highest Neutral: '{highest_neutral_da}' ({highest_neutral_pct:.2f}%)")
else:
    print("\nNo data available for analysis")

# 额外的统计信息
print(f"\n=== Additional Statistics ===")
total_samples_top3 = len(df_top3_da)
print(f"Total samples in top 3 DA categories: {total_samples_top3}")

# 显示每个DA标签的样本数量
print("\nSample distribution by DA label:")
for da_label in da_labels_for_plot:
    count = len(df_top3_da[df_top3_da['DA_label'] == da_label])
    percentage = (count / total_samples_top3) * 100
    print(f"  {da_label}: {count} samples ({percentage:.2f}%)")

