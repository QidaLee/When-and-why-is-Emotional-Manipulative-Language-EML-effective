import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# ===================== Configuration =====================
DA_COL = "DA_label"
EML_COL = "EML_label"
FILE_PATH = "output_data/all_turns_data_with_EML_DA_PR.xlsx"

# ===================== 1. Load Data =====================
df = pd.read_excel(FILE_PATH)
df = df.dropna(subset=[DA_COL, EML_COL])
df[EML_COL] = df[EML_COL].astype(int)

# ===================== 2. Contingency Table =====================
crosstab = pd.crosstab(df[DA_COL], df[EML_COL])
crosstab_pct = pd.crosstab(df[DA_COL], df[EML_COL], normalize='index') * 100

# Rename columns
crosstab.columns = ["No_EML", "EML"]
crosstab_pct.columns = ["No_EML (%)", "EML (%)"]

# Combine count and percentage
result_table = pd.concat([crosstab, crosstab_pct], axis=1)
result_table["EML_Ratio (%)"] = (result_table["EML"] /
                                 (result_table["No_EML"] + result_table["EML"]) * 100)
result_table = result_table.sort_values("EML_Ratio (%)", ascending=False)

# ===================== 3. Chi-Square Test =====================
chi2, p_val, dof, expected = chi2_contingency(crosstab)

print("=" * 80)
print("              EML & DA Association Analysis (Chi-Square Test)")
print("=" * 80)
print(f"Chi-square statistic = {chi2:.2f}")
print(f"P-value              = {p_val:.12f}")
print(f"Degrees of freedom    = {dof}")

if p_val < 0.05:
    print("Result: Statistically significant association (p < 0.05)")
else:
    print("Result: No significant association")
print("=" * 80)

# ===================== 4. Print Result Table =====================
print("\nEML Occurrence by DA Type (Sorted by EML Ratio)")
print(result_table.round(2))

# Save to Excel
result_table.to_excel("EML_DA_Association_Results.xlsx", index=True)

# ===================== 5. Plot 1: Stacked Bar Chart (Colored Version) =====================
plt.figure(figsize=(15, 7))
da_names = result_table.index
no_eml = result_table["No_EML (%)"]
eml = result_table["EML (%)"]

x = np.arange(len(da_names))
width = 0.7

# Color gradients for each bar
colors_no_eml = plt.cm.Blues(np.linspace(0.3, 0.9, len(da_names)))
colors_eml = plt.cm.Oranges(np.linspace(0.3, 0.9, len(da_names)))

# Use different colors for each DA category
bars1 = plt.bar(x, no_eml, width, label="No EML", color=colors_no_eml, edgecolor="white", linewidth=1.5)
bars2 = plt.bar(x, eml, width, bottom=no_eml, label="EML", color=colors_eml, edgecolor="white", linewidth=1.5)

plt.xticks(x, da_names, rotation=45, ha="right", fontsize=11)
plt.ylabel("Percentage (%)", fontsize=12)
plt.title("Distribution of EML Across Different DA Types", fontsize=14, pad=15)
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=11)

# Add value labels on bars
for i, (no, em) in enumerate(zip(no_eml, eml)):
    if no > 5:
        plt.text(i, no/2, f'{no:.1f}%', ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    if em > 5:
        plt.text(i, no + em/2, f'{em:.1f}%', ha='center', va='center', fontsize=8, color='white', fontweight='bold')

plt.tight_layout()
plt.savefig("EML_DA_Stacked_Bar_Colorful.png", dpi=300, bbox_inches="tight")
plt.show()

# ===================== 6. Plot 2: Top DA with Highest EML Ratio (Colored Gradient) =====================
plt.figure(figsize=(10, 6))
top_da = result_table["EML_Ratio (%)"].head(10).sort_values()

# Create color gradient based on values
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_da)))

bars = plt.barh(range(len(top_da)), top_da.values, color=colors, edgecolor='white', linewidth=2)

plt.yticks(range(len(top_da)), top_da.index, fontsize=11)
plt.xlabel("EML Ratio (%)", fontsize=12)
plt.title("Top 10 DA Types with Highest EML Proportion", fontsize=14, pad=15)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, top_da.values)):
    plt.text(val + 0.5, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig("EML_DA_Top_Ratio_Colorful.png", dpi=300, bbox_inches="tight")
plt.show()

# ===================== 7. Plot 3: Heatmap =====================
plt.figure(figsize=(10, 8))
sns.heatmap(
    crosstab_pct.sort_values("EML (%)", ascending=False),
    annot=True, fmt=".1f", cmap="Reds", linewidths=0.5, annot_kws={"size": 10}
)
plt.title("EML Percentage by DA Type (%)", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig("EML_DA_Heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

# ===================== 8. Additional: Horizontal Bar Chart with All DAs Colored =====================
plt.figure(figsize=(12, 8))

# Create color gradient based on EML ratio
all_da_ratio = result_table["EML_Ratio (%)"].sort_values()
colors_all = plt.cm.plasma(np.linspace(0.1, 0.9, len(all_da_ratio)))

bars = plt.barh(range(len(all_da_ratio)), all_da_ratio.values,
                color=colors_all, edgecolor='white', linewidth=1.5)

plt.yticks(range(len(all_da_ratio)), all_da_ratio.index, fontsize=10)
plt.xlabel("EML Ratio (%)", fontsize=12)
plt.title("All DA Types: EML Proportion (Colored by Value)", fontsize=14, pad=15)

# Add colorbar to show value mapping
sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=all_da_ratio.min(), vmax=all_da_ratio.max()))
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label('EML Ratio (%)', fontsize=10)

# Add value labels for bars with higher values
for bar, val in zip(bars, all_da_ratio.values):
    if val > 5:  # Only show labels for values > 5%
        plt.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig("EML_DA_All_Ratio_Colorful.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nAll figures and results saved successfully!")
print("enerated files:")
print("   - EML_DA_Association_Results.xlsx")
print("   - EML_DA_Stacked_Bar_Colorful.png")
print("   - EML_DA_Top_Ratio_Colorful.png")
print("   - EML_DA_Heatmap.png")
print("   - EML_DA_All_Ratio_Colorful.png")