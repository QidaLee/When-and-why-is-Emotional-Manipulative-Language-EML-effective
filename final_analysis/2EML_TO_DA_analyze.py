import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load dataset
df = pd.read_excel("all_turns_data_with_EML_DA_PR.xlsx")

# CORRECT: Previous EML → Current DA
df["prev_EML"] = df.groupby("Dialogue_ID")["EML_label"].shift(1)
df_valid = df.dropna(subset=["prev_EML"]).copy()

# Define DA groups
DA_POSITIVE = {"pa", "sp"}
DA_NEGATIVE = {"pr", "sn", "ic", "dc"}

# Classify CURRENT DA (supports comma-separated)
def classify_da_group_multi(label):
    if pd.isna(label):
        return "DAo"
    da_list = [d.strip() for d in str(label).split(",")]
    has_pos = any(d in DA_POSITIVE for d in da_list)
    has_neg = any(d in DA_NEGATIVE for d in da_list)
    if has_pos:
        return "DA+"
    elif has_neg:
        return "DA-"
    else:
        return "DAo"

df_valid["DA_group"] = df_valid["DA_label"].apply(classify_da_group_multi)

# --------------------------
# NO FILTERING！保留 DA+ DA- DAo 全部！
# --------------------------

# 计算真实概率（基于全部数据，不删除 DAo）
def calc_real_prob(df, em_val, target_da):
    subset = df[df["prev_EML"] == em_val]
    total = len(subset)
    if total == 0:
        return 0.0, 0
    count = len(subset[subset["DA_group"] == target_da])
    return count / total, total

# 计算 4 个真实条件概率
p_daP_em0, n_em0 = calc_real_prob(df_valid, 0, "DA+")
p_daP_em1, n_em1 = calc_real_prob(df_valid, 1, "DA+")
p_daM_em0, _     = calc_real_prob(df_valid, 0, "DA-")
p_daM_em1, _     = calc_real_prob(df_valid, 1, "DA-")

# 卡方检验（EML → DA_group，包含全部三类）
cm_chi = pd.crosstab(df_valid["prev_EML"], df_valid["DA_group"])
chi2, p_value, dof, expected = chi2_contingency(cm_chi)

# 输出正确结果
print("=== Conditional Probability Results (EML → DA) ===")
print(f"P(DA = DA+ | EML = 0) = {p_daP_em0:.4f}")
print(f"P(DA = DA+ | EML = 1) = {p_daP_em1:.4f}")
print(f"P(DA = DA- | EML = 0) = {p_daM_em0:.4f}")
print(f"P(DA = DA- | EML = 1) = {p_daM_em1:.4f}")

print("\n=== Sample Size Statistics ===")
print(f"Total pairs with previous EML=0: {n_em0}")
print(f"Total pairs with previous EML=1: {n_em1}")

print("\n=== Association Metrics (EML → DA) ===")
print(f"Chi-Square Statistic:  {chi2:.4f}")
print(f"p-value:               {p_value:.6f}")
sig = "significant" if p_value < 0.05 else "not significant"
print(f"\n→ Association is {sig} (p {'<' if p_value < 0.05 else '>='} 0.05)")

# ==================== NEW: Merged Table with All DA Categories ====================

print("\n" + "=" * 80)
print("=== Merged Analysis Table for ALL Original DA Categories ===")
print("=" * 80)


# Get all possible DA labels (accounting for comma-separated multi-labels)
def expand_da_labels(df, col="DA_label"):
    """Expand comma-separated multi-labels into individual labels"""
    all_labels = []
    for val in df[col].dropna():
        labels = [d.strip() for d in str(val).split(",")]
        all_labels.extend(labels)
    return sorted(set(all_labels))


# Get all unique DA categories
all_da_categories = expand_da_labels(df_valid)

# Store results for each category
results_data = []

for da_cat in all_da_categories:
    # Calculate P(DA = da_cat | EML = 0)
    subset_em0 = df_valid[df_valid["prev_EML"] == 0]
    total_em0 = len(subset_em0)
    if total_em0 > 0:
        count_em0 = sum(da_cat in [d.strip() for d in str(row).split(",")]
                        for row in subset_em0["DA_label"])
        prob_em0 = count_em0 / total_em0
    else:
        prob_em0 = 0.0
        count_em0 = 0

    # Calculate P(DA = da_cat | EML = 1)
    subset_em1 = df_valid[df_valid["prev_EML"] == 1]
    total_em1 = len(subset_em1)
    if total_em1 > 0:
        count_em1 = sum(da_cat in [d.strip() for d in str(row).split(",")]
                        for row in subset_em1["DA_label"])
        prob_em1 = count_em1 / total_em1
    else:
        prob_em1 = 0.0
        count_em1 = 0

    # Chi-square test for this DA category
    df_valid[f"has_{da_cat}"] = df_valid["DA_label"].apply(
        lambda x: da_cat in [d.strip() for d in str(x).split(",")] if pd.notna(x) else False
    )

    contingency = pd.crosstab(df_valid["prev_EML"], df_valid[f"has_{da_cat}"])

    # Calculate p-value if contingency table is 2x2 with sufficient data
    if contingency.shape == (2, 2) and contingency.min().min() > 0:
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        p_val_display = f"{p_value:.6f}"
        significant = "*" if p_value < 0.05 else ""
    else:
        p_value = np.nan
        p_val_display = "N/A"
        significant = ""

    results_data.append({
        "DA": da_cat,
        "P(DA|EML=0)": prob_em0,
        "P(DA|EML=1)": prob_em1,
        "Count(EML=0)": count_em0,
        "Count(EML=1)": count_em1,
        "p-value": p_val_display,
        "sig": significant
    })

# Create DataFrame for better display
results_df = pd.DataFrame(results_data)

# Display merged table
print("\n" + results_df.to_string(index = False))
print("\nNote: * indicates p < 0.05 (statistically significant)")

# Optional: Save to CSV
# results_df.to_csv("da_analysis_results.csv", index=False)
# print("\nResults saved to 'da_analysis_results.csv'")

# Summary of significant categories
sig_df = results_df[results_df["sig"] == "*"]
if len(sig_df) > 0:
    print(f"\n=== Summary: {len(sig_df)} DA categories significantly associated with previous EML (p < 0.05) ===")
    print(sig_df[["DA", "P(DA|EML=0)", "P(DA|EML=1)", "p-value"]].to_string(index = False))
else:
    print("\nNo DA categories showed significant association with previous EML (p < 0.05)")