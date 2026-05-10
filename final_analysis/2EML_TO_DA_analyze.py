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