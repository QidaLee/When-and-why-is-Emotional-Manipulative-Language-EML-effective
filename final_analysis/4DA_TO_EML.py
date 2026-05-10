import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load dataset
df = pd.read_excel("all_turns_data_with_EML_DA_PR.xlsx")

# ===================== Experiment 4 =====================
# CORRECT LOGIC:
# PREVIOUS TURN = DA (DA+ / DA-)
# CURRENT TURN  = EML (0 / 1)
# =========================================================

# 取前一句的 DA（和之前所有实验完全统一）
df["prev_DA"] = df.groupby("Dialogue_ID")["DA_label"].shift(1)

# 只保留有效连续对话对
df_valid = df.dropna(subset=["prev_DA"]).copy()

# 定义 DA 分组
DA_POSITIVE = {"pa", "sp"}
DA_NEGATIVE = {"pr", "sn", "ic", "dc"}

# 支持逗号分隔多标签分类
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

# 给【前一句DA】分类
df_valid["DA_group"] = df_valid["prev_DA"].apply(classify_da_group_multi)

# 只保留 DA+ / DA-（和实验风格统一）
df_filtered = df_valid[df_valid["DA_group"].isin(["DA+", "DA-"])].copy()

# 计算条件概率 P(EML | DA_group)
def calc_prob(df, da_group, em_val):
    subset = df[df["DA_group"] == da_group]
    total = len(subset)
    if total == 0:
        return 0.0, 0
    cnt = len(subset[subset["EML_label"] == em_val])
    return cnt / total, total

# 计算 4 个核心概率
p_em0_daP, n_daP = calc_prob(df_filtered, "DA+", 0)
p_em1_daP, _     = calc_prob(df_filtered, "DA+", 1)
p_em0_daM, n_daM = calc_prob(df_filtered, "DA-", 0)
p_em1_daM, _     = calc_prob(df_filtered, "DA-", 1)

# 卡方检验：前DA → 当前EML
cm_chi = pd.crosstab(df_filtered["DA_group"], df_filtered["EML_label"])
chi2, p_value, dof, expected = chi2_contingency(cm_chi)

# 输出结果
print("=== Experiment 4: Conditional Probability Results (DA → EML) ===")
print(f"P(EML = 0 | DA = DA+) = {p_em0_daP:.4f} (N={n_daP})")
print(f"P(EML = 1 | DA = DA+) = {p_em1_daP:.4f}")
print(f"P(EML = 0 | DA = DA-) = {p_em0_daM:.4f} (N={n_daM})")
print(f"P(EML = 1 | DA = DA-) = {p_em1_daM:.4f}")

print("\n=== Association Metrics (DA → EML) ===")
print(f"Chi-Square Statistic:  {chi2:.4f}")
print(f"p-value:               {p_value:.6f}")
sig = "significant" if p_value < 0.05 else "not significant"
print(f"\n→ Association is {sig} (p {'<' if p_value < 0.05 else '>='} 0.05)")
