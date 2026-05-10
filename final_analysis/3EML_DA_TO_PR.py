import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load dataset
df = pd.read_excel("all_turns_data_with_EML_DA_PR.xlsx")

# Get previous turn features (consistent logic)
df["prev_EML"] = df.groupby("Dialogue_ID")["EML_label"].shift(1)
df["prev_DA"] = df.groupby("Dialogue_ID")["DA_label"].shift(1)

# Keep valid consecutive turn pairs
df_valid = df.dropna(subset=["prev_EML", "prev_DA"]).copy()

# Define DA groups
DA_POSITIVE = {"pa", "sp"}
DA_NEGATIVE = {"pr", "sn", "ic", "dc"}

# Classify previous DA (supports comma-separated labels)
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

df_valid["DA_group"] = df_valid["prev_DA"].apply(classify_da_group_multi)

# Keep only DA+ and DA-
df_final = df_valid[df_valid["DA_group"].isin(["DA+", "DA-"])].copy()

# --------------------------
# Calculate 8 joint probabilities
# --------------------------
def calc_prob(df, da_group, em_val, pr_type):
    subset = df[(df["DA_group"] == da_group) & (df["prev_EML"] == em_val)]
    total = len(subset)
    if total == 0:
        return 0.0, 0
    cnt = len(subset[subset["PR_label"] == pr_type])
    return cnt / total, total

p_c_daP_em0, n0 = calc_prob(df_final, "DA+", 0, "Compliance")
p_c_daP_em1, n1 = calc_prob(df_final, "DA+", 1, "Compliance")
p_c_daM_em0, n2 = calc_prob(df_final, "DA-", 0, "Compliance")
p_c_daM_em1, n3 = calc_prob(df_final, "DA-", 1, "Compliance")

p_r_daP_em0, n4 = calc_prob(df_final, "DA+", 0, "Resistance")
p_r_daP_em1, n5 = calc_prob(df_final, "DA+", 1, "Resistance")
p_r_daM_em0, n6 = calc_prob(df_final, "DA-", 0, "Resistance")
p_r_daM_em1, n7 = calc_prob(df_final, "DA-", 1, "Resistance")

# --------------------------
# Chi-square test for (DA+EML) group vs PR
# --------------------------
# Create joint group: DA_EML
df_final["DA_EML"] = df_final["DA_group"] + "_EML" + df_final["prev_EML"].astype(int).astype(str)

# Contingency table: joint group × PR
contingency_table = pd.crosstab(df_final["DA_EML"], df_final["PR_label"])
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

# --------------------------
# Output
# --------------------------
print("=== Joint Conditional Probability Results (DA × EML → PR) ===")
print(f"P(PR = Compliance | DA+, EML=0) = {p_c_daP_em0:.4f} (N={n0})")
print(f"P(PR = Compliance | DA+, EML=1) = {p_c_daP_em1:.4f} (N={n1})")
print(f"P(PR = Compliance | DA-, EML=0) = {p_c_daM_em0:.4f} (N={n2})")
print(f"P(PR = Compliance | DA-, EML=1) = {p_c_daM_em1:.4f} (N={n3})")
print()
print(f"P(PR = Resistance | DA+, EML=0) = {p_r_daP_em0:.4f} (N={n4})")
print(f"P(PR = Resistance | DA+, EML=1) = {p_r_daP_em1:.4f} (N={n5})")
print(f"P(PR = Resistance | DA-, EML=0) = {p_r_daM_em0:.4f} (N={n6})")
print(f"P(PR = Resistance | DA-, EML=1) = {p_r_daM_em1:.4f} (N={n7})")

print("\n============================================================")
print("=== Chi-Square Test for (DA × EML) → PR Association ===")
print("============================================================")
print(f"Chi-Square Statistic:  {chi2:.4f}")
print(f"Degree of Freedom:     {dof}")
print(f"p-value:               {p_value:.6f}")

sig = "significant" if p_value < 0.05 else "not significant"
print(f"\n→ Association is {sig} (p {'<' if p_value < 0.05 else '>='} 0.05)")