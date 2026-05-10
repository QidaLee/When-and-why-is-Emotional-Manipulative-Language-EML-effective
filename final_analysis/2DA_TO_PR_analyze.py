import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load dataset
df = pd.read_excel("all_turns_data_with_EML_DA_PR.xlsx")

# Get previous DA label (same logic as EML)
df["prev_DA"] = df.groupby("Dialogue_ID")["DA_label"].shift(1)
df_valid = df.dropna(subset = ["prev_DA"]).copy()

# Define DA groups
DA_POSITIVE = {"pa", "sp"}
DA_NEGATIVE = {"pr", "sn", "ic", "dc"}


# ---------------------------------------------------------------------
#  CORRECT: Split comma-separated DA labels, check ANY match in group
# ---------------------------------------------------------------------
def classify_da_group_multi(label):
    if pd.isna(label):
        return "DAo"
    # Split by comma and strip spaces
    da_list = [d.strip() for d in str(label).split(",")]
    has_pos = any(d in DA_POSITIVE for d in da_list)
    has_neg = any(d in DA_NEGATIVE for d in da_list)

    if has_pos:
        return "DA+"
    elif has_neg:
        return "DA-"
    else:
        return "DAo"


# Apply to PREVIOUS DA
df_valid["DA_group"] = df_valid["prev_DA"].apply(classify_da_group_multi)

# Calculate group sizes
total_da_plus = len(df_valid[df_valid["DA_group"] == "DA+"])
total_da_minus = len(df_valid[df_valid["DA_group"] == "DA-"])


# Probability calculation function
def calculate_prob(df, target_group, pr_type):
    subset = df[df["DA_group"] == target_group]
    total = len(subset)
    if total == 0:
        return 0.0
    return len(subset[subset["PR_label"] == pr_type]) / total


p_comp_da_plus = calculate_prob(df_valid, "DA+", "Compliance")
p_comp_da_minus = calculate_prob(df_valid, "DA-", "Compliance")
p_resi_da_plus = calculate_prob(df_valid, "DA+", "Resistance")
p_resi_da_minus = calculate_prob(df_valid, "DA-", "Resistance")

# Chi-square (DA+ vs DA-)
df_chi = df_valid[df_valid["DA_group"].isin(["DA+", "DA-"])]
cm_chi = pd.crosstab(df_chi["DA_group"], df_chi["PR_label"])
chi2, p_value, dof, expected = chi2_contingency(cm_chi)

# Output
print("=== Conditional Probability Results (DA → PR) ===")
print(f"P(PR = Compliance | DA+) = {p_comp_da_plus:.4f}")
print(f"P(PR = Compliance | DA-) = {p_comp_da_minus:.4f}")
print(f"P(PR = Resistance | DA+) = {p_resi_da_plus:.4f}")
print(f"P(PR = Resistance | DA-) = {p_resi_da_minus:.4f}")

print("\n=== Sample Size Statistics ===")
print(f"Total pairs with previous DA+: {total_da_plus}")
print(f"Total pairs with previous DA-: {total_da_minus}")

print("\n=== Association Metrics (DA → PR) ===")
print(f"Chi-Square Statistic:  {chi2:.4f}")
print(f"p-value:               {p_value:.6f}")
sig = "significant" if p_value < 0.05 else "not significant"
print(f"\n→ Association is {sig} (p {'<' if p_value < 0.05 else '>='} 0.05)")
