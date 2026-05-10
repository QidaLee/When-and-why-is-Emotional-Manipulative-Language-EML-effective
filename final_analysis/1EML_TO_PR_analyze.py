import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load dataset
df = pd.read_excel("all_turns_data_with_EML_DA_PR.xlsx")

# Get previous EML label within the same dialogue
df["prev_EML"] = df.groupby("Dialogue_ID")["EML_label"].shift(1)

# Keep only valid turn pairs
df_pairs = df.dropna(subset=["prev_EML"]).copy()

# Contingency Matrix
cm = pd.crosstab(df_pairs["prev_EML"].astype(int), df_pairs["PR_label"])

# Calculate totals
total_eml0 = cm.loc[0].sum()
total_eml1 = cm.loc[1].sum()

# Conditional Probabilities
p_comp_given_0 = cm.loc[0, "Compliance"] / total_eml0
p_comp_given_1 = cm.loc[1, "Compliance"] / total_eml1
p_resi_given_0 = cm.loc[0, "Resistance"] / total_eml0
p_resi_given_1 = cm.loc[1, "Resistance"] / total_eml1

# Chi-Square Test
chi2, p_value, dof, expected = chi2_contingency(cm)

# Output exactly what you need
print("=== Conditional Probability Results ===")
print(f"P(PR = Compliance | EML = 0) = {p_comp_given_0:.4f}")
print(f"P(PR = Compliance | EML = 1) = {p_comp_given_1:.4f}")
print(f"P(PR = Resistance | EML = 0) = {p_resi_given_0:.4f}")
print(f"P(PR = Resistance | EML = 1) = {p_resi_given_1:.4f}")

print("\n=== Sample Size Statistics ===")
print(f"Total pairs with previous EML=0: {total_eml0}")
print(f"Total pairs with previous EML=1: {total_eml1}")

print("\n=== Association Metrics (EML → PR) ===")
print(f"Chi-Square Statistic:  {chi2:.4f}")
print(f"p-value:               {p_value:.6f}")

sig = "significant" if p_value < 0.05 else "not significant"
print(f"\n→ Association is {sig} (p {'<' if p_value < 0.05 else '>='} 0.05)")