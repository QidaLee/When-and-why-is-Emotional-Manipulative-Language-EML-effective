import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

warnings.filterwarnings('ignore')

# ----------------------
# 1. Data Loading and Preprocessing (Extreme Cleaning)
# ----------------------
# Load Experiment 3 matched data
df_matched = pd.read_csv('exp3_eml_pr_matched_data.csv', encoding = 'utf-8')
df_combined = df_matched.copy()

# Core cleaning: retain only key columns + filter outliers
df_combined = df_combined[['eml_da_type', 'EML_result']].dropna()
# Filter non-target PR results (only retain Compliance/Resistance)
df_combined = df_combined[df_combined['EML_result'].isin(['Compliance', 'Resistance'])]

# 1.1 Dependent Variable (DV) Binarization
df_combined['DV_Compliance'] = df_combined['EML_result'].map({'Compliance': 1, 'Resistance': 0}).astype(int)

# 1.2 Filter DA types with high sample size (Core: avoid collinearity)
# Only retain DA types with sample size ≥5 (further reduce collinearity risk)
da_count = df_combined['eml_da_type'].value_counts()
valid_da_types = da_count[da_count >= 5].index.tolist()
df_combined = df_combined[df_combined['eml_da_type'].isin(valid_da_types)]

# If too few valid DA types, use all (avoid model with no data)
if len(valid_da_types) < 2:
    valid_da_types = da_count.index.tolist()
    df_combined = df_combined[df_combined['eml_da_type'].isin(valid_da_types)]

print(f"✅ Final DA types included in model: {valid_da_types}")
print(f"✅ Sample size per DA type: \n{df_combined['eml_da_type'].value_counts()}")

# ----------------------
# 2. Simplify Model: Avoid Collinearity (Core Fix)
# ----------------------
# Approach: Do not set EML_flag separately (all data contains EML), use DA_type as core IV directly
# One-hot encoding: do not drop reference group (avoid subsequent collinearity), process manually later
da_dummies = pd.get_dummies(df_combined['eml_da_type'], prefix = 'DA', dtype = int)

# Merge data
df_model = pd.concat([
    df_combined[['DV_Compliance']].reset_index(drop = True),
    da_dummies.reset_index(drop = True)
], axis = 1)

# Manually drop one reference group (select DA type with largest sample size, e.g., ap)
ref_da = da_count.idxmax()  # Automatically select DA with largest sample size as reference group
if f'DA_{ref_da}' in df_model.columns:
    df_model = df_model.drop(columns = [f'DA_{ref_da}'])


# ----------------------
# 3. Collinearity Check (Optional: Verify model rationality)
# ----------------------
def check_vif(df):
    """Check Variance Inflation Factor (VIF < 10 = no severe collinearity)"""
    X_vif = sm.add_constant(df)
    vif_data = pd.DataFrame()
    vif_data['Variable'] = X_vif.columns
    vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    return vif_data


# Check collinearity
da_columns = [col for col in df_model.columns if col.startswith('DA_')]
vif_result = check_vif(df_model[da_columns])
print("\n📊 Collinearity Test Results (VIF < 10 = Normal):")
print(vif_result.round(2))

# ----------------------
# 4. Fit Simplified Logistic Regression (No interaction terms, first verify DA main effect)
# ----------------------
# Independent variables: DA type one-hot encoding
X = df_model[da_columns].astype(float)
X = sm.add_constant(X)  # Add constant term
y = df_model['DV_Compliance'].astype(float)

# Fit model (add fault tolerance parameters)
try:
    logit_model = sm.Logit(y, X)
    # Increase maxiter + loose convergence criteria to avoid non-convergence
    result = logit_model.fit(maxiter = 100, disp = 0, tol = 1e-4)
except:
    # If Logit fitting fails, switch to more stable GLM model (logistic link)
    print("⚠️ Logit model fitting failed, switching to GLM (logistic link)")
    logit_model = sm.GLM(y, X, family = sm.families.Binomial())
    result = logit_model.fit(maxiter = 100, disp = 0)

# ----------------------
# 5. Result Output (Adapt to simplified model)
# ----------------------
print("\n=== Logistic Regression Results (DA Main Effect on EML-PR) ===")
if hasattr(result, 'prsquared'):
    print(f"Model Goodness of Fit (Pseudo R²): {result.prsquared:.4f}")
else:
    print("Model Goodness of Fit: Pseudo R² not available for GLM")
print(f"Overall Model Significance (p-value): {result.pvalues[0]:.4f}")
print(f"Valid Sample Size: {len(df_model)}")

# Organize results
reg_results = pd.DataFrame({
    'DA_Type': [col.replace('DA_', '') if col != 'const' else 'Intercept' for col in result.params.index],
    'Coefficient': result.params.values.round(4),
    'Std.Error': result.bse.values.round(4) if hasattr(result, 'bse') else [np.nan] * len(result.params),
    'p-value': result.pvalues.values.round(4),
    'OR (Odds Ratio)': np.exp(result.params.values).round(4),
})


# Mark significance
def mark_significance(p):
    if pd.isna(p):
        return ''
    elif p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    elif p < 0.1:
        return '.'
    else:
        return ''


reg_results['Significance'] = reg_results['p-value'].apply(mark_significance)
# Sort by p-value to highlight significant DA types
reg_results = reg_results.sort_values('p-value')

print("\n2. Detailed Regression Results (Reference Group: DA_{ref_da}):")
print(reg_results)

# ----------------------
# 6. Extract Core Conclusions (Adapt to simplified model)
# ----------------------
print("\n=== Core Conclusions ===")
# Filter DA type rows (exclude intercept)
da_results = reg_results[reg_results['DA_Type'] != 'Intercept']
for idx, row in da_results.iterrows():
    da_type = row['DA_Type']
    p_val = row['p-value']
    or_val = row['OR (Odds Ratio)']
    sig = row['Significance']

    if p_val < 0.05:
        print(f"✅ DA Type {da_type}: Significant effect on EML persuasive outcomes (p={p_val:.4f}{sig})")
        if or_val > 1:
            print(
                f"   → Compared to {ref_da}, {da_type} DA type increases the probability of EML-triggered compliance by {or_val:.2f}x (OR={or_val:.4f})")
        else:
            print(
                f"   → Compared to {ref_da}, {da_type} DA type reduces the probability of EML-triggered compliance to {or_val:.2f}x (OR={or_val:.4f})")
    else:
        print(f"❌ DA Type {da_type}: No significant effect on EML persuasive outcomes (p={p_val:.4f})")

# ----------------------
# 7. Save Results
# ----------------------
with pd.ExcelWriter('exp3_logistic_regression_results_final.xlsx', engine = 'openpyxl') as writer:
    reg_results.to_excel(writer, sheet_name = 'Regression_Results', index = False)
    df_model.to_excel(writer, sheet_name = 'Model_Data', index = False)
    vif_result.to_excel(writer, sheet_name = 'VIF_Check', index = False)

print("\n📊 Final results saved to: exp3_logistic_regression_results_final.xlsx")
print(f"\n=== Model Interpretation Notes ===")
print(f"1. Reference DA Type: {ref_da} (largest sample size)")
print(f"2. OR > 1: DA type enhances EML persuasive effect; OR < 1: weakens")
print(f"3. p < 0.05: DA type has significant moderating effect on EML-PR relationship")