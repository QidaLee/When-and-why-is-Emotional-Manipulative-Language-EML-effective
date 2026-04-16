import pandas as pd
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np

# ========== STEP 1 ==========
print("=" * 50)
print("STEP 1: Loading EML results (using eml_label directly)")
print("=" * 50)

# 直接读取模型输出的 eml_label
df_eml = pd.read_excel('../output_data/all_turns_data_with_eml_label.xlsx')
print(f"✓ Loaded {len(df_eml)} rows from EML output")

# ========== STEP 2 ==========
print("\n" + "=" * 50)
print("STEP 2: Merging eml_label with manual labeled dataset")
print("=" * 50)

df_persuasion = pd.read_excel('Persusasion_For_Good_manual_label_with_majority.xlsx')

# 只保留需要的列，使用 eml_label 而不是 first_num
df_eml_merged = df_eml[['Dialogue_ID', 'Role', 'Turn', 'Sentence', 'eml_label']].copy()

# Initialize new column
df_persuasion['eml_label'] = None

# Step 2a: Match by Sentence
print("\nMatching by Sentence...")
sentence_match_mask = df_persuasion['Sentence'].isin(df_eml_merged['Sentence'])
sentence_matched_indices = df_persuasion[sentence_match_mask].index

for idx in sentence_matched_indices:
    sentence = df_persuasion.loc[idx, 'Sentence']
    match = df_eml_merged[df_eml_merged['Sentence'] == sentence].iloc[0]
    df_persuasion.loc[idx, 'eml_label'] = match['eml_label']

print(f"  Matched by Sentence: {len(sentence_matched_indices)} rows")

# Step 2b: Match by Dialogue_ID, Role, Turn
unmatched_mask = df_persuasion['eml_label'].isna()
print(f"\nMatching remaining {unmatched_mask.sum()} rows by (Dialogue_ID, Role, Turn)...")

for idx in df_persuasion[unmatched_mask].index:
    dialogue_id = df_persuasion.loc[idx, 'Dialogue_ID']
    role = df_persuasion.loc[idx, 'Role']
    turn = df_persuasion.loc[idx, 'Turn']

    match = df_eml_merged[
        (df_eml_merged['Dialogue_ID'] == dialogue_id) &
        (df_eml_merged['Role'] == role) &
        (df_eml_merged['Turn'] == turn)
    ]

    if len(match) > 0:
        df_persuasion.loc[idx, 'eml_label'] = match.iloc[0]['eml_label']

# Step 2c: Final match by Dialogue_ID + Role first 2 chars + Turn
still_unmatched_mask = df_persuasion['eml_label'].isna()
print(f"\nFinal matching for {still_unmatched_mask.sum()} remaining rows...")

for idx in df_persuasion[still_unmatched_mask].index:
    dialogue_id = df_persuasion.loc[idx, 'Dialogue_ID']
    role = str(df_persuasion.loc[idx, 'Role'])
    role_first2 = role[:2] if len(role) >= 2 else role
    turn = df_persuasion.loc[idx, 'Turn']

    match = df_eml_merged[
        (df_eml_merged['Dialogue_ID'] == dialogue_id) &
        (df_eml_merged['Role'].astype(str).str[:2] == role_first2) &
        (df_eml_merged['Turn'] == turn)
    ]

    if len(match) > 0:
        df_persuasion.loc[idx, 'eml_label'] = match.iloc[0]['eml_label']

# Check agreement
def check_agreement(row):
    val1 = row.get('manipulation_majority')
    val2 = row.get('eml_label')
    if pd.isna(val1) or pd.isna(val2):
        return None
    return 1 if val1 == val2 else 0

df_persuasion['same_eml_label'] = df_persuasion.apply(check_agreement, axis=1)

# Save updated file
df_persuasion.to_excel('Persusasion_For_Good_manual_label_with_majority.xlsx', index=False)

# ========== STEP 3: EVALUATION ==========
print("\n" + "=" * 50)
print("STEP 3: Classification Performance Evaluation")
print("=" * 50)

# Filter valid rows
analysis_df = df_persuasion.dropna(subset=['manipulation_majority', 'eml_label']).copy()
print(f"\nValid samples for evaluation: {len(analysis_df)}")

# True = manual label
# Pred = model label (eml_label)
y_true = analysis_df['manipulation_majority'].astype(int)
y_pred = analysis_df['eml_label'].astype(int)

# Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)

print("\n" + "-" * 50)
print("CLASSIFICATION METRICS")
print("-" * 50)
print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1-Score:  {f1:.4f} ({f1 * 100:.2f}%)")

# Confusion Matrix
print("\n" + "-" * 50)
print("CONFUSION MATRIX")
print("-" * 50)
cm = confusion_matrix(y_true, y_pred)
print(f"                 Predicted")
print(f"                 0      1")
print(f"Actual    0    {cm[0, 0]:4d}   {cm[0, 1]:4d}")
print(f"          1    {cm[1, 0]:4d}   {cm[1, 1]:4d}")

# Report
print("\n" + "-" * 50)
print("CLASSIFICATION REPORT")
print("-" * 50)
print(classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1'], zero_division=0))

# Additional stats
print("\n" + "-" * 50)
print("ADDITIONAL STATISTICS")
print("-" * 50)

true_counts = y_true.value_counts().sort_index()
pred_counts = y_pred.value_counts().sort_index()

print("\nTrue Label Distribution (manipulation_majority):")
print(f"  Class 0: {true_counts.get(0, 0)} ({true_counts.get(0, 0)/len(y_true)*100:.2f}%)")
print(f"  Class 1: {true_counts.get(1, 0)} ({true_counts.get(1, 0)/len(y_true)*100:.2f}%)")

print("\nPredicted Label Distribution (eml_label):")
print(f"  Class 0: {pred_counts.get(0, 0)} ({pred_counts.get(0, 0)/len(y_pred)*100:.2f}%)")
print(f"  Class 1: {pred_counts.get(1, 0)} ({pred_counts.get(1, 0)/len(y_pred)*100:.2f}%)")

# Error analysis
fp = ((y_pred == 1) & (y_true == 0)).sum()
fn = ((y_pred == 0) & (y_true == 1)).sum()
print(f"\nFalse Positives: {fp}")
print(f"False Negatives: {fn}")

# Baseline
majority = y_true.mode()[0]
baseline = (y_true == majority).mean()
print(f"\nBaseline Accuracy (majority class): {baseline*100:.2f}%")

print("\n" + "=" * 50)
print("FINAL SUMMARY")
print("=" * 50)
print(f"✓ Using eml_label directly (no first-number extraction)")
print(f"✓ Total evaluated samples: {len(analysis_df)}")
print(f"✓ Accuracy: {accuracy*100:.2f}%")
print(f"✓ F1 Score: {f1*100:.2f}%")
print("\n✓ Analysis completed successfully!")