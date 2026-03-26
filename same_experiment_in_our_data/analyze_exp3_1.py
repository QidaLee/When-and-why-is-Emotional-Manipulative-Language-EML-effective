import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# ----------------------
# Global Configuration
# ----------------------
TIME_GAP_THRESHOLD = 3  # Temporal window threshold (adjustable)
VALID_PR_LABELS = ['Compliance', 'Resistance']  # Valid PR labels
OUTPUT_FILE = 'exp3_eml_pr_matched_data.csv'


# ----------------------
# Step 1: Data Loading and Preprocessing
# ----------------------
def load_and_preprocess_data(file_path):
    """Load dataset and preprocess core fields"""
    df = pd.read_excel(file_path)

    # Convert core fields to numeric type
    df['Turn'] = pd.to_numeric(df['Turn'], errors='coerce')
    df['EML_label'] = pd.to_numeric(df['EML_label'], errors='coerce')  # Note uppercase E

    # Filter invalid values
    df = df.dropna(subset=['Dialogue_ID', 'Turn', 'Role'])
    df['PR_label'] = df['PR_label'].fillna('NaN')

    # Mark valid PR records
    df['is_valid_pr'] = df['PR_label'].isin(VALID_PR_LABELS)

    return df


# Load dataset (replace with your actual file path)
df = load_and_preprocess_data('output_data/all_turns_data_with_EML_DA_PR.xlsx')


# ----------------------
# Step 2: Filter Valid Dialogues (≥1 EML + ≥1 Valid PR)
# ----------------------
def filter_valid_dialogues(df):
    """Filter dialogues that meet criteria: at least 1 EML and 1 valid PR"""
    dialogue_stats = df.groupby('Dialogue_ID').agg({
        'EML_label': lambda x: (x == 1).any(),  # At least 1 EML utterance
        'is_valid_pr': lambda x: x.any()        # At least 1 valid PR utterance
    }).reset_index()

    valid_dialogues = dialogue_stats[
        (dialogue_stats['EML_label'] == True) &
        (dialogue_stats['is_valid_pr'] == True)
    ]['Dialogue_ID'].tolist()

    df_valid = df[df['Dialogue_ID'].isin(valid_dialogues)].copy()
    return df_valid, valid_dialogues


df_valid, valid_dialogues = filter_valid_dialogues(df)
print(f"Number of valid dialogues (≥1 EML + ≥1 valid PR): {len(valid_dialogues)}")


# ----------------------
# Step 3: Match EML Utterances to PR Outcomes (Core Logic)
# ----------------------
def match_eml_to_pr(df_valid):
    """
    Match each EML utterance to:
    1. EML_result: Closest valid PR after EML (different speaker, turn gap ≤3)
    2. PR_Status_before: Closest valid PR before EML (different speaker, turn gap ≤3)
    """
    # Extract all EML utterances (EML_label=1)
    eml_utterances = df_valid[df_valid['EML_label'] == 1].copy()
    eml_utterances = eml_utterances[
        ['Dialogue_ID', 'Turn', 'Role', 'DA_label']  # DA type field is DA_label
    ].rename(
        columns={
            'Turn': 'eml_turn',
            'Role': 'eml_role',
            'DA_label': 'eml_da_type'
        }
    ).reset_index(drop=True)

    # Initialize result columns
    eml_utterances['EML_result'] = 'NaN'
    eml_utterances['pr_turn_after'] = np.nan
    eml_utterances['PR_Status_before'] = 'NaN'
    eml_utterances['pr_turn_before'] = np.nan

    # Iterate through each EML utterance
    for idx, eml_row in eml_utterances.iterrows():
        dialogue_id = eml_row['Dialogue_ID']
        eml_turn = eml_row['eml_turn']
        eml_role = eml_row['eml_role']

        # All valid PRs from different speakers in the same dialogue
        dialogue_pr = df_valid[
            (df_valid['Dialogue_ID'] == dialogue_id) &
            (df_valid['is_valid_pr'] == True) &
            (df_valid['Role'] != eml_role)  # Exclude self-referential PR
        ].copy()

        if len(dialogue_pr) == 0:
            continue

        # ----------------------
        # Match PR after EML (EML_result)
        # ----------------------
        pr_after = dialogue_pr[
            (dialogue_pr['Turn'] > eml_turn) &
            (dialogue_pr['Turn'] - eml_turn <= TIME_GAP_THRESHOLD)
        ].copy()

        if len(pr_after) > 0:
            pr_after['turn_gap'] = pr_after['Turn'] - eml_turn
            pr_after = pr_after.sort_values('turn_gap').iloc[0]  # Select closest PR
            eml_utterances.loc[idx, 'EML_result'] = pr_after['PR_label']
            eml_utterances.loc[idx, 'pr_turn_after'] = pr_after['Turn']

        # ----------------------
        # Match PR before EML (PR_Status_before)
        # ----------------------
        pr_before = dialogue_pr[
            (dialogue_pr['Turn'] < eml_turn) &
            (eml_turn - dialogue_pr['Turn'] <= TIME_GAP_THRESHOLD)
        ].copy()

        if len(pr_before) > 0:
            pr_before['turn_gap'] = eml_turn - pr_before['Turn']
            pr_before = pr_before.sort_values('turn_gap').iloc[0]  # Select closest PR
            eml_utterances.loc[idx, 'PR_Status_before'] = pr_before['PR_label']
            eml_utterances.loc[idx, 'pr_turn_before'] = pr_before['Turn']

    # Keep only EML utterances with valid PR outcomes
    eml_matched = eml_utterances[eml_utterances['EML_result'] != 'NaN'].copy()
    return eml_matched


# Execute matching
eml_matched_data = match_eml_to_pr(df_valid)

# ----------------------
# Step 4: Output Results
# ----------------------
print("\n=== Experiment 3 Step 1 Results Summary ===")
total_eml = len(df_valid[df_valid['EML_label'] == 1])
matched_eml = len(eml_matched_data)
print(f"Total EML utterances: {total_eml}")
print(f"EML utterances matched with valid subsequent PR: {matched_eml}")
print(f"Matching rate: {matched_eml / total_eml:.2%}")

# Save matched data
eml_matched_data.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
print(f"\nMatched data saved to: {OUTPUT_FILE}")

# Preview first 5 records
print("\nFirst 5 matched results:")
print(eml_matched_data[
          ['Dialogue_ID', 'eml_turn', 'eml_role', 'eml_da_type',
           'PR_Status_before', 'EML_result']
      ].head())