import pandas as pd
import numpy as np

# Read the Excel file
df = pd.read_excel('Persusasion_For_Good_manual_label.xlsx')


# Function to get majority vote
def get_majority(row, col_prefix):
    values = [row[f'{col_prefix}_davide'],
              row[f'{col_prefix}_diletta'],
              row[f'{col_prefix}_qida']]

    # Filter out null values if needed
    valid_values = [v for v in values if pd.notna(v)]

    if len(valid_values) == 0:
        return None

    # If all three values are different, return None
    if len(set(valid_values)) == 3:
        return None

    # Otherwise return the most frequent value
    return max(set(valid_values), key = valid_values.count)


# Apply the function to create new columns
df['manipulation_majority'] = df.apply(lambda row: get_majority(row, 'manipulation'), axis = 1)
df['persuasion_result_majority'] = df.apply(lambda row: get_majority(row, 'persuasion_result'), axis = 1)

# Save to a new Excel file
df.to_excel('Persusasion_For_Good_manual_label_with_majority.xlsx', index = False)

print("Processing complete!")
print(f"Total rows: {len(df)}")
print(f"manipulation_majority non-null count: {df['manipulation_majority'].notna().sum()}")
print(f"persuasion_result_majority non-null count: {df['persuasion_result_majority'].notna().sum()}")

# Preview first few rows
print("\nPreview (first 5 rows):")
print(df[['index', 'Dialogue_ID', 'Turn', 'manipulation_davide', 'manipulation_diletta',
          'manipulation_qida', 'manipulation_majority',
          'persuasion_result_davide', 'persuasion_result_diletta',
          'persuasion_result_qida', 'persuasion_result_majority']].head())