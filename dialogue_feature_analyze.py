import pandas as pd
import matplotlib.pyplot as plt

# 1. Read data file (use read_excel for .xlsx, read_csv for .csv)
# Primary option for Excel files (.xlsx)
df = pd.read_excel("output_data/all_turns_data_with_EML_DA_PR.xlsx")

# Alternative for CSV files (uncomment below if using .csv)
# df = pd.read_csv("output_data/all_turns_data_with_EML_DA_PR.csv", encoding='gbk')  # Try 'gb2312' or 'utf-8-sig' if encoding error occurs

# 2. Define target labels
target_pr_labels = ['Compliance', 'Resistance']  # PR labels to count
eml_positive_label = 1  # Value indicating EML presence (adjust if your EML label uses different value)


# 3. Function to calculate metrics per dialogue
def calculate_dialogue_metrics(group):
    """
    Calculate key metrics for a single dialogue:
    - Total Compliance/Resistance occurrences
    - Total EML occurrences
    """
    # Count PR occurrences (Compliance/Resistance)
    pr_count = len(group[group['PR_label'].isin(target_pr_labels)])

    # Count EML occurrences (adjust column name if needed: EML_label/eml_label)
    eml_count = len(group[group['EML_label'] == eml_positive_label])

    return pd.Series({
        'total_pr_count': pr_count,
        'total_eml_count': eml_count
    })


# Group by Dialogue_ID and calculate metrics
dialogue_metrics = df.groupby('Dialogue_ID').apply(calculate_dialogue_metrics).reset_index()

# 4. Core statistical results
total_dialogues = len(dialogue_metrics)

# PR statistics
multi_pr_dialogues = len(dialogue_metrics[dialogue_metrics['total_pr_count'] >= 2])
multi_pr_ratio = multi_pr_dialogues / total_dialogues

# EML statistics
eml_present_dialogues = len(dialogue_metrics[dialogue_metrics['total_eml_count'] > 0])
eml_present_ratio = eml_present_dialogues / total_dialogues
total_eml_occurrences = dialogue_metrics['total_eml_count'].sum()  # Global EML count across all dialogues

# Cross-analysis: Dialogues with both PR (≥2) and EML
pr_2plus_eml_dialogues = len(dialogue_metrics[
                                 (dialogue_metrics['total_pr_count'] >= 2) &
                                 (dialogue_metrics['total_eml_count'] > 0)
                                 ])
pr_2plus_eml_ratio = pr_2plus_eml_dialogues / total_dialogues if total_dialogues > 0 else 0

# 5. Print comprehensive statistics
print("=== Dialogue-level Comprehensive Statistics ===")
print(f"Total number of dialogues: {total_dialogues}")

print("\n--- Compliance/Resistance (PR) Statistics ---")
print(f"Dialogues with ≥2 PR occurrences: {multi_pr_dialogues} ({multi_pr_ratio:.2%})")
print(f"Total PR occurrences across all dialogues: {dialogue_metrics['total_pr_count'].sum()}")

print("\n--- EML Statistics ---")
print(f"Dialogues with at least 1 EML occurrence: {eml_present_dialogues} ({eml_present_ratio:.2%})")
print(f"Total EML occurrences across all dialogues: {total_eml_occurrences}")

print("\n--- Cross-analysis: PR (≥2) + EML ---")
print(f"Dialogues with ≥2 PR occurrences AND at least 1 EML: {pr_2plus_eml_dialogues} ({pr_2plus_eml_ratio:.2%})")

# 6. Detailed distribution tables
print("\n=== PR Occurrence Distribution ===")
pr_distribution = dialogue_metrics['total_pr_count'].value_counts().sort_index()
for count, num_dialogues in pr_distribution.items():
    print(f"Dialogues with {count} PR occurrence(s): {num_dialogues}")

print("\n=== EML Occurrence Distribution ===")
eml_distribution = dialogue_metrics['total_eml_count'].value_counts().sort_index()
for count, num_dialogues in eml_distribution.items():
    print(f"Dialogues with {count} EML occurrence(s): {num_dialogues}")

# 7. Visualization (separate plots saved as individual files)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --------------------------
# Plot 1: PR Distribution (saved as separate file)
# --------------------------
plt.figure(figsize = (8, 5))
pr_distribution.plot(kind = 'bar', color = 'skyblue')
plt.title('Distribution of PR (Compliance/Resistance) Occurrences per Dialogue')
plt.xlabel('Number of PR Occurrences')
plt.ylabel('Number of Dialogues')
plt.xticks(rotation = 0)
plt.grid(axis = 'y', linestyle = '--', alpha = 0.7)
plt.tight_layout()
plt.savefig('pr_occurrence_distribution.png')  # PR plot saved separately
plt.close()  # Close the plot to free memory

# --------------------------
# Plot 2: EML Distribution (saved as separate file)
# --------------------------
plt.figure(figsize = (8, 5))
eml_distribution.plot(kind = 'bar', color = 'lightcoral')
plt.title('Distribution of EML Occurrences per Dialogue')
plt.xlabel('Number of EML Occurrences')
plt.ylabel('Number of Dialogues')
plt.xticks(rotation = 0)
plt.grid(axis = 'y', linestyle = '--', alpha = 0.7)
plt.tight_layout()
plt.savefig('eml_occurrence_distribution.png')  # EML plot saved separately
plt.close()  # Close the plot to free memory

# Optional: Show plots (comment out if you only need saved files)
# plt.show()