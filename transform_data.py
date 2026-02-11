import pandas as pd
import numpy as np
from pathlib import Path

# ============================================
# 1. Load data
# ============================================
file_path = Path("./data/300_dialog.xlsx")
df = pd.read_excel(file_path)

print(f"Original data shape: {df.shape}")
print(f"Unique values in 'ee_label_1':\n{df['ee_label_1'].unique()}")
print(f"\nValue distribution of 'ee_label_1':\n{df['ee_label_1'].value_counts()}")

# ============================================
# 2. Define mapping rules (based on your label statistics)
# ============================================
# Compliance: acceptance, agreement, donation confirmation
# Resistance: rejection, opposition, negative reactions
# Neutral: neutral inquiries, ordinary responses, polite conversation

mapping_rules = {
    # ---------- Compliance ----------
    "agree-donation": "Compliance",
    "confirm-donation": "Compliance",
    "positive-to-inquiry": "Compliance",
    "positive-reaction-to-donation": "Compliance",

    # ---------- Resistance ----------
    "disagree-donation": "Resistance",
    "disagree-donation-more": "Resistance",
    "negative-reaction-to-donation": "Resistance",
    "negative-to-inquiry": "Resistance",

    # ---------- Neutral / Other ----------
    "acknowledgement": "Neutral",
    "greeting": "Neutral",
    "neutral-reaction-to-donation": "Neutral",
    "neutral-to-inquiry": "Neutral",
    "ask-donation-procedure": "Neutral",
    "ask-org-info": "Neutral",
    "ask-persuader-donation-intention": "Neutral",
    "closing": "Neutral",
    "other": "Neutral",
    "task-related-inquiry": "Neutral",
    "thank": "Neutral",
    "you-are-welcome": "Neutral",
    "provide-donation-amount": "Neutral",
    "personal-related-inquiry": "Neutral",
    "off-task": "Neutral",
    "": "Neutral",
    None: "Neutral",
    np.nan: "Neutral"
}


# ============================================
# 3. Apply mapping with error handling
# ============================================
def map_to_persuasion_result(label):
    """Map ee_label_1 to persuasion result"""
    if pd.isna(label) or label == "":
        return "Neutral"

    result = mapping_rules.get(label, None)

    if result is None:
        print(f"Warning: Undefined label '{label}' - defaulting to Neutral")
        return "Neutral"

    return result


df['persuasion_result'] = df['ee_label_1'].apply(map_to_persuasion_result)

# ============================================
# 4. Validate results
# ============================================
print("\n" + "=" * 50)
print("Mapping Results Summary")
print("=" * 50)

print("\n[persuasion_result Distribution]")
persuasion_counts = df['persuasion_result'].value_counts()
print(persuasion_counts)

print("\n[persuasion_result Percentages]")
persuasion_percentages = df['persuasion_result'].value_counts(normalize = True) * 100
for label, pct in persuasion_percentages.items():
    print(f"  {label}: {pct:.1f}%")

print("\n[ee_label_1 to persuasion_result Mapping Details]")
mapping_check = df.groupby(['ee_label_1', 'persuasion_result']).size().reset_index(name = 'count')
print(mapping_check)

# ============================================
# 5. Save results
# ============================================
output_path = file_path.parent / "300_dialog_with_result.xlsx"
df.to_excel(output_path, index = False)
print(f"\n✅ Saved: {output_path}")

csv_path = file_path.parent / "300_dialog_with_result.csv"
df.to_csv(csv_path, index = False, encoding = 'utf-8-sig')
print(f"✅ Saved: {csv_path}")

# ============================================
# 6. Data preview
# ============================================
print("\n" + "=" * 50)
print("Data Preview (Key Columns)")
print("=" * 50)
preview_cols = ['B2', 'Turn', 'Message Content', 'ee_label_1', 'persuasion_result', 'pos', 'neu', 'neg']
available_cols = [col for col in preview_cols if col in df.columns]
print(df[available_cols].head(15))

# ============================================
# 7. Mapping coverage analysis
# ============================================
print("\n" + "=" * 50)
print("Mapping Coverage Analysis")
print("=" * 50)

all_labels = set(df['ee_label_1'].unique())
mapped_labels = set()
for k in mapping_rules.keys():
    if isinstance(k, str) and k and k not in ["", None]:
        mapped_labels.add(k)
    if k is None or (isinstance(k, float) and np.isnan(k)):
        continue

defined_labels = mapped_labels.intersection(all_labels)
undefined_labels = all_labels - mapped_labels

print(f"Total unique ee_label_1 values in data: {len(all_labels)}")
print(f"Labels defined in mapping rules: {len(defined_labels)}")
print(f"Labels NOT defined in mapping rules: {len(undefined_labels)}")

if undefined_labels:
    print(f"\nUndefined labels:")
    for label in sorted(undefined_labels):
        if pd.isna(label) or label == "":
            continue
        count = df[df['ee_label_1'] == label].shape[0]
        print(f"  - '{label}': {count} occurrences")

# ============================================
# 8. Class distribution analysis
# ============================================
print("\n" + "=" * 50)
print("Class Distribution Analysis")
print("=" * 50)

class_counts = df['persuasion_result'].value_counts()
class_percentages = df['persuasion_result'].value_counts(normalize = True) * 100

distribution_df = pd.DataFrame({
    'Count': class_counts,
    'Percentage': class_percentages
})
print("\n[Class Distribution Table]")
print(distribution_df)

print("\n[Class Balance Check]")
min_class = class_counts.min()
max_class = class_counts.max()
imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')

print(f"Largest class size: {max_class} ({class_counts.index[0]})")
print(f"Smallest class size: {min_class} ({class_counts.index[-1]})")
print(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}:1")

if imbalance_ratio > 3:
    print("Warning: Significant class imbalance detected.")
    print("Consider using:")
    print("  - Weighted loss function (class weights)")
    print("  - Oversampling minority classes")
    print("  - Focal loss instead of standard cross-entropy")
else:
    print("Class distribution is relatively balanced.")

# ============================================
# 9. Export mapping dictionary for documentation
# ============================================
print("\n" + "=" * 50)
print("Exporting Mapping Dictionary")
print("=" * 50)

mapping_docs = []
for label, result in mapping_rules.items():
    if isinstance(label, str) and label and label not in ["", None]:
        mapping_docs.append({
            'ee_label_1': label,
            'persuasion_result': result
        })
    elif label is None or (isinstance(label, float) and np.isnan(label)):
        continue

mapping_df = pd.DataFrame(mapping_docs)
mapping_df = mapping_df.sort_values(['persuasion_result', 'ee_label_1'])

mapping_path = file_path.parent / "label_mapping_documentation.csv"
mapping_df.to_csv(mapping_path, index = False, encoding = 'utf-8-sig')
print(f"Mapping documentation saved: {mapping_path}")

print("\n[Mapping Summary by Category]")
for result in ['Compliance', 'Resistance', 'Neutral']:
    labels = mapping_df[mapping_df['persuasion_result'] == result]['ee_label_1'].tolist()
    print(f"\n{result} ({len(labels)} labels):")
    label_str = ', '.join(labels[:5])
    if len(labels) > 5:
        label_str += '...'
    print(f"  {label_str}")

# ============================================
# 10. Summary statistics
# ============================================
print("\n" + "=" * 50)
print("Dataset Summary")
print("=" * 50)

total_messages = len(df)
total_compliance = class_counts.get('Compliance', 0)
total_resistance = class_counts.get('Resistance', 0)
total_neutral = class_counts.get('Neutral', 0)

print(f"Total messages: {total_messages}")
print(f"Total Compliance examples: {total_compliance} ({total_compliance / total_messages * 100:.1f}%)")
print(f"Total Resistance examples: {total_resistance} ({total_resistance / total_messages * 100:.1f}%)")
print(f"Total Neutral examples: {total_neutral} ({total_neutral / total_messages * 100:.1f}%)")

print("\n[Training Feasibility Check]")
min_samples_for_training = 100
if total_compliance < min_samples_for_training or total_resistance < min_samples_for_training:
    print(f"Warning: Minority class has less than {min_samples_for_training} samples.")
    print("Consider data augmentation or few-shot learning approaches.")
else:
    print("Sufficient samples for training.")

print("\nData processing completed successfully!")