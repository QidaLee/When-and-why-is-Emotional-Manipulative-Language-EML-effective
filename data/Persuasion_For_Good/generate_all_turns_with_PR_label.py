import pandas as pd
import numpy as np

# ====================== Configuration ======================
# Data file paths
INPUT_FILE = "all_turns_data.xlsx"
OUTPUT_FILE = "all_turns_data_with_PR_label.xlsx"

# Label mapping rules
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


# ====================== Core Processing Functions ======================
def parse_label_list(label_str):
    """
    Parse the ee_label_1 column string list and convert to actual label list
    Handles formats: [] / ['label1'] / ['label1', 'label2']
    """
    # Handle null values
    if pd.isna(label_str) or label_str == "[]":
        return []

    # Convert to string and clean
    label_str = str(label_str).strip().strip('[]').replace("'", "").replace('"', '')

    # Split labels
    if label_str == "":
        return []
    labels = [label.strip() for label in label_str.split(',') if label.strip()]

    return labels


def map_labels(raw_labels, mapping_rules):
    """Map raw labels to classification labels (Compliance/Resistance/Neutral)"""
    mapped_labels = []

    for label in raw_labels:
        # Look up mapping rule, default to Neutral
        mapped_label = mapping_rules.get(label.strip(), "Neutral")
        mapped_labels.append(mapped_label)

    return mapped_labels


def determine_final_label(mapped_labels):
    """
    Determine final label based on mapped label list:
    - Only one category: return that category
    - Compliance + Neutral: Compliance
    - Resistance + Neutral: Resistance
    - Compliance + Resistance: Conflict
    """
    # Deduplicate and get unique label set
    unique_labels = list(set(mapped_labels))

    # Handle empty list
    if not unique_labels:
        return "Neutral"

    # Only one label category
    if len(unique_labels) == 1:
        return unique_labels[0]

    # Priority decision for multiple label categories
    has_compliance = "Compliance" in unique_labels
    has_resistance = "Resistance" in unique_labels
    has_neutral = "Neutral" in unique_labels

    # Compliance + Resistance (regardless of Neutral) → Conflict
    if has_compliance and has_resistance:
        return "Conflict"

    # Compliance + Neutral → Compliance
    elif has_compliance and has_neutral:
        return "Compliance"

    # Resistance + Neutral → Resistance
    elif has_resistance and has_neutral:
        return "Resistance"

    # Default to Neutral for other cases
    else:
        return "Neutral"


# ====================== 新增：冲突时取第一个标签 ======================
def get_persuasion_result(mapped_labels):
    """
    最终说服结果规则：
    - 如果是 Conflict → 返回 mapped_labels[0]（第一个标签）
    - 否则返回正常的 final label
    """
    final_label = determine_final_label(mapped_labels)

    if final_label == "Conflict" and len(mapped_labels) > 0:
        return mapped_labels[0]
    return final_label


def process_ee_label_column(df, mapping_rules):
    """Complete processing pipeline for ee_label_1 column"""
    print("Processing ee_label_1 column...")

    # 1. Parse raw label list
    df["ee_label_1_parsed"] = df["ee_label_1"].apply(parse_label_list)

    # 2. Map labels
    df["ee_label_1_mapped"] = df["ee_label_1_parsed"].apply(lambda x: map_labels(x, mapping_rules))

    # 3. Determine final label
    df["ee_label_final"] = df["ee_label_1_mapped"].apply(determine_final_label)

    # ====================== 新增：生成 persuasion_result ======================
    df["persuasion_result"] = df["ee_label_1_mapped"].apply(get_persuasion_result)

    print("Label processing completed!")
    return df


# ====================== Main Program ======================
if __name__ == "__main__":
    # 1. Load data
    print(f"Loading data: {INPUT_FILE}")
    df = pd.read_excel(INPUT_FILE)
    print(f"Data loaded successfully, total {len(df)} rows")

    # 2. Check if required column exists
    if "ee_label_1" not in df.columns:
        raise ValueError("Data missing ee_label_1 column!")

    # 3. Process label column
    df_processed = process_ee_label_column(df, mapping_rules)

    # 4. Display final label distribution
    print("\nFinal label distribution (ee_label_final):")
    label_distribution = df_processed["ee_label_final"].value_counts()
    for label, count in label_distribution.items():
        percentage = count / len(df_processed) * 100
        print(f"  {label}: {count} rows ({percentage:.2f}%)")

    # 新增：显示说服结果分布（无Conflict）
    print("\nPersuasion Result distribution (Conflict → first label):")
    pr_distribution = df_processed["persuasion_result"].value_counts()
    for label, count in pr_distribution.items():
        percentage = count / len(df_processed) * 100
        print(f"  {label}: {count} rows ({percentage:.2f}%)")

    # 5. Save processed data
    df_processed.to_excel(OUTPUT_FILE, index = False)
    print(f"\nProcessed data saved to: {OUTPUT_FILE}")

    # 6. Display processing examples (first 10 rows)
    print("\nProcessing examples (first 10 rows):")
    sample_cols = ["Sentence", "ee_label_1_mapped", "ee_label_final", "persuasion_result"]
    print(df_processed[sample_cols].head(10).to_string(index = False))