import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Read data
df = pd.read_csv('./output_data/agreement_with_our_predictions.csv')

# View basic data information
print("="*50)
print("Basic Data Information")
print("="*50)
# print(f"Data shape: {df.shape} (rows, columns)")
# print(f"\nColumn names: {list(df.columns)}")
# print(f"\nData types:")
# print(df.dtypes)
# print(f"\nMissing value statistics:")
# missing_data = df.isnull().sum()
# missing_percent = (missing_data / len(df)) * 100
# missing_df = pd.DataFrame({
#     'Missing count': missing_data,
#     'Missing percentage(%)': missing_percent.round(2)
# })
# print(missing_df[missing_df['Missing count'] > 0])

# View first 5 rows of data
# print(f"\nFirst 5 rows preview:")
# print(df.head())

# View unique value counts for each column
# print(f"\nUnique value counts per column:")
# for col in df.columns:
#     print(f"{col}: {df[col].nunique()} unique values")

# Display detailed information for selected categorical columns
print(f"\nDetailed information for selected columns:")

# Specify which columns you want to analyze (modify this list as needed)
selected_columns = ['speaker', 'prediction_label_DA', 'prediction_label_PR']  # Replace with your actual column names

for col in selected_columns:
    if col in df.columns:  # Check if column exists
        if df[col].nunique() <= 20:  # Only show columns with limited categories
            print(f"\n{col} value distribution:")
            value_counts = df[col].value_counts()
            print(value_counts)
            print(f"Percentage:")
            print((value_counts / len(df) * 100).round(2))
        else:
            print(f"\n{col} has {df[col].nunique()} unique values (too many to display)")
    else:
        print(f"\nWarning: Column '{col}' not found in dataframe")


# Data preprocessing and reorganization
# 1. Confirm core variable definitions
print("="*60)
print("Core Variable Definition Confirmation")
print("="*60)

# Confirm manipulation variable (whether it's manipulative dialogue)
print(f"1. Manipulation variable (whether it's manipulative dialogue):")
print(f"   Values: {sorted(df['manipulation'].unique())}")
print(f"   Distribution:")
manip_count = df['manipulation'].value_counts().sort_index()
for val, count in manip_count.items():
    print(f"     {val}: {count} entries ({count/len(df)*100:.2f}%)")

# Confirm dialogue act types (inferred from prediction_id)
print(f"\n2. Dialogue act types (based on prediction_id):")
id_to_label = dict(zip(df['prediction_id'], df['prediction_label_PR']))
for pid in sorted(df['prediction_id'].unique()):
    label = id_to_label[pid]
    count = len(df[df['prediction_id'] == pid])
    print(f"   prediction_id {pid} = {label}: {count} entries ({count/len(df)*100:.2f}%)")

# Confirm emotional manipulation results (PR_labels_from_GPT)
print(f"\n3. Emotional manipulation results (PR_labels_from_GPT):")
pr_count = df['PR_labels_from_GPT'].value_counts()
for label, count in pr_count.items():
    print(f"   {label}: {count} entries ({count/len(df)*100:.2f}%)")

# 2. Create comprehensive analysis dataset
df_analysis = df.copy()

# Rename key columns for analysis
df_analysis = df_analysis.rename(columns={
    'manipulation': 'is_manipulation',  # Whether it's manipulative dialogue (0/1)
    'prediction_label_DA': 'dialogue_act',  # Dialogue act type
    'PR_labels_from_GPT': 'emotional_result'  # Emotional manipulation result
})

# Calculate annotation consistency among 4 annotators
# annotation_cols = ['manipulation_davide', 'manipulation_diletta',
#                   'manipulation_inga', 'manipulation_matias']
# df_analysis['annotation_consensus'] = df_analysis[annotation_cols].std(axis=1) == 0
#
# print(f"\n4. Annotation consistency:")
# consensus_count = df_analysis['annotation_consensus'].value_counts()
# for val, count in consensus_count.items():
#     status = "Fully consistent" if val else "With disagreements"
#     print(f"   {status}: {count} entries ({count/len(df_analysis)*100:.2f}%)")

# Display reorganized core data structure
print(f"\n5. Reorganized core data structure:")
core_cols = ['speaker', 'conversation_id', 'text', 'is_manipulation',
             'dialogue_act', 'emotional_result']
print(df_analysis[core_cols].head(3))

print(f"\nData preparation complete! Total {len(df_analysis)} dialogue entries available for subsequent analysis.")

# 第一部分：描述性统计分析
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('对话数据核心变量描述性统计', fontsize=16, fontweight='bold')

# 1. 操纵对话分布
ax1 = axes[0, 0]
manip_data = df_analysis['is_manipulation'].value_counts().sort_index()
labels_manip = ['非操纵对话 (0)', '操纵对话 (1)']
colors1 = ['#3498db', '#e74c3c']
wedges, texts, autotexts = ax1.pie(manip_data.values, labels=labels_manip, colors=colors1,
                                   autopct='%1.2f%%', startangle=90)
ax1.set_title('操纵对话vs非操纵对话分布\n(共2370条)', fontweight='bold', pad=20)

# 2. 对话行为类型分布
ax2 = axes[0, 1]
dialogue_data = df_analysis['dialogue_act'].value_counts()
colors2 = ['#2ecc71', '#f39c12', '#9b59b6']
bars = ax2.bar(dialogue_data.index, dialogue_data.values, color=colors2, alpha=0.8)
ax2.set_title('对话行为类型分布', fontweight='bold', pad=20)
ax2.set_ylabel('对话数量')
# 添加数值标签
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 20,
             f'{int(height)}\n({height/len(df_analysis)*100:.2f}%)',
             ha='center', va='bottom', fontweight='bold')

# 3. 情感操纵结果分布
ax3 = axes[1, 0]
emotion_data = df_analysis['emotional_result'].value_counts()
colors3 = ['#3498db', '#e74c3c', '#f39c12']
bars2 = ax3.bar(emotion_data.index, emotion_data.values, color=colors3, alpha=0.8)
ax3.set_title('情感操纵结果分布', fontweight='bold', pad=20)
ax3.set_ylabel('对话数量')
# 添加数值标签
for bar in bars2:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 20,
             f'{int(height)}\n({height/len(df_analysis)*100:.2f}%)',
             ha='center', va='bottom', fontweight='bold')


plt.tight_layout()
plt.savefig('./results/descriptive_analysis.png', dpi=300, bbox_inches='tight')
plt.close()





# Part 1: Descriptive Statistical Analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Descriptive Statistics of Core Dialogue Variables', fontsize=16, fontweight='bold')

# 1. Manipulative vs Non-manipulative Dialogue Distribution
ax1 = axes[0, 0]
manip_data = df_analysis['is_manipulation'].value_counts().sort_index()
labels_manip = ['Non-manipulative (0)', 'Manipulative (1)']
colors1 = ['#3498db', '#e74c3c']
wedges, texts, autotexts = ax1.pie(manip_data.values, labels=labels_manip, colors=colors1,
                                   autopct='%1.2f%%', startangle=90)
ax1.set_title('Distribution of Manipulative vs Non-manipulative Dialogues\n(Total: 2370 entries)', fontweight='bold', pad=20)

# 2. Dialogue Act Type Distribution
ax2 = axes[0, 1]
dialogue_data = df_analysis['dialogue_act'].value_counts()
colors2 = ['#2ecc71', '#f39c12', '#9b59b6']
bars = ax2.bar(dialogue_data.index, dialogue_data.values, color=colors2, alpha=0.8)
ax2.set_title('Dialogue Act Type Distribution', fontweight='bold', pad=20)
ax2.set_ylabel('Number of Dialogues')
# Add value labels
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 20,
             f'{int(height)}\n({height/len(df_analysis)*100:.2f}%)',
             ha='center', va='bottom', fontweight='bold')

# 3. Emotional Manipulation Result Distribution
ax3 = axes[1, 0]
emotion_data = df_analysis['emotional_result'].value_counts()
colors3 = ['#3498db', '#e74c3c', '#f39c12']
bars2 = ax3.bar(emotion_data.index, emotion_data.values, color=colors3, alpha=0.8)
ax3.set_title('Emotional Manipulation Result Distribution', fontweight='bold', pad=20)
ax3.set_ylabel('Number of Dialogues')
# Add value labels
for bar in bars2:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 20,
             f'{int(height)}\n({height/len(df_analysis)*100:.2f}%)',
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('./results/descriptive_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
