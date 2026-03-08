import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Set plot style and font
# plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
# plt.rcParams['title.weight'] = 'bold'

# Load data and recalculate validation metrics
df = pd.read_excel('output_data/100_sample_with_DA_EML.xlsx')

# EML Model Validation
eml_validation = df[['Sentence', 'manipulation_qida', 'eml_label']].dropna()
eml_accuracy = accuracy_score(eml_validation['manipulation_qida'], eml_validation['eml_label'])
eml_cm = confusion_matrix(eml_validation['manipulation_qida'], eml_validation['eml_label'])
eml_human_dist = eml_validation['manipulation_qida'].value_counts().sort_index()
eml_model_dist = eml_validation['eml_label'].value_counts().sort_index()

# Dialogue Act Model Validation
da_validation = df[['Sentence', 'dialogue_act_qida', 'DA_label_General']].dropna()
da_accuracy = accuracy_score(da_validation['dialogue_act_qida'], da_validation['DA_label_General'])
all_da_labels = sorted(list(set(da_validation['dialogue_act_qida'].unique()) |
                           set(da_validation['DA_label_General'].unique())))
da_cm = confusion_matrix(da_validation['dialogue_act_qida'], da_validation['DA_label_General'],
                        labels=all_da_labels)
human_dist = da_validation['dialogue_act_qida'].value_counts()
model_dist = da_validation['DA_label_General'].value_counts()

# 1. EML Confusion Matrix (单独图片)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(eml_cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Non-manipulative (0)', 'Manipulative (1)'],
            yticklabels=['Non-manipulative (0)', 'Manipulative (1)'],
            annot_kws={'size': 14, 'weight': 'bold'})
ax.set_xlabel('Model Prediction (eml_label)', fontsize=14, labelpad=10)
ax.set_ylabel('Human Annotation (manipulation_qida)', fontsize=14, labelpad=10)
ax.set_title(f'EML Model Confusion Matrix\nAccuracy: {eml_accuracy*100:.2f}%',
             fontsize=16, pad=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('results/01_eml_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ 01_eml_confusion_matrix.png generated")

# 2. DA Confusion Matrix (单独图片)
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(da_cm, annot=True, fmt='d', cmap='Greens', ax=ax,
            xticklabels=all_da_labels, yticklabels=all_da_labels,
            annot_kws={'size': 12, 'weight': 'bold'})
ax.set_xlabel('Model Prediction (DA_label_General)', fontsize=14, labelpad=10)
ax.set_ylabel('Human Annotation (dialogue_act_qida)', fontsize=14, labelpad=10)
ax.set_title(f'Dialogue Act Model Confusion Matrix\nAccuracy: {da_accuracy*100:.2f}%',
             fontsize=16, pad=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('results/02_da_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ 02_da_confusion_matrix.png generated")

# 3. EML Label Distribution (单独图片)
fig, ax = plt.subplots(figsize=(10, 8))
x_pos = np.arange(len(eml_human_dist))
width = 0.35
bars1 = ax.bar(x_pos - width/2, eml_human_dist.values, width,
               label='Human Annotation', color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=2)
bars2 = ax.bar(x_pos + width/2, eml_model_dist.values, width,
               label='Model Prediction', color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=2)

ax.set_xlabel('EML Label', fontsize=14, labelpad=10)
ax.set_ylabel('Sample Count', fontsize=14, labelpad=10)
ax.set_title('EML Label Distribution: Human Annotation vs Model Prediction',
             fontsize=16, pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(['Non-manipulative (0)', 'Manipulative (1)'], fontsize=12)
ax.legend(fontsize=12, loc='upper right')

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{int(height)}', ha='center', va='bottom', fontsize=12, fontweight='bold')
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{int(height)}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('results/03_eml_label_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ 03_eml_label_distribution.png generated")

# 4. DA Label Distribution (单独图片)
fig, ax = plt.subplots(figsize=(14, 8))
all_da_labels_sorted = sorted(all_da_labels)
human_counts = [human_dist.get(label, 0) for label in all_da_labels_sorted]
model_counts = [model_dist.get(label, 0) for label in all_da_labels_sorted]
x_pos_da = np.arange(len(all_da_labels_sorted))
width = 0.35

bars3 = ax.bar(x_pos_da - width/2, human_counts, width,
               label='Human Annotation', color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=2)
bars4 = ax.bar(x_pos_da + width/2, model_counts, width,
               label='Model Prediction', color='#d62728', alpha=0.8, edgecolor='black', linewidth=2)

ax.set_xlabel('Dialogue Act Label', fontsize=14, labelpad=10)
ax.set_ylabel('Sample Count', fontsize=14, labelpad=10)
ax.set_title('Dialogue Act Label Distribution: Human Annotation vs Model Prediction',
             fontsize=16, pad=20)
ax.set_xticks(x_pos_da)
ax.set_xticklabels(all_da_labels_sorted, fontsize=12)
ax.legend(fontsize=12, loc='upper right')

# Add value labels (only for non-zero)
for bar in bars3:
    height = bar.get_height()
    if height > 0:
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
for bar in bars4:
    height = bar.get_height()
    if height > 0:
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('results/04_da_label_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ 04_da_label_distribution.png generated")

print("\n📊 All 4 individual charts generated successfully!")
print("File List:")
print("1. 01_eml_confusion_matrix.png - EML模型混淆矩阵")
print("2. 02_da_confusion_matrix.png - 对话行为模型混淆矩阵")
print("3. 03_eml_label_distribution.png - EML标签分布对比")
print("4. 04_da_label_distribution.png - 对话行为标签分布对比")