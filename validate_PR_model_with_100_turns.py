import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import warnings
import os
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 忽略警告
warnings.filterwarnings('ignore')

# ==================================================
# 1. 配置参数
# ==================================================
DATA_PATH = './data/Persuasion_For_Good/300_dialog_with_result.csv'
MODEL_PATH = './models/PR_model'
MAX_LENGTH = 128
OUTPUT_DIR = './output_data'
PLOT_CONFUSION_MATRIX = True

# 创建输出文件夹
os.makedirs(OUTPUT_DIR, exist_ok = True)

# ==================================================
# 2. 加载数据
# ==================================================
print("=" * 60)
print("Loading Validation Data")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# 可选：只验证100个样本（和参考文件名对应）
sample_size = 100
df = df.sample(n = sample_size, random_state = 42).reset_index(drop = True)

texts = df['Unit'].astype(str).tolist()
labels_raw = df['persuasion_result'].tolist()

# 构建标签映射
unique_labels = sorted(list(set(labels_raw)))
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}
labels = [label2id[label] for label in labels_raw]

print(f"\nLabel mapping: {label2id}")
print(f"Class distribution:\n{pd.Series(labels_raw).value_counts()}")

# ==================================================
# 3. 加载模型和Tokenizer（核心修复：禁用token_type_ids）
# ==================================================
print("\n" + "=" * 60)
print("Loading Pre-trained Model")
print("=" * 60)

# 加载tokenizer并禁用token_type_ids
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast = True
)

# 加载模型
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels = len(unique_labels),
    id2label = id2label,
    label2id = label2id
)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
print(f"Model loaded on device: {device}")


# ==================================================
# 4. 批量预测函数（核心修复：过滤掉token_type_ids参数）
# ==================================================
def batch_predict(texts, batch_size=16):
    """批量预测，避免内存溢出，修复token_type_ids错误"""
    all_predictions = []
    all_probabilities = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # 分词 - 禁用token_type_ids生成
        inputs = tokenizer(
            batch_texts,
            padding = 'max_length',
            truncation = True,
            max_length = MAX_LENGTH,
            return_tensors = 'pt',
            return_token_type_ids = False  # 关键：不生成token_type_ids
        )

        # 移到指定设备
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 预测（禁用梯度计算）
        with torch.no_grad():
            # 双重保险：过滤掉可能存在的token_type_ids
            inputs = {k: v for k, v in inputs.items() if k not in ['token_type_ids']}
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim = -1)
            predictions = torch.argmax(logits, dim = -1)

        # 转换为numpy并收集结果
        all_predictions.extend(predictions.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

    return np.array(all_predictions), np.array(all_probabilities)


# ==================================================
# 5. 执行预测
# ==================================================
print("\n" + "=" * 60)
print("Running Predictions")
print("=" * 60)

# 执行批量预测
predictions, probabilities = batch_predict(texts)

# ==================================================
# 6. 详细评估
# ==================================================
print("\n" + "=" * 60)
print("Detailed Evaluation Results")
print("=" * 60)

# 计算核心指标
accuracy = accuracy_score(labels, predictions)
f1_macro = f1_score(labels, predictions, average = 'macro')
f1_weighted = f1_score(labels, predictions, average = 'weighted')

# 打印核心指标
print(f"Overall Accuracy: {accuracy:.4f}")
print(f"Macro F1 Score: {f1_macro:.4f}")
print(f"Weighted F1 Score: {f1_weighted:.4f}")

# 打印详细分类报告
print("\nDetailed Classification Report:")
class_report = classification_report(
    labels,
    predictions,
    target_names = unique_labels,
    digits = 4,
    output_dict = True
)
print(classification_report(
    labels,
    predictions,
    target_names = unique_labels,
    digits = 4
))

# 计算混淆矩阵
cm = confusion_matrix(labels, predictions)
cm_df = pd.DataFrame(
    cm,
    index = unique_labels,
    columns = unique_labels
)

print("\nConfusion Matrix:")
print(cm_df)

# ==================================================
# 7. 可视化混淆矩阵
# ==================================================
if PLOT_CONFUSION_MATRIX:
    plt.figure(figsize = (10, 8))
    sns.heatmap(
        cm_df,
        annot = True,
        fmt = 'd',
        cmap = 'Blues',
        cbar = True
    )
    plt.title('Confusion Matrix - PR Model Validation (100 Samples)', fontsize = 14)
    plt.xlabel('Predicted Label', fontsize = 12)
    plt.ylabel('True Label', fontsize = 12)
    plt.xticks(rotation = 45)
    plt.yticks(rotation = 0)
    plt.tight_layout()
    cm_save_path = os.path.join(OUTPUT_DIR, 'pr_model_confusion_matrix_100samples.png')
    plt.savefig(cm_save_path, dpi = 300)
    plt.show()
    print(f"\nConfusion matrix saved to: {cm_save_path}")

# ==================================================
# 8. 错误分析示例
# ==================================================
print("\n" + "=" * 60)
print("Error Analysis (First 5 Misclassified Examples)")
print("=" * 60)

# 找出错误分类的样本
errors = []
for i, (true, pred) in enumerate(zip(labels, predictions)):
    if true != pred:
        errors.append({
            'text': texts[i][:200] + '...' if len(texts[i]) > 200 else texts[i],
            'true_label': id2label[true],
            'pred_label': id2label[pred],
            'true_prob': probabilities[i][true],
            'pred_prob': probabilities[i][pred]
        })

# 打印前5个错误样本
if errors:
    for i, error in enumerate(errors[:5]):
        print(f"\nError {i + 1}:")
        print(f"Text: {error['text']}")
        print(f"True Label: {error['true_label']} (Prob: {error['true_prob']:.4f})")
        print(f"Pred Label: {error['pred_label']} (Prob: {error['pred_prob']:.4f})")
else:
    print("No misclassified examples found!")

# ==================================================
# 9. 保存评估结果
# ==================================================
# 1) 保存核心评估指标
metrics_df = pd.DataFrame({
    'metric': ['accuracy', 'f1_macro', 'f1_weighted'],
    'value': [accuracy, f1_macro, f1_weighted]
})
metrics_save_path = os.path.join(OUTPUT_DIR, 'pr_model_validation_metrics_100samples.csv')
metrics_df.to_csv(metrics_save_path, index = False)

# 2) 保存完整预测结果（100样本版本）
results_df = pd.DataFrame({
    'text': texts,
    'true_label': [id2label[l] for l in labels],
    'pred_label': [id2label[p] for p in predictions],
    'is_correct': [l == p for l, p in zip(labels, predictions)]
})

# 添加每个类别的概率
for i, label in enumerate(unique_labels):
    results_df[f'prob_{label}'] = probabilities[:, i]

# 命名和参考文件风格一致
results_save_path = os.path.join(OUTPUT_DIR, '100_sample_with_PR_predictions.csv')
results_df.to_csv(results_save_path, index = False)

# 3) 保存混淆矩阵数据
cm_save_path_csv = os.path.join(OUTPUT_DIR, 'pr_model_confusion_matrix_100samples.csv')
cm_df.to_csv(cm_save_path_csv, index = True)

# 4) 保存分类报告
class_report_df = pd.DataFrame(class_report).T
class_report_save_path = os.path.join(OUTPUT_DIR, 'pr_model_classification_report_100samples.csv')
class_report_df.to_csv(class_report_save_path, index = True)

print(f"\n核心评估指标保存到: {metrics_save_path}")
print(f"100样本预测结果保存到: {results_save_path}")
print(f"混淆矩阵数据保存到: {cm_save_path_csv}")
print(f"分类报告保存到: {class_report_save_path}")

print("\n" + "=" * 60)
print("Validation Completed Successfully!")
print(f"所有输出文件已保存到: {OUTPUT_DIR}")
print("=" * 60)