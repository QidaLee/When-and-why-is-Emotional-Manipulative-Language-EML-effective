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
# 1. 配置参数（更新数据源和标签列）
# ==================================================
# 新的数据源（Excel文件）
DATA_PATH = './data/Persuasion_For_Good/100_sample_turns_data_with_manual_label.xlsx'
MODEL_PATH = './models/PR_model'
MAX_LENGTH = 128
OUTPUT_DIR = './output_data'
PLOT_CONFUSION_MATRIX = True

# 真实标签列名
LABEL_COLUMN = 'persuasion_result_qida'
# 标签值映射（数字 -> 文本描述）
LABEL_VALUE_MAP = {
    1: 'compliance',
    0: 'neutral',
    -1: 'resistance'
}

# 创建输出文件夹
os.makedirs(OUTPUT_DIR, exist_ok = True)

# ==================================================
# 2. 加载Excel数据并处理标签
# ==================================================
print("=" * 60)
print("Loading Validation Data (Excel)")
print("=" * 60)

# 读取Excel文件
df = pd.read_excel(DATA_PATH)
print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# 检查标签列是否存在
if LABEL_COLUMN not in df.columns:
    raise ValueError(f"标签列 {LABEL_COLUMN} 不存在于数据文件中！")

# 提取文本和标签（使用指定的标签列）
texts = df['Unit'].astype(str).tolist() if 'Unit' in df.columns else df.iloc[:, 0].astype(str).tolist()
labels_raw = df[LABEL_COLUMN].tolist()

# 数据清洗：移除空值或无效标签
valid_indices = []
valid_texts = []
valid_labels = []
for idx, (text, label) in enumerate(zip(texts, labels_raw)):
    if pd.notna(label) and label in LABEL_VALUE_MAP.keys():
        valid_indices.append(idx)
        valid_texts.append(text)
        valid_labels.append(label)

print(f"\n原始数据量: {len(texts)}, 有效标签数据量: {len(valid_labels)}")

# 构建标签映射（适配-1/0/1的标签值）
# 先转换为文本描述，再构建id映射（保持和训练时的标签命名一致）
labels_text = [LABEL_VALUE_MAP[label] for label in valid_labels]
unique_labels = sorted(LABEL_VALUE_MAP.values())  # ['compliance', 'neutral', 'resistance']
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

# 将标签转换为模型需要的id
labels = [label2id[LABEL_VALUE_MAP[label]] for label in valid_labels]

print(f"\n标签映射（数字值 -> 文本 -> ID）:")
for num_val, text_val in LABEL_VALUE_MAP.items():
    print(f"  {num_val} -> {text_val} -> ID: {label2id[text_val]}")

print(f"\nClass distribution (文本标签):\n{pd.Series(labels_text).value_counts()}")
print(f"Class distribution (原始数字标签):\n{pd.Series(valid_labels).value_counts()}")

# ==================================================
# 3. 加载模型和Tokenizer
# ==================================================
print("\n" + "=" * 60)
print("Loading Pre-trained Model")
print("=" * 60)

# 加载tokenizer并禁用token_type_ids
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast = True
)

# 加载模型（确保标签数量和映射匹配）
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
# 4. 批量预测函数（修复token_type_ids问题）
# ==================================================
def batch_predict(texts, batch_size=16):
    """批量预测，避免内存溢出"""
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
predictions, probabilities = batch_predict(valid_texts)

# 将预测ID转换回文本标签和原始数字标签
pred_labels_text = [id2label[pred_id] for pred_id in predictions]
# 反向映射：文本标签 -> 原始数字
text2num = {v: k for k, v in LABEL_VALUE_MAP.items()}
pred_labels_num = [text2num[text] for text in pred_labels_text]
true_labels_num = valid_labels  # 原始数字标签

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

# 打印详细分类报告（使用文本标签，更易读）
print("\nDetailed Classification Report (文本标签):")
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

# 打印数字标签的分类报告（便于对照原始数据）
print("\nClassification Report (原始数字标签):")
num_target_names = [f"{num} ({text})" for num, text in LABEL_VALUE_MAP.items()]
num_labels_mapped = [list(LABEL_VALUE_MAP.keys()).index(text2num[label]) for label in labels_text]
num_pred_mapped = [list(LABEL_VALUE_MAP.keys()).index(text2num[label]) for label in pred_labels_text]
print(classification_report(
    num_labels_mapped,
    num_pred_mapped,
    target_names = num_target_names,
    digits = 4
))

# 计算混淆矩阵（文本标签）
cm = confusion_matrix(labels, predictions)
cm_df = pd.DataFrame(
    cm,
    index = [f"True: {label}" for label in unique_labels],
    columns = [f"Pred: {label}" for label in unique_labels]
)

print("\nConfusion Matrix (文本标签):")
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
    plt.title('Confusion Matrix - PR Model (100 Samples)', fontsize = 14)
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
for i, (true_id, pred_id) in enumerate(zip(labels, predictions)):
    if true_id != pred_id:
        errors.append({
            'text': valid_texts[i][:200] + '...' if len(valid_texts[i]) > 200 else valid_texts[i],
            'true_label_num': true_labels_num[i],
            'true_label_text': id2label[true_id],
            'pred_label_num': pred_labels_num[i],
            'pred_label_text': id2label[pred_id],
            'true_prob': probabilities[i][true_id],
            'pred_prob': probabilities[i][pred_id]
        })

# 打印前5个错误样本
if errors:
    for i, error in enumerate(errors[:5]):
        print(f"\nError {i + 1}:")
        print(f"Text: {error['text']}")
        print(f"True: {error['true_label_num']} ({error['true_label_text']}) (Prob: {error['true_prob']:.4f})")
        print(f"Pred: {error['pred_label_num']} ({error['pred_label_text']}) (Prob: {error['pred_prob']:.4f})")
else:
    print("No misclassified examples found!")

# ==================================================
# 9. 保存评估结果（包含原始数字标签）
# ==================================================
# 1) 保存核心评估指标
metrics_df = pd.DataFrame({
    'metric': ['accuracy', 'f1_macro', 'f1_weighted'],
    'value': [accuracy, f1_macro, f1_weighted]
})
metrics_save_path = os.path.join(OUTPUT_DIR, 'pr_model_validation_metrics_100samples.csv')
metrics_df.to_csv(metrics_save_path, index = False)

# 2) 保存完整预测结果（包含原始数字标签）
results_df = pd.DataFrame({
    'text': valid_texts,
    'true_label_num': true_labels_num,
    'true_label_text': labels_text,
    'pred_label_num': pred_labels_num,
    'pred_label_text': pred_labels_text,
    'is_correct': [t == p for t, p in zip(labels, predictions)]
})

# 添加每个类别的概率
for i, label in enumerate(unique_labels):
    results_df[f'prob_{label}'] = probabilities[:, i]

# 命名和参考文件风格一致
results_save_path = os.path.join(OUTPUT_DIR, '100_sample_with_PR_predictions.csv')
results_df.to_csv(results_save_path, index = False, encoding = 'utf-8-sig')

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