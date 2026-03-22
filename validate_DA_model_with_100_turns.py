import os
import time
import numpy as np
import pandas as pd
import torch
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import joblib
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# 忽略警告
warnings.filterwarnings('ignore')

# -------------------------- 1. 全局配置 --------------------------
# 模型配置
MODEL_TYPE = "distilbert-base-uncased"
LABEL_TYPE = "General"
MAX_SEQ_LENGTH = 64

# 数据路径配置
DATA_DIR = "data/Persuasion_For_Good"
# 包含真实DA标签的数据源
NEW_DATA_PATH = os.path.join(DATA_DIR, "100_sample_turns_data_with_manual_label.xlsx")
# DA真实标签列名（根据你的数据实际列名修改）
DA_TRUE_LABEL_COL = "dialogue_act_qida"  # 替换为你实际的DA真实标签列名

# 输出路径配置
OUTPUT_DIR = "output_data"
os.makedirs(OUTPUT_DIR, exist_ok = True)
# 预测结果文件
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "100_sample_with_DA.csv")
# 验证分析报告文件
ANALYSIS_REPORT_PATH = os.path.join(OUTPUT_DIR, "DA_validation_analysis.csv")
# 混淆矩阵图片路径
CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, "DA_confusion_matrix.png")

# 模型存储路径
MODEL_BASE_DIR = "./models"
MODEL_SAVE_DIR = os.path.join(MODEL_BASE_DIR, "bert-base-uncased_DA_MRDA_our_label")
LABEL_ENCODER_PATH = os.path.join(MODEL_SAVE_DIR, "label_encoder.pkl")

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device Info: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")


# -------------------------- 2. 加载训练好的模型 --------------------------
def load_trained_model():
    print(f"\nLoading Trained Model from {MODEL_SAVE_DIR}")

    # 检查模型目录是否存在
    if not os.path.exists(MODEL_SAVE_DIR):
        raise FileNotFoundError(f"Model directory not found: {MODEL_SAVE_DIR}")

    # 加载标签编码器
    try:
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        print(f"Loaded Label Encoder (Total Labels: {len(label_encoder.classes_)})")
        print(f"Labels List: {list(label_encoder.classes_)}")
    except Exception as e:
        raise ValueError(f"Failed to load Label Encoder: {e}\nCheck if {LABEL_ENCODER_PATH} exists!")

    # 加载Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_DIR)
        print(f"Loaded Tokenizer from {MODEL_SAVE_DIR}")
    except Exception as e:
        raise ValueError(f"Failed to load Tokenizer: {e}")

    # 加载模型
    try:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_DIR)
        model.to(DEVICE)
        model.eval()
        print(f"Loaded Trained Model (Device: {DEVICE})")
    except Exception as e:
        raise ValueError(f"Failed to load Model: {e}\nCheck model files in {MODEL_SAVE_DIR}!")

    return model, tokenizer, label_encoder


# -------------------------- 3. 加载数据（包含真实标签） --------------------------
def load_persuasion_data(file_path, true_label_col):
    print(f"\nLoading Data from {file_path}")
    start_time = time.time()

    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # 读取Excel文件
    df = pd.read_excel(file_path)

    # 检查关键列
    required_cols = ["Sentence", true_label_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in Excel: {missing_cols}\nAvailable columns: {list(df.columns)}")

    # 过滤有效文本
    df["Sentence_str"] = df["Sentence"].astype(str).apply(lambda x: x.strip())
    valid_text_mask = df["Sentence"].notna() & (df["Sentence_str"] != "")

    # 过滤有效标签（非空且在标签编码器的类别中）
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    valid_label_mask = df[true_label_col].notna() & df[true_label_col].isin(label_encoder.classes_)

    # 合并过滤条件
    valid_mask = valid_text_mask & valid_label_mask
    raw_df = df[valid_mask].copy().drop(columns = ["Sentence_str"])

    # 提取文本和真实标签
    texts = raw_df["Sentence"].astype(str).apply(lambda x: x.strip()).tolist()
    true_labels = raw_df[true_label_col].tolist()

    load_time = time.time() - start_time
    print(f"Loaded Data Summary:")
    print(f"  - Total rows in Excel: {len(df)}")
    print(f"  - Valid rows (text + label): {len(raw_df)}")
    print(f"  - Invalid rows filtered: {len(df) - len(raw_df)}")
    print(f"  - Time Cost: {load_time:.2f}s")
    print(f"  - Preserved Columns: {list(raw_df.columns)}")

    # 打印真实标签分布
    print(f"\nTrue DA Label Distribution:")
    label_counts = pd.Series(true_labels).value_counts()
    for label, count in label_counts.items():
        percentage = count / len(true_labels) * 100
        print(f"  {label}: {count} ({percentage:.2f}%)")

    return texts, true_labels, raw_df


# -------------------------- 4. 批量预测DA标签（带概率输出） --------------------------
def batch_predict_da(model, tokenizer, label_encoder, texts, batch_size=64):
    all_pred_labels = []
    all_pred_probs = []  # 保存预测概率
    model.eval()

    print(f"\nStarting DA Label Prediction (Batch Size: {batch_size})")
    start_time = time.time()

    # 分批次处理
    for i in tqdm(range(0, len(texts), batch_size), desc = "Predicting Batches"):
        batch_texts = texts[i:i + batch_size]

        # 文本编码（修复token_type_ids问题）
        encodings = tokenizer(
            batch_texts,
            truncation = True,
            padding = "max_length",
            max_length = MAX_SEQ_LENGTH,
            return_tensors = "pt",
            return_token_type_ids = False  # 关键：禁用token_type_ids
        )

        input_ids = encodings["input_ids"].to(DEVICE)
        attention_mask = encodings["attention_mask"].to(DEVICE)

        # 预测
        with torch.no_grad():
            # 过滤掉不支持的参数
            outputs = model(input_ids = input_ids, attention_mask = attention_mask)
            logits = outputs.logits
            # 计算概率
            probs = torch.nn.functional.softmax(logits, dim = -1)
            # 获取预测ID
            pred_ids = torch.argmax(logits, dim = -1).cpu().numpy()
            # 获取预测概率
            pred_probs = probs.cpu().numpy()

        # 解码标签
        batch_labels = label_encoder.inverse_transform(pred_ids)
        all_pred_labels.extend(batch_labels)
        all_pred_probs.extend(pred_probs)

    pred_time = time.time() - start_time
    avg_time_per_text = pred_time / len(texts) * 1000
    print(f"Prediction Completed!")
    print(f"  - Total Time: {pred_time:.2f}s")
    print(f"  - Avg Time per Utterance: {avg_time_per_text:.2f}ms")
    print(f"  - Total Predicted Labels: {len(all_pred_labels)}")

    return all_pred_labels, np.array(all_pred_probs)


# -------------------------- 5. 对比分析（核心新增功能） --------------------------
def analyze_validation_results(true_labels, pred_labels, pred_probs, label_encoder):
    print(f"\n{'=' * 60}")
    print("DA Model Validation Analysis")
    print(f"{'=' * 60}")

    # 1. 基础指标计算
    accuracy = accuracy_score(true_labels, pred_labels)
    f1_macro = f1_score(true_labels, pred_labels, average = 'macro')
    f1_weighted = f1_score(true_labels, pred_labels, average = 'weighted')

    print(f"\n1. Core Metrics:")
    print(f"   - Overall Accuracy: {accuracy:.4f}")
    print(f"   - Macro F1 Score: {f1_macro:.4f}")
    print(f"   - Weighted F1 Score: {f1_weighted:.4f}")

    # 2. 详细分类报告
    print(f"\n2. Detailed Classification Report:")
    class_report = classification_report(
        true_labels,
        pred_labels,
        target_names = label_encoder.classes_,
        digits = 4,
        output_dict = True
    )
    print(classification_report(
        true_labels,
        pred_labels,
        target_names = label_encoder.classes_,
        digits = 4
    ))

    # 3. 混淆矩阵
    print(f"\n3. Confusion Matrix:")
    cm = confusion_matrix(true_labels, pred_labels, labels = label_encoder.classes_)
    cm_df = pd.DataFrame(
        cm,
        index = [f"True: {label}" for label in label_encoder.classes_],
        columns = [f"Pred: {label}" for label in label_encoder.classes_]
    )
    print(cm_df)

    # 4. 可视化混淆矩阵
    plt.figure(figsize = (12, 10))
    sns.heatmap(
        cm_df,
        annot = True,
        fmt = 'd',
        cmap = 'Blues',
        cbar = True,
        linewidths = 0.5
    )
    plt.title(f'DA Model Confusion Matrix (Accuracy: {accuracy:.4f})', fontsize = 14)
    plt.xlabel('Predicted Label', fontsize = 12)
    plt.ylabel('True Label', fontsize = 12)
    plt.xticks(rotation = 45, ha = 'right')
    plt.yticks(rotation = 0)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi = 300)
    plt.show()
    print(f"\nConfusion matrix saved to: {CONFUSION_MATRIX_PATH}")

    # 5. 错误分析
    print(f"\n4. Error Analysis (First 5 Misclassified Examples):")
    errors = []
    for i, (true, pred) in enumerate(zip(true_labels, pred_labels)):
        if true != pred:
            # 获取真实标签和预测标签的概率
            label_idx = {label: idx for idx, label in enumerate(label_encoder.classes_)}
            true_prob = pred_probs[i][label_idx[true]]
            pred_prob = pred_probs[i][label_idx[pred]]
            errors.append({
                'index': i,
                'true_label': true,
                'pred_label': pred,
                'true_label_prob': true_prob,
                'pred_label_prob': pred_prob
            })

    if errors:
        for i, error in enumerate(errors[:5]):
            print(f"\n   Error {i + 1} (Index: {error['index']}):")
            print(f"      True Label: {error['true_label']} (Prob: {error['true_label_prob']:.4f})")
            print(f"      Pred Label: {error['pred_label']} (Prob: {error['pred_label_prob']:.4f})")
        print(f"\n   Total Misclassified Samples: {len(errors)} ({len(errors) / len(true_labels) * 100:.2f}%)")
    else:
        print(f"   No misclassified samples found!")

    # 6. 保存分析报告
    analysis_data = {
        'metric': ['accuracy', 'f1_macro', 'f1_weighted'],
        'value': [accuracy, f1_macro, f1_weighted]
    }
    analysis_df = pd.DataFrame(analysis_data)
    analysis_df.to_csv(ANALYSIS_REPORT_PATH, index = False)

    # 保存分类报告
    class_report_df = pd.DataFrame(class_report).T
    class_report_save_path = os.path.join(OUTPUT_DIR, "DA_classification_report.csv")
    class_report_df.to_csv(class_report_save_path, index = True)

    # 保存混淆矩阵
    cm_save_path = os.path.join(OUTPUT_DIR, "DA_confusion_matrix.csv")
    cm_df.to_csv(cm_save_path, index = True)

    print(f"\n5. Analysis Files Saved:")
    print(f"   - Core Metrics: {ANALYSIS_REPORT_PATH}")
    print(f"   - Classification Report: {class_report_save_path}")
    print(f"   - Confusion Matrix: {cm_save_path}")
    print(f"   - Confusion Matrix Plot: {CONFUSION_MATRIX_PATH}")

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm_df,
        'classification_report': class_report,
        'error_count': len(errors)
    }


# -------------------------- 6. 保存结果（包含对比列） --------------------------
def save_labeled_results(raw_df, pred_labels, pred_probs, true_labels, label_encoder, output_path):
    print(f"\nSaving Labeled Results to {output_path}")

    # 确保长度一致
    if len(raw_df) != len(pred_labels) or len(raw_df) != len(true_labels):
        raise ValueError(
            f"Length mismatch: raw_df({len(raw_df)}), pred_labels({len(pred_labels)}), true_labels({len(true_labels)})")

    # 新增列
    result_df = raw_df.copy()
    result_df[f"DA_label_{LABEL_TYPE}"] = pred_labels
    result_df[f"DA_true_label"] = true_labels
    result_df[f"DA_pred_correct"] = result_df[f"DA_true_label"] == result_df[f"DA_label_{LABEL_TYPE}"]

    # 添加预测概率列
    label_classes = label_encoder.classes_
    for i, label in enumerate(label_classes):
        result_df[f"DA_prob_{label}"] = pred_probs[:, i]

    # 保存为CSV
    result_df.to_csv(output_path, index = False, encoding = "utf-8-sig")

    # 输出统计信息
    print(f"Results Saved Successfully!")
    print(f"\nResult Sample (First 5 Rows):")
    display_cols = ["Sentence", f"DA_true_label", f"DA_label_{LABEL_TYPE}", f"DA_pred_correct"]
    print(result_df[display_cols].head())

    print(f"\nPredicted DA Label Distribution:")
    pred_label_count = result_df[f"DA_label_{LABEL_TYPE}"].value_counts()
    for label, count in pred_label_count.items():
        percentage = count / len(result_df) * 100
        print(f"  {label}: {count} rows ({percentage:.2f}%)")

    # 输出正确/错误统计
    correct_count = result_df[f"DA_pred_correct"].sum()
    incorrect_count = len(result_df) - correct_count
    print(f"\nPrediction Accuracy Summary:")
    print(f"  - Correct Predictions: {correct_count} ({correct_count / len(result_df) * 100:.2f}%)")
    print(f"  - Incorrect Predictions: {incorrect_count} ({incorrect_count / len(result_df) * 100:.2f}%)")


# -------------------------- 主函数 --------------------------
def main():
    print("=" * 60)
    print("DA Model Validation with Comparative Analysis")
    print("=" * 60)

    # 检查数据文件
    if not os.path.exists(NEW_DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {NEW_DATA_PATH}")

    # 检查模型目录
    print(f"\nChecking model directory: {MODEL_SAVE_DIR}")
    if not os.path.exists(MODEL_SAVE_DIR):
        raise FileNotFoundError(f"Model directory not found: {MODEL_SAVE_DIR}")

    # 列出模型目录文件（调试用）
    print(f"Files in model directory:")
    for file in os.listdir(MODEL_SAVE_DIR):
        print(f"  - {file}")

    # 加载模型
    model, tokenizer, label_encoder = load_trained_model()

    # 加载数据（包含真实标签）
    texts, true_labels, raw_df = load_persuasion_data(NEW_DATA_PATH, DA_TRUE_LABEL_COL)

    # 空数据检查
    if len(texts) == 0:
        print("No valid texts found in the input file!")
        return

    # 预测标签
    pred_labels, pred_probs = batch_predict_da(model, tokenizer, label_encoder, texts)

    # 保存结果（包含真实标签对比）
    save_labeled_results(raw_df, pred_labels, pred_probs, true_labels, label_encoder, OUTPUT_PATH)

    # 执行对比分析
    analysis_results = analyze_validation_results(true_labels, pred_labels, pred_probs, label_encoder)

    print(f"\n{'=' * 60}")
    print(f"DA Validation Completed Successfully!")
    print(f"{'=' * 60}")
    print(f"Key Results:")
    print(f"  - Accuracy: {analysis_results['accuracy']:.4f}")
    print(f"  - Macro F1: {analysis_results['f1_macro']:.4f}")
    print(f"  - Weighted F1: {analysis_results['f1_weighted']:.4f}")
    print(f"  - Misclassified Samples: {analysis_results['error_count']}")
    print(f"{'=' * 60}")
    print(f"Output Files:")
    print(f"  - Predicted Results: {OUTPUT_PATH}")
    print(f"  - Analysis Report: {ANALYSIS_REPORT_PATH}")
    print(f"  - Confusion Matrix: {CONFUSION_MATRIX_PATH}")
    print(f"{'=' * 60}")


# -------------------------- 运行入口 --------------------------
if __name__ == "__main__":
    main()