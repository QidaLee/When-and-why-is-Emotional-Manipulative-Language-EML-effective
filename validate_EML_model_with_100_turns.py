import torch
import pandas as pd
import os
import re
import string
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import Dataset, ClassLabel
from tqdm import tqdm

# 忽略警告
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ====================== 核心配置项（根据实际情况修改） ======================
# 数据路径
DATA_PATH = r"D:\Master_study\master_thesis_programs\my_project\data\Persuasion_For_Good\100_sample_turns_data_with_manual_label.xlsx"
# EML真实标签列名（根据你的数据修改）
EML_TRUE_LABEL_COL = "manipulation_qida"  # 替换为实际的EML真实标签列名
# 输出配置
OUTPUT_DIR = r"D:\Master_study\master_thesis_programs\my_project\output_data"
os.makedirs(OUTPUT_DIR, exist_ok = True)
# 预测结果文件
PREDICTION_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "100_sample_eml_validation_results.xlsx")
# 分析报告文件
ANALYSIS_REPORT_PATH = os.path.join(OUTPUT_DIR, "EML_validation_analysis.csv")
# 混淆矩阵图片路径
CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, "EML_confusion_matrix.png")

# 模型配置
MODEL_SAVE_PATH = "./models/mentalmanip_model"
# 置信度阈值（和原代码保持一致）
CONFIDENCE_THRESHOLD = 0.6
# 文本最大长度
MAX_LENGTH = 128


# ===========================================================================

# ---------------------- 1. 文本清洗函数（和训练时保持一致） ----------------------
def clean_text(text):
    """文本清洗：与训练集预处理逻辑完全一致"""
    text = str(text).lower().strip()
    # 保留关键标点，移除其他标点
    keep_punct = {"?", "!", "."}
    text = "".join([c for c in text if c not in string.punctuation or c in keep_punct])
    # 移除多余空格
    text = re.sub(r"\s+", " ", text)
    return text


# ---------------------- 2. 加载模型和Tokenizer ----------------------
def load_eml_model():
    """加载训练好的EML检测模型"""
    print("=" * 60)
    print("Loading EML Detection Model")
    print("=" * 60)

    # 检查模型路径
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model path not found: {MODEL_SAVE_PATH}\n"
            "Please train the model first by setting RETRAIN_MODEL = True in the training script."
        )

    # 加载Tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_SAVE_PATH)
    print(f"✅ Tokenizer loaded from: {MODEL_SAVE_PATH}")

    # 加载模型
    model = BertForSequenceClassification.from_pretrained(
        MODEL_SAVE_PATH,
        num_labels = 2,
        ignore_mismatched_sizes = True
    )

    # 设置设备（CPU/GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # 设置为评估模式

    print(f"✅ Model loaded successfully (Device: {device})")
    print(f"✅ Confidence threshold for EML label: {CONFIDENCE_THRESHOLD}")

    return tokenizer, model, device


# ---------------------- 3. 加载并预处理验证数据 ----------------------
def load_validation_data():
    """加载验证数据，包含真实标签"""
    print("\n" + "=" * 60)
    print("Loading Validation Data")
    print("=" * 60)

    # 检查数据文件
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    # 读取Excel文件
    df = pd.read_excel(DATA_PATH, dtype = str)
    print(f"📊 Total rows in raw data: {len(df)}")
    print(f"📋 Columns available: {df.columns.tolist()}")

    # 检查真实标签列
    if EML_TRUE_LABEL_COL not in df.columns:
        raise ValueError(
            f"EML true label column '{EML_TRUE_LABEL_COL}' not found!\n"
            f"Available columns: {df.columns.tolist()}\n"
            "Please check and modify EML_TRUE_LABEL_COL in the configuration."
        )

    # 自动识别文本列
    text_column = None
    for col in df.columns:
        if col.lower() in ["sentence", "text", "content", "dialogue", "unit"]:
            text_column = col
            break
    if not text_column:
        text_column = df.columns[0]
    print(f"🔤 Identified text column: '{text_column}'")

    # 数据清洗和过滤
    # 1. 清洗文本
    df["clean_text"] = df[text_column].fillna("").apply(clean_text)

    # 2. 过滤有效文本（长度>5）
    valid_text_mask = df["clean_text"].str.len() > 5

    # 3. 过滤有效标签（只能是0/1）
    df[EML_TRUE_LABEL_COL] = pd.to_numeric(df[EML_TRUE_LABEL_COL], errors = 'coerce')
    valid_label_mask = df[EML_TRUE_LABEL_COL].isin([0, 1]) & df[EML_TRUE_LABEL_COL].notna()

    # 4. 合并过滤条件
    valid_mask = valid_text_mask & valid_label_mask
    df_valid = df[valid_mask].copy()

    print(f"📊 Valid data after filtering: {len(df_valid)} rows")
    print(f"❌ Filtered out: {len(df) - len(df_valid)} rows (invalid text/label)")

    # 真实标签分布
    true_label_counts = df_valid[EML_TRUE_LABEL_COL].value_counts()
    print(f"\n🎯 True EML Label Distribution:")
    print(f"   - Non-EML (0): {true_label_counts.get(0, 0)} ({true_label_counts.get(0, 0) / len(df_valid) * 100:.2f}%)")
    print(f"   - EML (1): {true_label_counts.get(1, 0)} ({true_label_counts.get(1, 0) / len(df_valid) * 100:.2f}%)")

    return df_valid, text_column


# ---------------------- 4. 批量预测EML标签 ----------------------
def batch_predict_eml(df_valid, tokenizer, model, device):
    """批量预测EML标签，返回预测结果和置信度"""
    print("\n" + "=" * 60)
    print("Running EML Prediction (Batch Mode)")
    print("=" * 60)

    text_list = df_valid["clean_text"].tolist()
    all_pred_labels = []
    all_pred_probs = []
    all_confidences = []

    # 批量处理（避免内存溢出）
    batch_size = 16
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(text_list), batch_size), desc = "Predicting EML Labels"):
            batch_texts = text_list[i:i + batch_size]

            # 分词
            inputs = tokenizer(
                batch_texts,
                truncation = True,
                padding = "max_length",
                max_length = MAX_LENGTH,
                return_tensors = "pt"
            ).to(device)

            # 推理
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim = 1)

            # 获取置信度和预测标签
            batch_confidences = torch.max(probs, dim = 1).values.cpu().numpy()
            batch_pred_ids = torch.argmax(probs, dim = 1).cpu().numpy()

            # 应用置信度阈值（和原代码逻辑一致）
            batch_pred_labels = []
            for pred_id, confidence in zip(batch_pred_ids, batch_confidences):
                if pred_id == 1 and confidence >= CONFIDENCE_THRESHOLD:
                    batch_pred_labels.append(1)
                else:
                    batch_pred_labels.append(0)

            # 收集结果
            all_pred_labels.extend(batch_pred_labels)
            all_pred_probs.extend(probs.cpu().numpy())
            all_confidences.extend([round(c, 3) for c in batch_confidences])

    # 添加预测结果到DataFrame
    df_valid["eml_pred_label"] = all_pred_labels
    df_valid["eml_confidence"] = all_confidences
    df_valid["eml_pred_correct"] = df_valid[EML_TRUE_LABEL_COL] == df_valid["eml_pred_label"]

    # 添加各类别的概率
    df_valid["eml_prob_non_eml"] = [p[0] for p in all_pred_probs]
    df_valid["eml_prob_eml"] = [p[1] for p in all_pred_probs]

    # 预测结果分布
    pred_label_counts = df_valid["eml_pred_label"].value_counts()
    print(f"\n📈 Predicted EML Label Distribution:")
    print(f"   - Non-EML (0): {pred_label_counts.get(0, 0)} ({pred_label_counts.get(0, 0) / len(df_valid) * 100:.2f}%)")
    print(f"   - EML (1): {pred_label_counts.get(1, 0)} ({pred_label_counts.get(1, 0) / len(df_valid) * 100:.2f}%)")

    # 预测准确性统计
    correct_count = df_valid["eml_pred_correct"].sum()
    accuracy = correct_count / len(df_valid)
    print(f"\n🎯 Prediction Accuracy (preliminary): {accuracy:.4f} ({correct_count}/{len(df_valid)})")

    return df_valid, np.array(all_pred_probs)


# ---------------------- 5. 验证分析（核心功能） ----------------------
def analyze_eml_validation(df_valid, pred_probs):
    """完整的EML验证分析：指标计算、混淆矩阵、错误分析"""
    print("\n" + "=" * 60)
    print("EML Model Validation Analysis")
    print("=" * 60)

    # 提取真实标签和预测标签
    true_labels = df_valid[EML_TRUE_LABEL_COL].values
    pred_labels = df_valid["eml_pred_label"].values

    # 1. 核心指标计算
    accuracy = accuracy_score(true_labels, pred_labels)
    f1_binary = f1_score(true_labels, pred_labels, average = "binary")  # 二分类F1
    f1_macro = f1_score(true_labels, pred_labels, average = "macro")

    print(f"\n1. Core Evaluation Metrics:")
    print(f"   - Overall Accuracy: {accuracy:.4f}")
    print(f"   - Binary F1 Score (EML detection): {f1_binary:.4f}")
    print(f"   - Macro F1 Score: {f1_macro:.4f}")

    # 2. 详细分类报告
    print(f"\n2. Detailed Classification Report:")
    class_report = classification_report(
        true_labels,
        pred_labels,
        target_names = ["Non-EML (0)", "EML (1)"],
        digits = 4,
        output_dict = True
    )
    print(classification_report(
        true_labels,
        pred_labels,
        target_names = ["Non-EML (0)", "EML (1)"],
        digits = 4
    ))

    # 3. 混淆矩阵
    print(f"\n3. Confusion Matrix:")
    cm = confusion_matrix(true_labels, pred_labels)
    cm_df = pd.DataFrame(
        cm,
        index = ["True: Non-EML (0)", "True: EML (1)"],
        columns = ["Pred: Non-EML (0)", "Pred: EML (1)"]
    )
    print(cm_df)

    # 可视化混淆矩阵
    plt.figure(figsize = (8, 6))
    sns.heatmap(
        cm_df,
        annot = True,
        fmt = 'd',
        cmap = 'Blues',
        cbar = True,
        linewidths = 0.5
    )
    plt.title(f'EML Model Confusion Matrix (Accuracy: {accuracy:.4f})', fontsize = 14)
    plt.xlabel('Predicted Label', fontsize = 12)
    plt.ylabel('True Label', fontsize = 12)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi = 300)
    plt.show()
    print(f"\n📊 Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")

    # 4. 错误分析
    print(f"\n4. Error Analysis (First 5 Misclassified Examples):")
    # 筛选错误样本
    error_df = df_valid[~df_valid["eml_pred_correct"]].copy()

    if len(error_df) > 0:
        # 添加错误样本的概率信息
        error_df["true_label_prob"] = error_df.apply(
            lambda row: row["eml_prob_non_eml"] if row[EML_TRUE_LABEL_COL] == 0 else row["eml_prob_eml"],
            axis = 1
        )
        error_df["pred_label_prob"] = error_df.apply(
            lambda row: row["eml_prob_non_eml"] if row["eml_pred_label"] == 0 else row["eml_prob_eml"],
            axis = 1
        )

        # 打印前5个错误样本
        for i, (idx, row) in enumerate(error_df.head(5).iterrows()):
            print(f"\n   Error {i + 1} (Index: {idx}):")
            print(f"      Text: {row['clean_text'][:100]}..." if len(
                row['clean_text']) > 100 else f"      Text: {row['clean_text']}")
            print(
                f"      True Label: {'EML (1)' if row[EML_TRUE_LABEL_COL] == 1 else 'Non-EML (0)'} (Prob: {row['true_label_prob']:.4f})")
            print(
                f"      Pred Label: {'EML (1)' if row['eml_pred_label'] == 1 else 'Non-EML (0)'} (Prob: {row['pred_label_prob']:.4f})")
            print(f"      Confidence: {row['eml_confidence']}")

        # 错误类型统计
        false_positives = len(error_df[(error_df[EML_TRUE_LABEL_COL] == 0) & (error_df["eml_pred_label"] == 1)])
        false_negatives = len(error_df[(error_df[EML_TRUE_LABEL_COL] == 1) & (error_df["eml_pred_label"] == 0)])

        print(f"\n   Error Statistics:")
        print(f"   - Total Misclassified: {len(error_df)} ({len(error_df) / len(df_valid) * 100:.2f}%)")
        print(f"   - False Positives (Non-EML predicted as EML): {false_positives}")
        print(f"   - False Negatives (EML predicted as Non-EML): {false_negatives}")
    else:
        print(f"   ✨ No misclassified samples found!")

    # 5. 保存分析报告
    # 核心指标
    analysis_df = pd.DataFrame({
        'metric': ['accuracy', 'binary_f1', 'macro_f1'],
        'value': [accuracy, f1_binary, f1_macro]
    })
    analysis_df.to_csv(ANALYSIS_REPORT_PATH, index = False)

    # 分类报告
    class_report_df = pd.DataFrame(class_report).T
    class_report_save_path = os.path.join(OUTPUT_DIR, "EML_classification_report.csv")
    class_report_df.to_csv(class_report_save_path, index = True)

    # 混淆矩阵
    cm_save_path = os.path.join(OUTPUT_DIR, "EML_confusion_matrix.csv")
    cm_df.to_csv(cm_save_path, index = True)

    print(f"\n5. Analysis Files Saved:")
    print(f"   - Core Metrics: {ANALYSIS_REPORT_PATH}")
    print(f"   - Classification Report: {class_report_save_path}")
    print(f"   - Confusion Matrix: {cm_save_path}")
    print(f"   - Confusion Matrix Plot: {CONFUSION_MATRIX_PATH}")

    # 返回核心指标
    return {
        'accuracy': accuracy,
        'binary_f1': f1_binary,
        'macro_f1': f1_macro,
        'error_count': len(error_df) if len(error_df) > 0 else 0
    }


# ---------------------- 6. 保存验证结果 ----------------------
def save_validation_results(df_valid, text_column):
    """保存完整的验证结果"""
    # 整理列顺序，保留原始列 + 预测结果列
    result_columns = [col for col in df_valid.columns if col != "clean_text"]
    # 将文本列放在第一列
    result_columns.insert(0, text_column)
    result_df = df_valid[result_columns].copy()

    # 保存为Excel
    result_df.to_excel(PREDICTION_OUTPUT_PATH, index = False)
    print(f"\n💾 Validation results saved to: {PREDICTION_OUTPUT_PATH}")

    # 最终统计
    print(f"\n📋 Final Validation Summary:")
    print(f"   - Total Valid Samples: {len(result_df)}")
    print(f"   - Correct Predictions: {result_df['eml_pred_correct'].sum()}")
    print(f"   - Incorrect Predictions: {len(result_df) - result_df['eml_pred_correct'].sum()}")
    print(f"   - Final Accuracy: {result_df['eml_pred_correct'].mean():.4f}")


# ---------------------- 主函数 ----------------------
def main():
    print("=" * 70)
    print("EML Model Validation Script (Emotional Manipulation Detection)")
    print("=" * 70)

    # 1. 加载模型
    tokenizer, model, device = load_eml_model()

    # 2. 加载验证数据
    df_valid, text_column = load_validation_data()

    # 3. 批量预测
    df_valid, pred_probs = batch_predict_eml(df_valid, tokenizer, model, device)

    # 4. 验证分析
    analysis_results = analyze_eml_validation(df_valid, pred_probs)

    # 5. 保存结果
    save_validation_results(df_valid, text_column)

    print("\n" + "=" * 70)
    print("EML Validation Completed Successfully!")
    print("=" * 70)
    print(f"🎯 Key Results:")
    print(f"   - Accuracy: {analysis_results['accuracy']:.4f}")
    print(f"   - Binary F1 (EML): {analysis_results['binary_f1']:.4f}")
    print(f"   - Macro F1: {analysis_results['macro_f1']:.4f}")
    print(f"   - Misclassified Samples: {analysis_results['error_count']}")
    print("=" * 70)


# ---------------------- 运行入口 ----------------------
if __name__ == "__main__":
    main()