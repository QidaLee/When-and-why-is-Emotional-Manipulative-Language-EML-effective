import os
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import joblib
from tqdm import tqdm

# -------------------------- 1. 全局配置（根据你的实际路径修改） --------------------------
# 模型类型改为distilbert（因为你的模型是distilbert-base-uncased）
MODEL_TYPE = "distilbert-base-uncased"  # 从bert改为distilbert
LABEL_TYPE = "General"  # 保持与你训练时一致
MAX_SEQ_LENGTH = 64

# 数据路径配置（保持不变或按需修改）
DATA_DIR = "data/Persuasion_For_Good"
NEW_DATA_PATH = os.path.join(DATA_DIR, "100_sample_turns_data_with_manual_label.xlsx")

# 输出路径配置
OUTPUT_DIR = "output_data"
os.makedirs(OUTPUT_DIR, exist_ok = True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "100_sample_with_DA.csv")

# 【关键修改】模型存储路径 - 直接指向你训练好的模型位置
MODEL_BASE_DIR = "./models"
MODEL_SAVE_DIR = os.path.join(MODEL_BASE_DIR, "bert-base-uncased_DA_MRDA_our_label")  # 直接使用文件夹名
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

    # 加载Tokenizer（从模型保存目录加载，会自动识别为distilbert）
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_DIR)
        print(f"Loaded Tokenizer from {MODEL_SAVE_DIR}")
    except Exception as e:
        raise ValueError(f"Failed to load Tokenizer: {e}")

    # 加载模型
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_SAVE_DIR
            # 不需要指定num_labels，因为模型配置文件里已经保存了
        )
        model.to(DEVICE)
        model.eval()
        print(f"Loaded Trained Model (Device: {DEVICE})")
    except Exception as e:
        raise ValueError(f"Failed to load Model: {e}\nCheck model files in {MODEL_SAVE_DIR}!")

    return model, tokenizer, label_encoder


# -------------------------- 3. 加载数据（适配Sentence列） --------------------------
def load_persuasion_data(file_path):
    print(f"\nLoading Data from {file_path}")
    start_time = time.time()

    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # 读取Excel文件
    df = pd.read_excel(file_path)

    # 检查Sentence列
    if "Sentence" not in df.columns:
        raise ValueError(f"'Sentence' column not found in Excel! Columns available: {list(df.columns)}")

    # 过滤有效文本（非空 + 去除空格后非空字符串）
    df["Sentence_str"] = df["Sentence"].astype(str).apply(lambda x: x.strip())
    valid_mask = df["Sentence"].notna() & (df["Sentence_str"] != "")
    raw_df = df[valid_mask].copy().drop(columns = ["Sentence_str"])
    texts = raw_df["Sentence"].astype(str).apply(lambda x: x.strip()).tolist()

    load_time = time.time() - start_time
    print(
        f"Loaded Data: {len(texts)} valid utterances (Total rows in Excel: {len(df)}, Invalid rows filtered: {len(df) - len(texts)})")
    print(f"Time Cost: {load_time:.2f}s")
    print(f"Raw Columns Preserved: {list(raw_df.columns)}")

    return texts, raw_df


# -------------------------- 4. 批量预测DA标签 --------------------------
def batch_predict_da(model, tokenizer, label_encoder, texts, batch_size=64):
    all_pred_labels = []
    model.eval()

    print(f"\nStarting DA Label Prediction (Batch Size: {batch_size})")
    start_time = time.time()

    # 分批次处理
    for i in tqdm(range(0, len(texts), batch_size), desc = "Predicting Batches"):
        batch_texts = texts[i:i + batch_size]

        # 文本编码
        encodings = tokenizer(
            batch_texts,
            truncation = True,
            padding = "max_length",
            max_length = MAX_SEQ_LENGTH,
            return_tensors = "pt"
        )

        input_ids = encodings["input_ids"].to(DEVICE)
        attention_mask = encodings["attention_mask"].to(DEVICE)

        # 预测
        with torch.no_grad():
            outputs = model(input_ids = input_ids, attention_mask = attention_mask)
            pred_ids = torch.argmax(outputs.logits, dim = -1).cpu().numpy()

        # 解码标签
        batch_labels = label_encoder.inverse_transform(pred_ids)
        all_pred_labels.extend(batch_labels)

    pred_time = time.time() - start_time
    avg_time_per_text = pred_time / len(texts) * 1000
    print(f"Prediction Completed!")
    print(f"Total Time: {pred_time:.2f}s | Avg Time per Utterance: {avg_time_per_text:.2f}ms")
    print(f"Total Predicted Labels: {len(all_pred_labels)}")

    return all_pred_labels


# -------------------------- 5. 保存结果 --------------------------
def save_labeled_results(raw_df, pred_labels, output_path):
    print(f"\nSaving Labeled Results to {output_path}")

    # 确保预测标签数量与数据行数一致
    if len(raw_df) != len(pred_labels):
        raise ValueError(
            f"Length mismatch: raw_df has {len(raw_df)} rows, but pred_labels has {len(pred_labels)} labels")

    # 新增DA标签列
    result_df = raw_df.copy()
    result_df[f"DA_label_{LABEL_TYPE}"] = pred_labels

    # 保存为CSV
    result_df.to_csv(output_path, index = False, encoding = "utf-8-sig")

    # 输出统计信息
    print(f"Results Saved Successfully!")
    print(f"\nResult Sample (First 5 Rows):")
    print(result_df[["Sentence", f"DA_label_{LABEL_TYPE}"]].head())

    print(f"\nDA Label Distribution:")
    label_count = result_df[f"DA_label_{LABEL_TYPE}"].value_counts()
    for label, count in label_count.items():
        percentage = count / len(result_df) * 100
        print(f"  {label}: {count} rows ({percentage:.2f}%)")


# -------------------------- 主函数 --------------------------
def main():
    print("=" * 60)
    print("Dialogue Act Prediction Script")
    print("=" * 60)

    # 检查数据目录
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok = True)
        print(f"Created data directory: {DATA_DIR}")

    # 检查数据文件
    if not os.path.exists(NEW_DATA_PATH):
        raise FileNotFoundError(
            f"Data file not found: {NEW_DATA_PATH}\nPlease put the Excel file in the correct location!")

    # 检查模型目录
    print(f"\nChecking model directory: {MODEL_SAVE_DIR}")
    if not os.path.exists(MODEL_SAVE_DIR):
        raise FileNotFoundError(f"Model directory not found: {MODEL_SAVE_DIR}\nExpected model files in this directory!")

    # 列出模型目录中的文件（用于调试）
    print(f"Files in model directory:")
    for file in os.listdir(MODEL_SAVE_DIR):
        print(f"  - {file}")

    # 加载模型
    model, tokenizer, label_encoder = load_trained_model()

    # 加载数据
    texts, raw_df = load_persuasion_data(NEW_DATA_PATH)

    # 如果没有有效文本，退出
    if len(texts) == 0:
        print("No valid texts found in the input file!")
        return

    # 预测标签
    pred_labels = batch_predict_da(model, tokenizer, label_encoder, texts)

    # 保存结果
    save_labeled_results(raw_df, pred_labels, OUTPUT_PATH)

    print(f"\n{'=' * 60}")
    print(f"All Tasks Completed Successfully!")
    print(f"{'=' * 60}")
    print(f"Input File: {NEW_DATA_PATH}")
    print(f"Output File: {OUTPUT_PATH}")
    print(f"DA Label Level: {LABEL_TYPE}")
    print(f"Model Used: {MODEL_SAVE_DIR}")
    print(f"{'=' * 60}")


# -------------------------- 运行入口 --------------------------
if __name__ == "__main__":
    main()