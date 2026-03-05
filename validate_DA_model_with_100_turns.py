import os
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import joblib
from tqdm import tqdm

# -------------------------- 1. 全局配置（适配你的实际数据） --------------------------
MODEL_TYPE = "bert-base-uncased"
LABEL_TYPE = "General"
MAX_SEQ_LENGTH = 64

# 数据路径配置（修改为你的实际路径）
DATA_DIR = "data/Persuasion_For_Good"
NEW_DATA_PATH = os.path.join(DATA_DIR, "100_sample_turns_data_with_manual_label.xlsx")

# 输出路径配置（改为output_data文件夹）
OUTPUT_DIR = "output_data"
os.makedirs(OUTPUT_DIR, exist_ok = True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "100_sample_with_DA.csv")

# 模型存储路径（你确认模型文件已存在）
MODEL_BASE_DIR = "./models"
MODEL_SAVE_DIR = os.path.join(MODEL_BASE_DIR, f"{MODEL_TYPE.replace('/', '_')}_DA_MRDA_{LABEL_TYPE.lower()}")
LABEL_ENCODER_PATH = os.path.join(MODEL_SAVE_DIR, "label_encoder.pkl")

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device Info: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")


# -------------------------- 2. 加载训练好的模型 --------------------------
def load_trained_model():
    print(f"\nLoading Trained Model from {MODEL_SAVE_DIR}")

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
        print(f"Loaded Tokenizer (Model Type: {MODEL_TYPE})")
    except Exception as e:
        raise ValueError(f"Failed to load Tokenizer: {e}")

    # 加载模型
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_SAVE_DIR,
            num_labels = len(label_encoder.classes_)
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

    # 读取Excel文件
    df = pd.read_excel(file_path)

    # 检查Sentence列（替换原Unit列）
    if "Sentence" not in df.columns:
        raise ValueError(f"'Sentence' column not found in Excel! Columns available: {list(df.columns)}")

    # 过滤有效文本（非空 + 去除空格后非空字符串）
    df["Sentence_str"] = df["Sentence"].astype(str).apply(lambda x: x.strip())  # 对每个文本去空格
    valid_mask = df["Sentence"].notna() & (df["Sentence_str"] != "")  # 过滤条件
    raw_df = df[valid_mask].copy().drop(columns = ["Sentence_str"])  # 删除临时列
    texts = raw_df["Sentence"].astype(str).apply(lambda x: x.strip()).tolist()  # 文本去空格后提取

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

    # 新增DA标签列
    result_df = raw_df.copy()
    result_df[f"DA_label_{LABEL_TYPE}"] = pred_labels

    # 保存为CSV
    result_df.to_csv(output_path, index = False, encoding = "utf-8-sig")

    # 输出统计信息
    print(f"Results Saved Successfully!")
    print(f"\nResult Sample (First 5 Rows)")
    print(result_df[["Sentence", f"DA_label_{LABEL_TYPE}"]].head())

    print(f"\nDA Label Distribution")
    label_count = result_df[f"DA_label_{LABEL_TYPE}"].value_counts()
    for label, count in label_count.items():
        percentage = count / len(result_df) * 100
        print(f"{label}: {count} rows ({percentage:.2f}%)")


# -------------------------- 主函数 --------------------------
def main():
    # 检查数据目录
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok = True)
        print(f"Created data directory: {DATA_DIR}")
        raise FileNotFoundError(f"Please put '100_sample_turns_data_with_manual_label.xlsx' into {DATA_DIR} first!")

    # 检查模型目录
    if not os.path.exists(MODEL_SAVE_DIR):
        raise FileNotFoundError(f"Model directory not found: {MODEL_SAVE_DIR}\nPlease train the model first!")

    # 加载模型
    model, tokenizer, label_encoder = load_trained_model()

    # 加载数据
    texts, raw_df = load_persuasion_data(NEW_DATA_PATH)

    # 预测标签
    pred_labels = batch_predict_da(model, tokenizer, label_encoder, texts)

    # 保存结果
    save_labeled_results(raw_df, pred_labels, OUTPUT_PATH)

    print(f"\nAll Tasks Completed Successfully!")
    print(f"Input File: {NEW_DATA_PATH}")
    print(f"Output File: {OUTPUT_PATH}")
    print(f"DA Label Level: {LABEL_TYPE}")


# -------------------------- 运行入口 --------------------------
if __name__ == "__main__":
    main()