import os
import re
import time
import numpy as np
import pandas as pd
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm
import joblib

# -------------------------- 1. 全局配置与初始化 --------------------------
DATA_DIR = "./data/MRDA"
LABEL_TYPE = "Our_label"  # Basic/General/Full/Our_label
SEED = 42
USE_SAVED_MODEL = False
TEST_SMALL_BATCH = False
BATCH_SIZE = 1000
# 优先用轻量化模型，适配集群CPU/GPU环境
MODEL_TYPE = "distilbert-base-uncased"
# MODEL_TYPE: distilbert-base-uncased/roberta-base/bert-base-uncased/roberta-large/bert-large-uncased/albert-base-v2

# 训练配置（GPU环境下增大批次）
MAX_SEQ_LENGTH = 64
TRAIN_BATCH_SIZE = 32  # GPU环境建议32，显存不足可改为16
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 2

# 路径配置
MODEL_BASE_DIR = "./models"
MODEL_SAVE_DIR = os.path.join(MODEL_BASE_DIR, f"{MODEL_TYPE.replace('/', '_')}_DA_MRDA_{LABEL_TYPE.lower()}")
TOKENIZER_SAVE_PATH = MODEL_SAVE_DIR  # tokenizer和模型同目录，简化加载
LABEL_ENCODER_PATH = os.path.join(MODEL_SAVE_DIR, "label_encoder.pkl")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 设备配置（自动检测GPU/CPU）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# NLTK初始化
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# -------------------------- 2. 数据加载（兼容原有逻辑） --------------------------
def load_mrda_data(file_path, label_type="Basic"):
    texts = []
    labels = []

    # 根据你的数据格式（6列）调整索引
    # 列映射: 0=id, 1=text, 2=Basic, 3=General, 4=Full, 5=Our_label
    label_idx = {"Basic": 2, "General": 3, "Full": 4, "Our_label": 5}[label_type]

    print(f"Starting to load {os.path.basename(file_path)}...")
    print(f"Label type: {label_type}, Column index: {label_idx}")

    start_time = time.time()
    sample_count = 0
    error_count = 0

    with open(file_path, "r", encoding = "utf-8") as f:
        for i, line in enumerate(tqdm(f, desc = f"Loading {os.path.basename(file_path)}")):
            line = line.strip()
            if not line:
                continue

            parts = line.split("|")

            # 显示前几行用于验证
            if i < 5:
                print(f"Debug - Line {i}: columns={len(parts)}")
                for j, part in enumerate(parts):
                    print(f"  Column {j}: '{part}'")

            # 检查是否有足够的列
            if len(parts) < 6:
                print(f"Warning: Line {i} has only {len(parts)} columns, skipping: {line[:100]}")
                error_count += 1
                continue

            try:
                text = parts[1].strip()  # 第2列是文本
                label = parts[label_idx].strip()  # 根据标签类型选择列

                if not text or not label:
                    print(f"Warning: Empty text or label at line {i}: {line[:100]}")
                    error_count += 1
                    continue

                texts.append(text)
                labels.append(label)
                sample_count += 1

            except Exception as e:
                print(f"Error parsing line {i}: {e}")
                print(f"Line content: {line}")
                error_count += 1
                continue

    load_time = time.time() - start_time

    print(f"\n{'=' * 50}")
    print(f"File: {os.path.basename(file_path)}")
    print(f"Loaded {sample_count} samples successfully")
    print(f"Errors encountered: {error_count}")
    print(f"Loading time: {load_time:.2f}s")
    print(f"Label type: {label_type}")

    if sample_count > 0:
        # 显示标签统计
        label_counter = Counter(labels)
        print(f"\nLabel distribution:")
        for label, count in label_counter.most_common():
            print(f"  {label}: {count}")

        # 显示标签类型示例
        print(f"\nSample labels:")
        for i, (text, label) in enumerate(zip(texts[:5], labels[:5])):
            print(f"  {i + 1}. '{text}' -> {label}")
    else:
        print("ERROR: No valid samples loaded!")
        print("Please check:")
        print("1. File format (should be pipe-separated)")
        print("2. Column count (should be 6)")
        print("3. Label type selection (Our_label should be column 5)")

    return texts, labels

# -------------------------- 3. 数据预处理与数据集类 --------------------------
class MRDADataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_seq_length, label_encoder):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        label = torch.tensor(self.label_encoder.transform([self.labels[idx]])[0], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label
        }

# -------------------------- 4. 模型加载/初始化（核心修复处） --------------------------
def load_saved_model():
    try:
        print(f"Loading saved model from {MODEL_SAVE_DIR}...")
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_SAVE_DIR,
            num_labels=len(label_encoder.classes_)
        )
        model.to(DEVICE)
        print("Model loaded successfully!")
        return model, tokenizer, label_encoder
    except Exception as e:
        print(f"Failed to load saved model: {e}")
        print("Will train new model...")
        return None, None, None

# 【核心修复】修复id2label和label2id的初始化逻辑
def init_model(label_encoder):
    """初始化预训练模型，接收label_encoder作为参数（包含标签列表）"""
    num_labels = len(label_encoder.classes_)
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE)
    # 正确构建id2label和label2id（基于标签编码器的类别列表）
    id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
    label2id = {label: i for i, label in enumerate(label_encoder.classes_)}
    # 加载配置和模型
    config = AutoConfig.from_pretrained(
        MODEL_TYPE,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_TYPE,
        config=config
    )
    model.to(DEVICE)
    return model, tokenizer

# -------------------------- 5. 评估指标函数 --------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}

# -------------------------- 6. 核心训练/预测逻辑（修复调用处） --------------------------
def main():
    # 1. 加载数据
    train_texts, train_labels = load_mrda_data(os.path.join(DATA_DIR, "train_set.txt"), LABEL_TYPE)
    val_texts, val_labels = load_mrda_data(os.path.join(DATA_DIR, "val_set.txt"), LABEL_TYPE)
    test_texts, test_labels = load_mrda_data(os.path.join(DATA_DIR, "test_set.txt"), LABEL_TYPE)

    # 2. 尝试加载保存的模型
    model, tokenizer, label_encoder = load_saved_model() if USE_SAVED_MODEL else (None, None, None)

    # 3. 训练新模型
    if model is None or tokenizer is None:
        # 先初始化标签编码器（全局拟合，避免数据泄露）
        label_encoder = LabelEncoder()
        label_encoder.fit(train_labels + val_labels + test_labels)
        # 【核心修复】传入label_encoder，而非num_labels
        model, tokenizer = init_model(label_encoder)
        # 保存标签编码器
        joblib.dump(label_encoder, LABEL_ENCODER_PATH)

        # 创建数据集
        train_dataset = MRDADataset(train_texts, train_labels, tokenizer, MAX_SEQ_LENGTH, label_encoder)
        val_dataset = MRDADataset(val_texts, val_labels, tokenizer, MAX_SEQ_LENGTH, label_encoder)

        # 训练配置（GPU环境开启fp16）
        training_args = TrainingArguments(
            output_dir=MODEL_SAVE_DIR,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            warmup_ratio=WARMUP_RATIO,
            weight_decay=WEIGHT_DECAY,
            logging_dir=os.path.join(MODEL_SAVE_DIR, "logs"),
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),  # GPU时启用混合精度加速
            seed=SEED,
            disable_tqdm=False,
            report_to="none",
            # GPU训练加速：启用多线程数据加载
            dataloader_num_workers=4
        )

        # 初始化Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
        )

        # 开始训练
        print("\nStarting model training...")
        start_time = time.time()
        trainer.train()
        train_time = time.time() - start_time
        print(f"Training completed (Time: {train_time:.2f}s)")

        # 保存模型和tokenizer
        print(f"\nSaving model to {MODEL_SAVE_DIR}...")
        trainer.save_model(MODEL_SAVE_DIR)
        tokenizer.save_pretrained(MODEL_SAVE_DIR)
        print("Model saved successfully!")

    # 4. 模型评估
    print("\nStarting test set evaluation...")
    start_time = time.time()
    test_dataset = MRDADataset(test_texts, test_labels, tokenizer, MAX_SEQ_LENGTH, label_encoder)
    trainer = Trainer(model=model)
    test_predictions = trainer.predict(test_dataset)
    test_preds = np.argmax(test_predictions.predictions, axis=-1)
    test_true = label_encoder.transform(test_labels)

    # 解码标签并生成报告
    test_preds_labels = label_encoder.inverse_transform(test_preds)
    test_true_labels = label_encoder.inverse_transform(test_true)

    eval_time = time.time() - start_time
    print(f"\n===== Test Set Evaluation Results (Time: {eval_time:.2f}s) =====")
    print(f"Overall Accuracy: {accuracy_score(test_true_labels, test_preds_labels):.4f}")
    print("\nClassification Report:")
    print(classification_report(test_true_labels, test_preds_labels, zero_division=0))

    # 5. 预测函数
    def predict_dialogue_act(text):
        encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)

        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred_id = torch.argmax(outputs.logits, dim=-1).cpu().numpy()[0]

        return label_encoder.inverse_transform([pred_id])[0]

    # 示例预测
    test_utterances = [
        "Can you share the project docs?",
        "The deadline is tomorrow.",
        "Let's move to the next topic."
    ]
    print("\n===== Example Predictions =====")
    for utterance in test_utterances:
        predicted_label = predict_dialogue_act(utterance)
        print(f"Utterance: {utterance}\nPredicted DA Label: {predicted_label}\n")

if __name__ == "__main__":
    main()