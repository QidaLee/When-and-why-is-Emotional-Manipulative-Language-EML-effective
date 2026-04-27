import os
import re
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score, f1_score
import nltk
from tqdm import tqdm
import joblib
import pandas as pd

# -------------------------- 1. Global Config --------------------------
DATA_DIR = "./data/MRDA"
MANUAL_DATA_PATH = "./data/Persuasion_For_Good/100_sample_turns_data_with_manual_label.xlsx"
SEED = 42
USE_SAVED_MODEL = False

MODEL_TYPE = "roberta-base"
MAX_SEQ_LENGTH = 128
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 1e-5
NUM_EPOCHS = 4
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 2

# Paths
MODEL_BASE_DIR = "./models"
MODEL_SAVE_DIR = os.path.join(MODEL_BASE_DIR, f"{MODEL_TYPE.replace('/', '_')}_DA_MULTILABEL_FINETUNED")
MLB_PATH = os.path.join(MODEL_SAVE_DIR, "mlb.pkl")
os.makedirs(MODEL_SAVE_DIR, exist_ok = True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# -------------------------- 2. Load Merged MRDA Data --------------------------
def load_merged_multi_label_data(file_path):
    texts = []
    multi_labels = []
    with open(file_path, "r", encoding = "utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split("|")
            if len(parts) < 3: continue
            text = parts[1].strip()
            label_str = parts[2].strip()
            labels = [l.strip() for l in label_str.split(",") if l.strip()]
            if text and len(labels) > 0:
                texts.append(text)
                multi_labels.append(labels)
    return texts, multi_labels


# -------------------------- 3. Load Manual Label Data (50% for fine-tune) --------------------------
def load_manual_finetune_data(file_path, val_ratio=0.5):
    df = pd.read_excel(file_path)

    def parse_labels(s):
        if pd.isna(s): return []
        return [x.strip() for x in str(s).split(",")]

    df["labels"] = df["DA_label"].apply(parse_labels)
    df = df[df["labels"].str.len() > 0]

    texts = df["Sentence"].astype(str).str.strip().tolist()
    labels = df["labels"].tolist()

    split_idx = int(len(texts) * (1 - val_ratio))
    return texts[:split_idx], labels[:split_idx]


# -------------------------- 4. Multi-Label Dataset --------------------------
class MultiLabelMRDADataset(Dataset):
    def __init__(self, texts, label_lists, tokenizer, max_seq_length, mlb):
        self.texts = texts
        self.multi_labels = mlb.transform(label_lists)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text, truncation = True, padding = "max_length",
            max_length = self.max_seq_length, return_tensors = "pt"
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        labels = torch.tensor(self.multi_labels[idx], dtype = torch.float32)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# -------------------------- 5. Clean Multi-Label Model --------------------------
class MultiLabelModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids = input_ids, attention_mask = attention_mask)
        logits = outputs.logits
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            return loss, logits
        return logits


# -------------------------- 6. Metrics --------------------------
def compute_multi_label_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = (torch.sigmoid(torch.tensor(logits)) > 0.35).int().numpy()
    f1_micro = f1_score(labels, predictions, average = "micro", zero_division = 0)
    return {"micro_f1": f1_micro}


# -------------------------- 7. Training --------------------------
def main():
    # === Load MRDA data ===
    train_texts, train_labels = load_merged_multi_label_data(os.path.join(DATA_DIR, "train_set_merged.txt"))
    val_texts, val_labels = load_merged_multi_label_data(os.path.join(DATA_DIR, "val_set_merged.txt"))
    test_texts, test_labels = load_merged_multi_label_data(os.path.join(DATA_DIR, "test_set_merged.txt"))

    # === Load 50% manual data for fine-tuning ===
    manual_texts, manual_labels = load_manual_finetune_data(MANUAL_DATA_PATH, val_ratio = 0.5)
    print(f"Manual fine-tune samples: {len(manual_texts)}")

    # === Combine ===
    combined_train_texts = train_texts + manual_texts
    combined_train_labels = train_labels + manual_labels
    print(f"Total training samples: {len(combined_train_texts)}")

    # === Label Encoder ===
    mlb = MultiLabelBinarizer()
    mlb.fit(combined_train_labels + val_labels + test_labels)
    joblib.dump(mlb, MLB_PATH)
    num_labels = len(mlb.classes_)
    print(f"Labels: {list(mlb.classes_)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE)

    # === Datasets ===
    train_dataset = MultiLabelMRDADataset(combined_train_texts, combined_train_labels, tokenizer, MAX_SEQ_LENGTH, mlb)
    val_dataset = MultiLabelMRDADataset(val_texts, val_labels, tokenizer, MAX_SEQ_LENGTH, mlb)
    test_dataset = MultiLabelMRDADataset(test_texts, test_labels, tokenizer, MAX_SEQ_LENGTH, mlb)

    # === Clean Model (NO class weights → NO device error) ===
    model = MultiLabelModel(MODEL_TYPE, num_labels)
    model.to(DEVICE)

    # === Training Args ===
    training_args = TrainingArguments(
        output_dir = MODEL_SAVE_DIR,
        num_train_epochs = NUM_EPOCHS,
        per_device_train_batch_size = TRAIN_BATCH_SIZE,
        per_device_eval_batch_size = EVAL_BATCH_SIZE,
        learning_rate = LEARNING_RATE,
        warmup_ratio = WARMUP_RATIO,
        weight_decay = WEIGHT_DECAY,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        load_best_model_at_end = True,
        metric_for_best_model = "micro_f1",
        greater_is_better = True,
        fp16 = torch.cuda.is_available(),
        seed = SEED,
        report_to = "none",
        dataloader_num_workers = 0,
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        compute_metrics = compute_multi_label_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience = EARLY_STOPPING_PATIENCE)]
    )

    trainer.train()

    # Save
    model.model.save_pretrained(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)

    # === Test Evaluation ===
    print("\n===== Test Results =====")
    test_pred = trainer.predict(test_dataset)
    logits = test_pred.predictions
    predictions = (torch.sigmoid(torch.tensor(logits)) > 0.35).cpu().numpy()
    true_labels = mlb.transform(test_labels)
    print(classification_report(true_labels, predictions, target_names = mlb.classes_, zero_division = 0))


if __name__ == "__main__":
    main()
