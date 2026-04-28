import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
import joblib
import pandas as pd

# ====================== GLOBAL CONFIG ======================
DATA_DIR = "./data/MRDA"
MANUAL_DATA_PATH = "./data/Persuasion_For_Good/100_sample_turns_data_with_manual_label.xlsx"

MODEL_TYPE = "roberta-base"
MAX_SEQ_LENGTH = 128
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
SEED = 42

# Paths
BASE_MODEL_DIR = "./models/roberta_BASE_MRDA"
FINETUNED_MODEL_DIR = "./models/roberta_FINETUNED_PERSUASION"
MLB_BASE_PATH = os.path.join(BASE_MODEL_DIR, "mlb.pkl")

os.makedirs(BASE_MODEL_DIR, exist_ok=True)
os.makedirs(FINETUNED_MODEL_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ====================== DATA LOADER ======================
def load_merged_multi_label_data(file_path):
    texts, labels = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split("|")
            if len(parts) <3: continue
            text = parts[1].strip()
            label_str = parts[2].strip()
            labs = [l.strip() for l in label_str.split(",") if l.strip()]
            if text and labs:
                texts.append(text)
                labels.append(labs)
    return texts, labels

def load_manual_data(file_path):
    df = pd.read_excel(file_path)
    def parse(s):
        if pd.isna(s): return []
        return [x.strip() for x in str(s).split(",")]
    df["labels"] = df["DA_label"].apply(parse)
    df = df[df["labels"].str.len()>0]
    texts = df["Sentence"].astype(str).str.strip().tolist()
    labels = df["labels"].tolist()
    return texts, labels

# ====================== DATASET ======================
class MultiLabelDataset(Dataset):
    def __init__(self, texts, label_lists, tokenizer, max_len, mlb):
        self.texts = texts
        self.labels = mlb.transform(label_lists)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32)
        }

# ====================== MODEL ======================
class MultiLabelModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            return loss, logits
        return logits

# ====================== METRICS ======================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.35).int().numpy()
    return {"micro_f1": f1_score(labels, preds, average="micro", zero_division=0)}

# ====================== STAGE 1: TRAIN ON MRDA ======================
def stage1_train_mrdA():
    print("\n===== STAGE 1: TRAIN ON MRDA ONLY =====")

    tr_texts, tr_labels = load_merged_multi_label_data(os.path.join(DATA_DIR, "train_set_merged.txt"))
    val_texts, val_labels = load_merged_multi_label_data(os.path.join(DATA_DIR, "val_set_merged.txt"))

    mlb = MultiLabelBinarizer()
    mlb.fit(tr_labels + val_labels)
    joblib.dump(mlb, MLB_BASE_PATH)
    num_labels = len(mlb.classes_)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE)
    train_ds = MultiLabelDataset(tr_texts, tr_labels, tokenizer, MAX_SEQ_LENGTH, mlb)
    val_ds = MultiLabelDataset(val_texts, val_labels, tokenizer, MAX_SEQ_LENGTH, mlb)

    model = MultiLabelModel(MODEL_TYPE, num_labels)
    model.to(DEVICE)

    args = TrainingArguments(
        output_dir=BASE_MODEL_DIR,
        num_train_epochs=4,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=2e-5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        seed=SEED,
        report_to="none"
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics, callbacks=[EarlyStoppingCallback(2)]
    )
    trainer.train()
    model.model.save_pretrained(BASE_MODEL_DIR)
    tokenizer.save_pretrained(BASE_MODEL_DIR)

# ====================== STAGE 2: FINE-TUNE ON MANUAL DATA ======================
def stage2_finetune_manual():
    print("\n===== STAGE 2: FINE-TUNE ON MANUAL LABELS =====")

    mlb = joblib.load(MLB_BASE_PATH)
    num_labels = len(mlb.classes_)

    # Load manual data
    tr_texts, tr_labels = load_manual_data(MANUAL_DATA_PATH)
    # Use 80% train, 20% val
    split = int(len(tr_texts)*0.8)
    tr_texts, val_texts = tr_texts[:split], tr_texts[split:]
    tr_labels, val_labels = tr_labels[:split], tr_labels[split:]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
    train_ds = MultiLabelDataset(tr_texts, tr_labels, tokenizer, MAX_SEQ_LENGTH, mlb)
    val_ds = MultiLabelDataset(val_texts, val_labels, tokenizer, MAX_SEQ_LENGTH, mlb)

    # Load BASE model
    model = MultiLabelModel(BASE_MODEL_DIR, num_labels)
    model.to(DEVICE)

    # Very small lr for fine-tune
    args = TrainingArguments(
        output_dir=FINETUNED_MODEL_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        learning_rate=3e-6,  # VERY SMALL
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        seed=SEED,
        report_to="none"
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics, callbacks=[EarlyStoppingCallback(1)]
    )
    trainer.train()
    model.model.save_pretrained(FINETUNED_MODEL_DIR)
    tokenizer.save_pretrained(FINETUNED_MODEL_DIR)

# ====================== MAIN ======================
if __name__ == "__main__":
    stage1_train_mrdA()
    stage2_finetune_manual()
    print("\nAll done! Final model at:", FINETUNED_MODEL_DIR)