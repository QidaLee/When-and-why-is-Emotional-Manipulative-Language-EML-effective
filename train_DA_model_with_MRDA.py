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

# -------------------------- 1. Global Config --------------------------
DATA_DIR = "./data/MRDA"
SEED = 42
USE_SAVED_MODEL = False
BATCH_SIZE = 1000

MODEL_TYPE = "distilbert-base-uncased"
MAX_SEQ_LENGTH = 128  # Increased for longer merged utterances
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 2

# Paths
MODEL_BASE_DIR = "./models"
MODEL_SAVE_DIR = os.path.join(MODEL_BASE_DIR, f"{MODEL_TYPE.replace('/', '_')}_DA_MULTILABEL_MERGED")
TOKENIZER_SAVE_PATH = MODEL_SAVE_DIR
MLB_PATH = os.path.join(MODEL_SAVE_DIR, "mlb.pkl")
os.makedirs(MODEL_SAVE_DIR, exist_ok = True)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# NLTK
nltk.download('stopwords', quiet = True)
nltk.download('punkt_tab', quiet = True)


# -------------------------- 2. Load New Merged 3-Column Dataset --------------------------
def load_merged_multi_label_data(file_path):
    """
    Load NEW merged dataset:
    Format: speaker | merged_text | da_labels (comma separated)
    Returns: texts, list_of_label_lists
    """
    texts = []
    multi_labels = []

    print(f"Loading merged multi-label file: {os.path.basename(file_path)}")

    with open(file_path, "r", encoding = "utf-8") as f:
        for i, line in enumerate(tqdm(f, desc = "Loading")):
            line = line.strip()
            if not line:
                continue

            parts = line.split("|")
            if len(parts) < 3:
                continue

            speaker = parts[0].strip()
            text = parts[1].strip()
            label_str = parts[2].strip()

            # Split comma-separated labels into a list
            labels = [l.strip() for l in label_str.split(",") if l.strip()]

            if text and len(labels) > 0:
                texts.append(text)
                multi_labels.append(labels)

    print(f"Loaded {len(texts)} multi-label samples")
    return texts, multi_labels


# -------------------------- 3. Multi-Label Dataset --------------------------
class MultiLabelMRDADataset(Dataset):
    def __init__(self, texts, label_lists, tokenizer, max_seq_length, mlb):
        self.texts = texts
        self.multi_labels = mlb.transform(label_lists)  # Convert to one-hot
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation = True,
            padding = "max_length",
            max_length = self.max_seq_length,
            return_tensors = "pt"
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        labels = torch.tensor(self.multi_labels[idx], dtype = torch.float32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# -------------------------- 4. Multi-Label Model Wrapper --------------------------
class MultiLabelModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels = num_labels
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids = input_ids, attention_mask = attention_mask)
        logits = outputs.logits

        if labels is not None:
            # Use BCEWithLogitsLoss for multi-label classification
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            return loss, logits

        return logits


# -------------------------- 5. Multi-Label Metrics --------------------------
def compute_multi_label_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
    f1_micro = f1_score(labels, predictions, average = "micro", zero_division = 0)
    return {"micro_f1": f1_micro}


# -------------------------- 6. Training & Prediction --------------------------
def main():
    # Load NEW merged datasets
    train_texts, train_labels = load_merged_multi_label_data(os.path.join(DATA_DIR, "train_set_merged.txt"))
    val_texts, val_labels = load_merged_multi_label_data(os.path.join(DATA_DIR, "val_set_merged.txt"))
    test_texts, test_labels = load_merged_multi_label_data(os.path.join(DATA_DIR, "test_set_merged.txt"))

    # MultiLabelBinarizer for one-hot encoding
    mlb = MultiLabelBinarizer()
    mlb.fit(train_labels + val_labels + test_labels)
    joblib.dump(mlb, MLB_PATH)
    num_labels = len(mlb.classes_)
    print(f"Number of DA labels (multi-label): {num_labels}")
    print(f"Labels: {list(mlb.classes_)}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE)

    # Create datasets
    train_dataset = MultiLabelMRDADataset(train_texts, train_labels, tokenizer, MAX_SEQ_LENGTH, mlb)
    val_dataset = MultiLabelMRDADataset(val_texts, val_labels, tokenizer, MAX_SEQ_LENGTH, mlb)
    test_dataset = MultiLabelMRDADataset(test_texts, test_labels, tokenizer, MAX_SEQ_LENGTH, mlb)

    # Initialize multi-label model
    model = MultiLabelModel(MODEL_TYPE, num_labels)
    model.to(DEVICE)

    # Training args
    training_args = TrainingArguments(
        output_dir = MODEL_SAVE_DIR,
        num_train_epochs = NUM_EPOCHS,
        per_device_train_batch_size = TRAIN_BATCH_SIZE,
        per_device_eval_batch_size = EVAL_BATCH_SIZE,
        learning_rate = LEARNING_RATE,
        warmup_ratio = WARMUP_RATIO,
        weight_decay = WEIGHT_DECAY,
        logging_steps = 100,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        load_best_model_at_end = True,
        metric_for_best_model = "micro_f1",
        greater_is_better = True,
        fp16 = torch.cuda.is_available(),
        seed = SEED,
        report_to = "none",
        dataloader_num_workers = 2
    )

    # Trainer
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        compute_metrics = compute_multi_label_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience = EARLY_STOPPING_PATIENCE)]
    )

    # Train
    print("\nStarting multi-label DA training...")
    trainer.train()

    # Save
    model.model.save_pretrained(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)
    print("Model saved.")

    # ---------------- Evaluation ----------------
    print("\nEvaluating on test set...")
    test_pred = trainer.predict(test_dataset)
    logits = test_pred.predictions
    predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
    true_labels = mlb.transform(test_labels)

    print("\n===== Multi-Label Test Results =====")
    print("Classification Report:")
    print(classification_report(true_labels, predictions, target_names = mlb.classes_, zero_division = 0))

    # ---------------- Multi-Label Prediction Function ----------------
    def predict_multi_da(text):
        encoding = tokenizer(
            text, truncation = True, padding = "max_length",
            max_length = MAX_SEQ_LENGTH, return_tensors = "pt"
        ).to(DEVICE)

        model.eval()
        with torch.no_grad():
            logits = model(**encoding)

        probs = torch.sigmoid(logits).cpu().numpy()[0]
        pred_indices = np.where(probs > 0.5)[0]
        if len(pred_indices) == 0:
            pred_indices = [np.argmax(probs)]
        return mlb.classes_[pred_indices].tolist()

    # Demo
    print("\n===== Multi-Label Prediction Examples =====")
    test_texts_demo = [
        "Can you share the project documents and deadline?",
        "I think we should finish this today and I agree with the plan",
        "I don't accept this and please send me the update"
    ]

    for t in test_texts_demo:
        preds = predict_multi_da(t)
        print(f"Text: {t}")
        print(f"Predicted DA labels: {preds}\n")


if __name__ == "__main__":
    main()
