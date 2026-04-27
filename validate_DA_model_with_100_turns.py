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
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

warnings.filterwarnings('ignore')

# -------------------------- 1. Global Config --------------------------
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
THRESHOLD = 0.3

# Data config
DATA_DIR = "data/Persuasion_For_Good"
TEST_DATA_PATH = os.path.join(DATA_DIR, "100_sample_turns_data_with_manual_label.xlsx")
TRUE_LABEL_COL = "DA_label"  # Your ground truth column

# Output
OUTPUT_DIR = "output_data"
os.makedirs(OUTPUT_DIR, exist_ok = True)
PRED_OUTPUT_CSV = os.path.join(OUTPUT_DIR, "multi_da_validation_results.csv")
REPORT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "multi_label_classification_report.csv")

# Model path (your trained multi-label model)
MODEL_PATH = "./models/distilbert-base-uncased_DA_MULTILABEL_MERGED"
MLB_PATH = os.path.join(MODEL_PATH, "mlb.pkl")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# -------------------------- 2. Load Multi-Label Model --------------------------
def load_multi_label_model():
    print("\nLoading multi-label DA model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()

    mlb = joblib.load(MLB_PATH)
    print(f"Label set: {list(mlb.classes_)}")
    return tokenizer, model, mlb


# -------------------------- 3. Load & Parse Multi-Label Test Data --------------------------
def load_test_data(file_path, true_col):
    print(f"\nLoading test data: {file_path}")
    df = pd.read_excel(file_path)

    # Parse multi-label: "pr, in" → ["pr", "in"]
    def parse_labels(s):
        if pd.isna(s) or s.strip() == "":
            return []
        return [x.strip() for x in str(s).split(",")]

    df["true_labels"] = df[true_col].apply(parse_labels)
    df = df[df["true_labels"].apply(len) > 0].copy()

    texts = df["Sentence"].astype(str).str.strip().tolist()
    true_labels = df["true_labels"].tolist()

    print(f"Valid test samples: {len(texts)}")
    return texts, true_labels, df


# -------------------------- 4. Multi-Label Prediction --------------------------
def predict_multi_labels(texts, tokenizer, model, mlb):
    all_preds = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc = "Predicting"):
            batch = texts[i:i + BATCH_SIZE]

            inputs = tokenizer(
                batch,
                padding = True,
                truncation = True,
                max_length = MAX_SEQ_LENGTH,
                return_tensors = "pt"
            ).to(DEVICE)

            logits = model(**inputs).logits
            probs = torch.sigmoid(logits)
            preds = (probs > THRESHOLD).cpu().numpy()

            for pred in preds:
                labels = list(mlb.classes_[np.where(pred)[0]])
                if not labels:
                    labels = [mlb.classes_[np.argmax(probs.cpu().numpy()[0])]]
                all_preds.append(labels)

    return all_preds


# -------------------------- 5. Multi-Label Evaluation --------------------------
def evaluate_multi_label(true_labels, pred_labels, mlb):
    y_true = mlb.transform(true_labels)
    y_pred = mlb.transform(pred_labels)

    micro_f1 = f1_score(y_true, y_pred, average = "micro", zero_division = 0)
    macro_f1 = f1_score(y_true, y_pred, average = "macro", zero_division = 0)

    print("\n===== Multi-Label Validation Results =====")
    print(f"Micro F1: {micro_f1:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nClassification Report:")
    report = classification_report(
        y_true, y_pred,
        target_names = mlb.classes_,
        zero_division = 0
    )
    print(report)

    return micro_f1, macro_f1


# -------------------------- 6. Save Results --------------------------
def save_results(df, pred_labels):
    df["pred_DA_labels"] = [",".join(lst) for lst in pred_labels]
    df["true_DA_labels"] = [",".join(lst) for lst in df["true_labels"]]
    df.to_csv(PRED_OUTPUT_CSV, index = False, encoding = "utf-8-sig")
    print(f"\nResults saved to: {PRED_OUTPUT_CSV}")


# -------------------------- Main --------------------------
def main():
    tokenizer, model, mlb = load_multi_label_model()
    texts, true_labels, df = load_test_data(TEST_DATA_PATH, TRUE_LABEL_COL)
    pred_labels = predict_multi_labels(texts, tokenizer, model, mlb)
    evaluate_multi_label(true_labels, pred_labels, mlb)
    save_results(df, pred_labels)


if __name__ == "__main__":
    main()
