import os
import numpy as np
import pandas as pd
import torch
import warnings
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import joblib
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, cohen_kappa_score
from sklearn.preprocessing import MultiLabelBinarizer

warnings.filterwarnings('ignore')

# -------------------------- 1. Global Config --------------------------
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
THRESHOLD = 0.3

# Data
DATA_DIR = "data/Persuasion_For_Good"
TEST_DATA_PATH = os.path.join(DATA_DIR, "100_sample_turns_data_with_manual_label.xlsx")
TRUE_LABEL_COL = "DA_label"

# Output
OUTPUT_DIR = "output_data"
os.makedirs(OUTPUT_DIR, exist_ok = True)
PRED_OUTPUT_CSV = os.path.join(OUTPUT_DIR, "multi_da_validation_results.csv")
METRICS_CSV = os.path.join(OUTPUT_DIR, "agreement_metrics.csv")

# Model
MODEL_PATH = "./models/roberta_FINETUNED_PERSUASION"
MLB_PATH = "./models/roberta_BASE_MRDA/mlb.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# -------------------------- 2. Load Model --------------------------
def load_multi_label_model():
    print("\nLoading multi-label DA model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    mlb = joblib.load(MLB_PATH)
    print(f"Label set: {list(mlb.classes_)}")
    return tokenizer, model, mlb


# -------------------------- 3. Load Test Data --------------------------
def load_test_data(file_path, true_col):
    print(f"\nLoading test data: {file_path}")
    df = pd.read_excel(file_path)

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


# -------------------------- 4. Prediction --------------------------
def predict_multi_labels(texts, tokenizer, model, mlb):
    all_preds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc = "Predicting"):
            batch = texts[i:i + BATCH_SIZE]
            inputs = tokenizer(batch, padding = True, truncation = True,
                               max_length = MAX_SEQ_LENGTH, return_tensors = "pt").to(DEVICE)
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits)
            preds = (probs > THRESHOLD).cpu().numpy()
            for pred in preds:
                labels = list(mlb.classes_[np.where(pred)[0]])
                if not labels:
                    labels = [mlb.classes_[np.argmax(probs.cpu().numpy()[0])]]
                all_preds.append(labels)
    return all_preds


# -------------------------- 5. MULTI-LABEL KAPPA & PABAK (FIXED VERSION) --------------------------
def compute_multi_label_agreement(y_true_bin, y_pred_bin):
    # Flatten all labels (per-label evaluation, NOT exact match)
    y_true_flat = y_true_bin.ravel()
    y_pred_flat = y_pred_bin.ravel()

    # Cohen's Kappa (per label)
    cohen_k = cohen_kappa_score(y_true_flat, y_pred_flat)

    # Observed agreement
    p_o = np.mean(y_true_flat == y_pred_flat)

    # PABAK (for multi-label, no longer negative!)
    pabak = 2 * p_o - 1

    # Fleiss Kappa for 2 raters = Cohen's Kappa
    fleiss_k = cohen_k

    return {
        "label_accuracy": p_o,
        "cohen_kappa": cohen_k,
        "fleiss_kappa": fleiss_k,
        "pabak": pabak
    }


# -------------------------- 6. Full Evaluation --------------------------
def evaluate_multi_label(true_labels, pred_labels, mlb):
    y_true = mlb.transform(true_labels)
    y_pred = mlb.transform(pred_labels)

    micro_f1 = f1_score(y_true, y_pred, average = "micro", zero_division = 0)
    macro_f1 = f1_score(y_true, y_pred, average = "macro", zero_division = 0)

    agree = compute_multi_label_agreement(y_true, y_pred)

    print("\n" + "=" * 70)
    print("FINAL VALIDATION RESULTS (MODEL vs HUMAN)")
    print("=" * 70)
    print(f"Micro F1:              {micro_f1:.4f}")
    print(f"Macro F1:              {macro_f1:.4f}")
    print(f"Label-wise Accuracy:   {agree['label_accuracy']:.4f}")
    print(f"Cohen’s Kappa:         {agree['cohen_kappa']:.4f}")
    print(f"Fleiss’ Kappa:         {agree['fleiss_kappa']:.4f}")
    print(f"PABAK:                 {agree['pabak']:.4f}")
    print("=" * 70)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names = mlb.classes_, zero_division = 0))

    metrics_df = pd.DataFrame([{
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "label_accuracy": agree["label_accuracy"],
        "cohen_kappa": agree["cohen_kappa"],
        "fleiss_kappa": agree["fleiss_kappa"],
        "pabak": agree["pabak"]
    }])
    metrics_df.to_csv(METRICS_CSV, index = False)
    print(f"\nMetrics saved to: {METRICS_CSV}")

    return micro_f1, macro_f1, agree


# -------------------------- 7. Save Results --------------------------
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
