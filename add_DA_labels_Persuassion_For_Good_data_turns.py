import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from tqdm import tqdm
import joblib

# ====================== Configuration ======================
MODEL_PATH = "models/distilbert-base-uncased_DA_MULTILABEL_MERGED"
MLB_FILE_PATH = f"{MODEL_PATH}/mlb.pkl"
DATA_PATH = "data/Persuasion_For_Good/all_turns_data.xlsx"
OUTPUT_PATH = "data/Persuasion_For_Good/all_turns_data_with_multi_DA.xlsx"
PREDICTION_THRESHOLD = 0.5
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ====================== Load Model & Tokenizer ======================
def load_multi_label_pipeline(model_dir, mlb_path):
    """
    Load pre-trained multi-label DA model, tokenizer and MultiLabelBinarizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model = model.to(DEVICE)
    model.eval()

    # Load multi-label encoder
    mlb = joblib.load(mlb_path)
    return tokenizer, model, mlb

# ====================== Multi-Label Prediction ======================
def predict_multi_utterance_labels(
    text_list,
    tokenizer,
    model,
    mlb,
    threshold = 0.5
):
    """
    Conduct batch multi-label prediction for complete speaker turns
    """
    all_predicted_label_sets = []

    with torch.no_grad():
        for idx in tqdm(
            range(0, len(text_list), BATCH_SIZE),
            desc = "Multi-label DA Prediction"
        ):
            batch_texts = text_list[idx : idx + BATCH_SIZE]

            # Tokenization
            inputs = tokenizer(
                batch_texts,
                padding = True,
                truncation = True,
                max_length = MAX_SEQ_LENGTH,
                return_tensors = "pt"
            ).to(DEVICE)

            # Model forward pass
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits)

            # Threshold filtering for multi-label selection
            batch_pred_binary = (probs > threshold).cpu().numpy()

            # Map binary vector to actual label names
            for binary_vec in batch_pred_binary:
                selected_labels = list(mlb.classes_[np.where(binary_vec == 1)[0]])
                # Fallback: assign the most probable label if no label exceeds threshold
                if len(selected_labels) == 0:
                    top_label_idx = np.argmax(probs.cpu().numpy()[0])
                    selected_labels = [mlb.classes_[top_label_idx]]
                all_predicted_label_sets.append(",".join(selected_labels))

    return all_predicted_label_sets

# ====================== Main Inference Pipeline ======================
def main():
    # Load model and multi-label encoder
    print("Loading multi-label DA model and tokenizer...")
    tokenizer, model, mlb = load_multi_label_pipeline(MODEL_PATH, MLB_FILE_PATH)
    print(f"Available DA label set: {list(mlb.classes_)}")

    # Load input dataset
    print(f"Loading input data from {DATA_PATH}")
    df = pd.read_excel(DATA_PATH)

    if "Sentence" not in df.columns:
        raise KeyError("Input data must contain the column 'Sentence'.")

    # Preprocess input text
    input_texts = df["Sentence"].fillna("").tolist()

    # Run multi-label prediction
    print("Starting multi-label dialogue act prediction...")
    predicted_labels = predict_multi_utterance_labels(input_texts, tokenizer, model, mlb, PREDICTION_THRESHOLD)

    # Append prediction results
    df["Multi_DA_Labels"] = predicted_labels

    # Save output file
    df.to_excel(OUTPUT_PATH, index = False)
    print(f"\nPrediction completed. Results saved to: {OUTPUT_PATH}")

    # Print label distribution overview
    print("\n===== Predicted Multi-Label Distribution =====")
    print(df["Multi_DA_Labels"].value_counts().head(20))

if __name__ == "__main__":
    main()