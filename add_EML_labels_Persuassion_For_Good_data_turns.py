import torch
import pandas as pd
import os
import re
import string
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import warnings

# Disable irrelevant warnings
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ====================== Configuration Items ======================
# Input data path (your all_turns_data.xlsx)
INPUT_DATA_PATH = r"D:\Master_study\master_thesis_programs\my_project\data\Persuasion_For_Good\all_turns_data.xlsx"
# Output data path (labeled results)
OUTPUT_DATA_PATH = r"D:\Master_study\master_thesis_programs\my_project\output_data\all_turns_data_with_eml_label.xlsx"
# Trained model path
MODEL_PATH = "./models/mentalmanip_model"
# EML confidence threshold (adjust as needed)
CONFIDENCE_THRESHOLD = 0.6


# =================================================================

def load_trained_model():
    """Load pre-trained model and tokenizer"""
    # Check model path exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model path does not exist: {MODEL_PATH}\n"
            "Please confirm the model path is correct"
        )

    # Load tokenizer and model
    print(f"Loading trained model from: {MODEL_PATH}")
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels = 2,
        ignore_mismatched_sizes = True
    )

    # Set model to evaluation mode
    model.eval()
    model.to("cpu")
    print("Model loaded successfully!")
    return tokenizer, model


def clean_text(text):
    """Text cleaning function (consistent with training data)"""
    if pd.isna(text) or text.strip() == "":
        return ""
    text = str(text).lower().strip()
    # Keep key punctuation (?, !, .)
    keep_punct = {"?", "!", "."}
    text = "".join([c for c in text if c not in string.punctuation or c in keep_punct])
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)
    return text


def label_eml_data(tokenizer, model):
    """Label EML for the donation conversation data"""
    # Load data
    print(f"\nLoading data from: {INPUT_DATA_PATH}")
    df = pd.read_excel(INPUT_DATA_PATH, dtype = str)
    print(f"Successfully loaded {len(df)} samples")

    # Verify Sentence column exists (your data's text column)
    if "Sentence" not in df.columns:
        raise ValueError("Data file missing 'Sentence' column! Please check your data format")

    # Clean text
    print("Cleaning text data...")
    df["clean_sentence"] = df["Sentence"].apply(clean_text)
    text_list = df["clean_sentence"].tolist()

    # Labeling preparation
    eml_labels = []
    confidence_scores = []
    total_samples = len(text_list)

    print(f"\nStarting EML labeling for {total_samples} samples (threshold={CONFIDENCE_THRESHOLD})...")

    # Batch inference with no gradient calculation
    with torch.no_grad():
        for idx, text in enumerate(text_list):
            # Handle empty/short text
            if len(text) < 5:
                eml_labels.append(0)
                confidence_scores.append(1.0)
                continue

            # Tokenize text
            inputs = tokenizer(
                text,
                truncation = True,
                padding = "max_length",
                max_length = 128,
                return_tensors = "pt"
            ).to("cpu")

            # Model inference
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim = 1)
            confidence = torch.max(probs).item()
            pred_label = torch.argmax(probs, dim = 1).item()

            # Apply confidence threshold
            if pred_label == 1 and confidence >= CONFIDENCE_THRESHOLD:
                eml_labels.append(1)
            else:
                eml_labels.append(0)

            confidence_scores.append(round(confidence, 3))

            # Progress update
            if (idx + 1) % 100 == 0:
                print(f"Labeled {idx + 1}/{total_samples} samples")

    # Add labels to dataframe
    df["eml_label"] = eml_labels  # 0=non-EML, 1=EML
    df["eml_confidence"] = confidence_scores

    # Remove temporary column
    df = df.drop(columns = ["clean_sentence"])

    # Save results
    os.makedirs(os.path.dirname(OUTPUT_DATA_PATH), exist_ok = True)
    df.to_excel(OUTPUT_DATA_PATH, index = False)
    print(f"\nLabeled data saved to: {OUTPUT_DATA_PATH}")

    # Print labeling statistics
    label_counts = df["eml_label"].value_counts()
    print("\nLabeling results summary:")
    print(f"Non-Emotional Manipulation (0): {label_counts.get(0, 0)} samples")
    print(f"Emotional Manipulation (1): {label_counts.get(1, 0)} samples")
    print(f"Average confidence score: {np.mean(confidence_scores):.3f}")


if __name__ == "__main__":
    # Load trained model (no training code)
    tokenizer, model = load_trained_model()

    # Label the data
    label_eml_data(tokenizer, model)

    print("\nAll labeling completed successfully!")