import torch
import pandas as pd
import os
import re
import string
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import load_dataset, Dataset, ClassLabel
import numpy as np
import warnings

# Disable irrelevant warnings
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ====================== Configuration Items to Modify ======================
# Data path
YOUR_DATA_PATH = r"D:\Master_study\master_thesis_programs\my_project\data\Persuasion_For_Good\100_sample_turns_data_with_manual_label.xlsx"
OUTPUT_PATH = r"D:\Master_study\master_thesis_programs\my_project\output_data\100_sample_eml_label_final.xlsx"

# Core switch: whether to retrain the model (False=use existing model, True=retrain)
RETRAIN_MODEL = False

# Model save path (must match the path used during training)
MODEL_SAVE_PATH = "./models/mentalmanip_model"
# ===========================================================================

# ---------------------- 1. Load MentalManip Dataset (Adapt to Your Format) ----------------------
def load_mentalmanip_dataset():
    """
    Adapt to your dataset format:
    - Text column: dialogue
    - Manipulation label: majority vote from 3 annotators (label as 1 if >=2 annotators mark as manipulative)
    """
    # Load HF dataset
    dataset = load_dataset("audreyeleven/MentalManip", "mentalmanip_detailed")
    df = dataset["train"].to_pandas()

    print("Dataset column names: ", df.columns.tolist())
    print("Total dataset rows: ", len(df))

    # Text cleaning function
    def clean_text(text):
        text = str(text).lower().strip()
        # Keep key punctuation (question mark/exclamation mark/period), remove other punctuation
        keep_punct = {"?", "!", "."}
        text = "".join([c for c in text if c not in string.punctuation or c in keep_punct])
        # Remove extra spaces
        text = re.sub(r"\s+", " ", text)
        return text

    # 1. Clean dialogue text
    df["clean_dialogue"] = df["dialogue"].apply(clean_text)

    # 2. Calculate manipulation label (majority vote: >=2 annotators mark as manipulative = 1)
    def get_manipulative_label(row):
        # Get manipulation labels from 3 annotators (convert to int)
        m1 = int(row.get("manipulative_1", 0))
        m2 = int(row.get("manipulative_2", 0))
        m3 = int(row.get("manipulative_3", 0))
        # Majority vote
        total = m1 + m2 + m3
        return 1 if total >= 2 else 0

    df["label"] = df.apply(get_manipulative_label, axis = 1)

    # Filter invalid data
    df = df.dropna(subset = ["clean_dialogue", "label"])
    df = df[df["clean_dialogue"].str.len() > 5]

    # Count label distribution
    label_counts = df["label"].value_counts()
    print("Dataset label distribution:")
    print("   - Non-manipulative (0): ", label_counts.get(0, 0), " rows")
    print("   - Manipulative (1): ", label_counts.get(1, 0), " rows")

    # Keep only required columns
    df_final = df[["clean_dialogue", "label"]].rename(columns = {"clean_dialogue": "text"})
    return df_final

# ---------------------- 2. Data Preprocessing (Fix ClassLabel Issue) ----------------------
def preprocess_dataset(df):
    """Convert to HF Dataset format and preprocess, fix ClassLabel issue"""
    # Convert to HF Dataset
    dataset = Dataset.from_pandas(df)

    # Key fix: convert label column to ClassLabel type
    class_label = ClassLabel(num_classes = 2, names = ["non-manipulative", "manipulative"])
    dataset = dataset.cast_column("label", class_label)

    # Load BERT Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Preprocessing function
    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            truncation = True,
            padding = "max_length",
            max_length = 128,
            return_attention_mask = True
        )

    # Batch preprocessing
    tokenized_dataset = dataset.map(preprocess_function, batched = True, batch_size = 32)

    # Set torch format
    tokenized_dataset.set_format(
        type = "torch",
        columns = ["input_ids", "attention_mask", "label"]
    )

    # Stratified split into train/test sets (stratify_by_column is supported now)
    train_test_split = tokenized_dataset.train_test_split(
        test_size = 0.2,
        seed = 42,
        stratify_by_column = "label"
    )

    return train_test_split, tokenizer

# ---------------------- 3. Train EML Detection Model ----------------------
def train_model():
    """Train BERT model"""
    # Load and preprocess dataset
    df = load_mentalmanip_dataset()
    dataset, tokenizer = preprocess_dataset(df)

    # Load BERT model
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels = 2,
        ignore_mismatched_sizes = True,
        hidden_dropout_prob = 0.2  # Regularization to reduce overfitting
    )

    # Evaluation metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis = -1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average = "binary")
        }

    # Training arguments (CPU-friendly configuration)
    training_args = TrainingArguments(
        output_dir = MODEL_SAVE_PATH,  # Use unified model path
        learning_rate = 2e-5,
        per_device_train_batch_size = 4,  # Use 4 for CPU, 8/16 for GPU
        per_device_eval_batch_size = 4,
        num_train_epochs = 3,
        weight_decay = 0.01,
        evaluation_strategy = "epoch",  # Evaluate once per epoch
        save_strategy = "epoch",  # Save once per epoch
        load_best_model_at_end = True,  # Load best model at end
        metric_for_best_model = "f1",  # Use F1 score as best metric
        fp16 = False,  # Disable mixed precision for CPU
        report_to = "none",  # Do not report to wandb
        disable_tqdm = False,  # Show progress bar
        no_cuda = True  # Force use CPU
    )

    # Build trainer
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = dataset["train"],
        eval_dataset = dataset["test"],
        compute_metrics = compute_metrics,
        data_collator = DataCollatorWithPadding(tokenizer = tokenizer)
    )

    # Start training
    print("\nStarting EML detection model training...")
    trainer.train()

    # Save model and Tokenizer
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print("Model training completed, saved to: ", MODEL_SAVE_PATH)

    # Evaluate best model
    eval_results = trainer.evaluate()
    print("\nModel evaluation results:")
    print("   - Accuracy: ", "{:.4f}".format(eval_results['eval_accuracy']))
    print("   - F1 score: ", "{:.4f}".format(eval_results['eval_f1']))

    return tokenizer, model

# ---------------------- 4. Load Existing Model ----------------------
def load_existing_model():
    """Load trained model and Tokenizer"""
    # Check if model path exists
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(
            "\nModel path does not exist: " + MODEL_SAVE_PATH + "\n"
            "Please set RETRAIN_MODEL = True to train the model first, or confirm the model path is correct"
        )

    # Load Tokenizer and model
    print("\nLoading existing model: ", MODEL_SAVE_PATH)
    tokenizer = BertTokenizer.from_pretrained(MODEL_SAVE_PATH)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_SAVE_PATH,
        num_labels = 2,
        ignore_mismatched_sizes = True
    )
    print("Existing model loaded successfully!")
    return tokenizer, model

# ---------------------- 5. Label Your Donation Data ----------------------
def label_donation_data(tokenizer, model):
    """Label donation data with trained model"""
    # Check input file
    if not os.path.exists(YOUR_DATA_PATH):
        raise FileNotFoundError("\nDonation data file does not exist: " + YOUR_DATA_PATH)

    # Read donation data
    df = pd.read_excel(YOUR_DATA_PATH, dtype = str)
    print("\nSuccessfully loaded donation data: total ", len(df), " samples")

    # Auto identify text column
    text_column = None
    for col in df.columns:
        if col.lower() in ["sentence", "text", "content", "dialogue"]:
            text_column = col
            break
    if not text_column:
        text_column = df.columns[0]
    print("Identified text column: ", text_column)

    # Text cleaning (consistent with training set)
    def clean_text(text):
        text = str(text).lower().strip()
        keep_punct = {"?", "!", "."}
        text = "".join([c for c in text if c not in string.punctuation or c in keep_punct])
        text = re.sub(r"\s+", " ", text)
        return text

    df["clean_text"] = df[text_column].fillna("").apply(clean_text)
    text_list = df["clean_text"].tolist()

    # Inference configuration
    model.eval()
    model.to("cpu")

    # Batch labeling (confidence threshold 0.6, accurately distinguish EML)
    print("\nStarting to label ", len(text_list), " donation texts (EML determination threshold=0.6)...")
    labels = []
    confidences = []

    with torch.no_grad():
        for i, text in enumerate(text_list):
            # Skip empty text
            if len(text) < 5:
                labels.append(0)
                confidences.append(1.0)
                continue

            # Tokenize
            inputs = tokenizer(
                text,
                truncation = True,
                padding = "max_length",
                max_length = 128,
                return_tensors = "pt"
            ).to("cpu")

            # Inference: get probabilities and label
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim = 1)
            confidence = torch.max(probs).item()
            pred_label = torch.argmax(probs, dim = 1).item()

            # Only label manipulative samples with high confidence as 1
            if pred_label == 1 and confidence >= 0.6:
                labels.append(1)
            else:
                labels.append(0)
            confidences.append(round(confidence, 3))

            # Progress prompt
            if (i + 1) % 50 == 0:
                print("   Labeled ", (i + 1), "/", len(text_list), " samples")

    # Save results
    df["eml_label"] = labels  # 0=non-EML, 1=EML
    df["confidence"] = confidences  # Confidence (0-1)
    df = df.drop(columns = ["clean_text"])

    # Create output directory
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok = True)
    df.to_excel(OUTPUT_PATH, index = False)

    # Count results
    label_counts = df["eml_label"].value_counts()
    print("\nLabeling completed! Results saved to: ", OUTPUT_PATH)
    print("\nLabeling results distribution:")
    print("   - Non-emotional manipulation (0): ", label_counts.get(0, 0), " samples")
    print("   - Emotional manipulation (1): ", label_counts.get(1, 0), " samples")
    print("   - Average confidence: ", "{:.3f}".format(np.mean(confidences)))

# ---------------------- Main Program ----------------------
if __name__ == "__main__":
    # Choose to load existing model or retrain based on switch
    if RETRAIN_MODEL:
        print("Mode: retrain model + label data")
        tokenizer, model = train_model()
    else:
        print("Mode: use existing model + label data")
        tokenizer, model = load_existing_model()

    # Label donation data
    label_donation_data(tokenizer, model)

    print("\nAll operations completed! EML labeling results saved.")