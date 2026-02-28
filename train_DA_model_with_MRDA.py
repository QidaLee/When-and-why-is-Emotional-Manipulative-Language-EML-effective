import os
import re
import time
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm  # Import progress bar tool

# -------------------------- 1. Initialize NLTK resources (download first run) --------------------------
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# -------------------------- 2. Configuration Items --------------------------
DATA_DIR = "./data/MRDA"  # Your data directory path
LABEL_TYPE = "Basic"      # Optional: Basic/General/Full, specify label level for classification
SEED = 42                 # Fixed random seed for reproducibility
# Optional: Small batch test (run 1000 samples first to verify if stuck), set to False later
TEST_SMALL_BATCH = False
BATCH_SIZE = 1000         # Number of samples for small batch test


# -------------------------- 3. Data Loading and Parsing --------------------------
def load_mrda_data(file_path, label_type="Basic"):
    """
    Load MRDA dataset file and parse into texts and labels
    :param file_path: Path to dataset file (train/test/val_set.txt)
    :param label_type: Label level for classification (Basic/General/Full)
    :return: texts (list) - list of utterance texts, labels (list) - list of corresponding labels
    """
    texts = []
    labels = []
    label_idx = {"Basic": 2, "General": 3, "Full": 4}[label_type]

    print(f"Start loading {os.path.basename(file_path)}...")
    start_time = time.time()
    with open(file_path, "r", encoding="utf-8") as f:
        # Add progress bar for file reading
        for line in tqdm(f, desc=f"Loading {os.path.basename(file_path)}"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 5:
                continue
            text = parts[1].strip()
            label = parts[label_idx].strip()
            texts.append(text)
            labels.append(label)

    # Small batch test: only take first N samples
    if TEST_SMALL_BATCH:
        texts = texts[:BATCH_SIZE]
        labels = labels[:BATCH_SIZE]

    load_time = time.time() - start_time
    print(
        f"Loaded {os.path.basename(file_path)}: Text count={len(texts)}, Label count={len(labels)} (Time: {load_time:.2f}s)")
    print(f"Label distribution: {Counter(labels)}")
    return texts, labels


# Load training/test/validation sets
train_texts, train_labels = load_mrda_data(os.path.join(DATA_DIR, "train_set.txt"), LABEL_TYPE)
val_texts, val_labels = load_mrda_data(os.path.join(DATA_DIR, "val_set.txt"), LABEL_TYPE)
test_texts, test_labels = load_mrda_data(os.path.join(DATA_DIR, "test_set.txt"), LABEL_TYPE)


# -------------------------- 4. Text Preprocessing Function --------------------------
def preprocess_text(text):
    """
    Text preprocessing pipeline: lowercase, remove special chars, remove stopwords, stemmer
    :param text: Raw input text
    :return: Preprocessed text string
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words and len(token) > 1]
    return " ".join(tokens)


# Add progress bar for preprocessing
print("\nStart text preprocessing...")
start_time = time.time()
train_texts_processed = [preprocess_text(t) for t in tqdm(train_texts, desc="Preprocessing train set")]
val_texts_processed = [preprocess_text(t) for t in tqdm(val_texts, desc="Preprocessing val set")]
test_texts_processed = [preprocess_text(t) for t in tqdm(test_texts, desc="Preprocessing test set")]
preprocess_time = time.time() - start_time
print(f"Text preprocessing finished (Time: {preprocess_time:.2f}s)\n")

# -------------------------- 5. Feature Extraction (TF-IDF) --------------------------
print("Start TF-IDF feature extraction...")
start_time = time.time()
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)
# Add progress hint for TF-IDF fitting
X_train = tfidf.fit_transform(tqdm(train_texts_processed, desc="Fitting TF-IDF on train set"))
X_val = tfidf.transform(tqdm(val_texts_processed, desc="Transforming val set"))
X_test = tfidf.transform(tqdm(test_texts_processed, desc="Transforming test set"))
tfidf_time = time.time() - start_time
print(f"TF-IDF feature extraction finished (Time: {tfidf_time:.2f}s)\n")

y_train = train_labels
y_val = val_labels
y_test = test_labels

# -------------------------- 6. Model Training (with progress display) --------------------------
print("Start model training with grid search...")
start_time = time.time()
# Initialize model (multi-threading acceleration)
lr = LogisticRegression(random_state=SEED, max_iter=1000, n_jobs=-1)

# Grid search parameters (support multi-class classification)
param_grid = {
    "C": [0.1, 1.0, 10.0],
    "solver": ["saga", "newton-cholesky"]
}

# Grid search: increase verbose level to show detailed progress
grid_search = GridSearchCV(
    estimator=lr,
    param_grid=param_grid,
    cv=3,
    scoring="f1_macro",
    verbose=2,  # Change to 2 to show detailed progress of each cross-validation round
    n_jobs=-1   # Multi-threading acceleration
)

# Training progress hint
print(f"Grid search started (CV=3, params={len(param_grid['C']) * len(param_grid['solver'])} combinations)...")
grid_search.fit(X_train, y_train)
train_time = time.time() - start_time
print(f"\nModel training finished (Time: {train_time:.2f}s)")
print(f"Best hyperparameters: {grid_search.best_params_}")
best_model = grid_search.best_estimator_

# -------------------------- 7. Model Evaluation (on test set) --------------------------
print("\nStart model evaluation on test set...")
start_time = time.time()
y_test_pred = best_model.predict(tqdm(X_test, desc="Predicting on test set"))
eval_time = time.time() - start_time

# Print evaluation metrics
print(f"\n===== Test Set Evaluation Results (Time: {eval_time:.2f}s) =====")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print("\nClassification Report (Precision/Recall/F1):")
print(classification_report(y_test, y_test_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
cm_df = pd.DataFrame(
    cm,
    index=sorted(Counter(y_test).keys()),
    columns=sorted(Counter(y_test).keys())
)
print("\nConfusion Matrix:")
print(cm_df)


# -------------------------- 8. Predict New Examples --------------------------
def predict_utterance(text, model, tfidf, preprocess_func):
    text_processed = preprocess_func(text)
    text_tfidf = tfidf.transform([text_processed])
    pred_label = model.predict(text_tfidf)[0]
    return pred_label


# Test prediction examples
test_examples = [
    "Can you pass the meeting notes to me?",
    "The project deadline is next Friday.",
    "Okay, let's move to the next topic."
]
print("\n===== Prediction Examples =====")
for example in test_examples:
    pred = predict_utterance(example, best_model, tfidf, preprocess_text)
    print(f"Text: {example}\nPredicted label: {pred}\n")