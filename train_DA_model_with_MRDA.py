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
from sklearn.decomposition import TruncatedSVD
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm  # Import progress bar tool

# -------------------------- 1. Initialize NLTK resources (download first run) --------------------------
# Download required NLTK resources quietly (only on first run)
nltk.download('stopwords', quiet = True)
nltk.download('punkt_tab', quiet = True)
stop_words = set(stopwords.words('english'))  # English stopwords set
stemmer = PorterStemmer()  # Initialize Porter Stemmer for word stemming

# -------------------------- 2. Configuration Items --------------------------
DATA_DIR = "./data/MRDA"  # Path to MRDA dataset directory
LABEL_TYPE = "Full"  # Label granularity: Basic/General/Full
SEED = 42  # Fixed random seed for reproducibility
TEST_SMALL_BATCH = False  # Set to True for quick testing with small batch
BATCH_SIZE = 1000  # Number of samples for small batch testing
TFIDF_MAX_FEATURES = 5000  # Max features for TF-IDF (controls memory usage)
SVD_COMPONENTS = 1000  # Number of dimensions after SVD dimensionality reduction
CV_FOLDS = 3  # Number of cross-validation folds


# -------------------------- 3. Data Loading and Parsing --------------------------
def load_mrda_data(file_path, label_type="Basic"):
    """
    Load and parse MRDA dataset files into text and label lists

    Args:
        file_path (str): Path to MRDA dataset file (train/test/val_set.txt)
        label_type (str): Label granularity level (Basic/General/Full)

    Returns:
        tuple: (texts, labels) - list of utterance texts and corresponding labels
    """
    texts = []
    labels = []
    # Map label type to column index in MRDA format
    label_idx = {"Basic": 2, "General": 3, "Full": 4}[label_type]

    print(f"Starting to load {os.path.basename(file_path)}...")
    start_time = time.time()

    # Read file with progress bar
    with open(file_path, "r", encoding = "utf-8") as f:
        for line in tqdm(f, desc = f"Loading {os.path.basename(file_path)}"):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            parts = line.split("|")
            if len(parts) != 5:  # Validate MRDA format
                continue
            text = parts[1].strip()
            label = parts[label_idx].strip()
            texts.append(text)
            labels.append(label)

    # Small batch test - use only first N samples for quick validation
    if TEST_SMALL_BATCH:
        texts = texts[:BATCH_SIZE]
        labels = labels[:BATCH_SIZE]

    load_time = time.time() - start_time
    print(
        f"Loaded {os.path.basename(file_path)}: {len(texts)} texts, {len(labels)} labels (Time: {load_time:.2f}s)")
    print(f"Label distribution: {Counter(labels)}")
    return texts, labels


# Load training, validation and test datasets
train_texts, train_labels = load_mrda_data(os.path.join(DATA_DIR, "train_set.txt"), LABEL_TYPE)
val_texts, val_labels = load_mrda_data(os.path.join(DATA_DIR, "val_set.txt"), LABEL_TYPE)
test_texts, test_labels = load_mrda_data(os.path.join(DATA_DIR, "test_set.txt"), LABEL_TYPE)


# -------------------------- 4. Text Preprocessing Function --------------------------
def preprocess_text(text):
    """
    Complete text preprocessing pipeline for MRDA utterances:
    1. Convert to lowercase
    2. Remove special characters and numbers
    3. Tokenization
    4. Stopword removal
    5. Stemming
    6. Reconstruct text string

    Args:
        text (str): Raw input utterance text

    Returns:
        str: Preprocessed text string
    """
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers (keep only letters and spaces)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Tokenization
    tokens = nltk.word_tokenize(text)
    # Remove stopwords, short tokens and apply stemming
    tokens = [
        stemmer.stem(token)
        for token in tokens
        if token not in stop_words and len(token) > 1
    ]
    # Reconstruct preprocessed text
    return " ".join(tokens)


# Apply preprocessing with progress bars
print("\nStarting text preprocessing...")
start_time = time.time()
train_texts_processed = [preprocess_text(t) for t in tqdm(train_texts, desc = "Preprocessing train set")]
val_texts_processed = [preprocess_text(t) for t in tqdm(val_texts, desc = "Preprocessing val set")]
test_texts_processed = [preprocess_text(t) for t in tqdm(test_texts, desc = "Preprocessing test set")]
preprocess_time = time.time() - start_time
print(f"Text preprocessing completed (Time: {preprocess_time:.2f}s)\n")

# -------------------------- 5. Feature Extraction (TF-IDF) --------------------------
print("Starting TF-IDF feature extraction...")
start_time = time.time()

# Initialize TF-IDF vectorizer with memory constraints
tfidf = TfidfVectorizer(
    max_features = 3000,  # Limit features to control memory usage
    ngram_range = (1, 1)  # Use unigrams and bigrams
)

# Fit TF-IDF on training data (learn vocabulary) and transform all sets
X_train = tfidf.fit_transform(tqdm(train_texts_processed, desc = "Fitting TF-IDF on train set"))
X_val = tfidf.transform(tqdm(val_texts_processed, desc = "Transforming val set"))
X_test = tfidf.transform(tqdm(test_texts_processed, desc = "Transforming test set"))

tfidf_time = time.time() - start_time
print(f"TF-IDF feature extraction completed (Time: {tfidf_time:.2f}s)")
print(f"TF-IDF feature dimensions: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}\n")

# Assign labels to variables
y_train = train_labels
y_val = val_labels
y_test = test_labels

# -------------------------- 6. Dimensionality Reduction (SVD) --------------------------
print("Starting dimensionality reduction with Truncated SVD...")
start_time = time.time()

# Initialize SVD (fixed integer n_components - critical fix for sklearn >=1.2)
svd = TruncatedSVD(
    n_components = 500,  # Must be integer (fixed from 0.9)
    random_state = SEED  # Reproducible results
)

# Fit SVD on training data only (prevent data leakage) and transform all sets
X_train_reduced = svd.fit_transform(X_train)
X_val_reduced = svd.transform(X_val)
X_test_reduced = svd.transform(X_test)

svd_time = time.time() - start_time
print(f"SVD dimensionality reduction completed (Time: {svd_time:.2f}s)")
print(f"Reduced dimensions: Train={X_train_reduced.shape}, Val={X_val_reduced.shape}, Test={X_test_reduced.shape}")
print(f"Explained variance ratio (total): {svd.explained_variance_ratio_.sum():.4f}\n")

# -------------------------- 7. Model Training with Grid Search --------------------------
print("Starting model training with grid search...")
start_time = time.time()

# Initialize Logistic Regression (removed n_jobs which is deprecated in sklearn >=1.8)
lr = LogisticRegression(
    random_state = SEED,
    max_iter = 1000,  # Increased iterations for convergence
    # multi_class = "auto"  # Auto-detect multi-class strategy
)

# Grid search parameters (balanced for performance and training time)
param_grid = {
    "C": [0.1, 1.0, 10.0],  # Regularization strength
    "solver": ["saga"]  # Optimizers for multi-class
}

# Initialize Grid Search with cross-validation
grid_search = GridSearchCV(
    estimator=lr,
    param_grid=param_grid,
    cv=CV_FOLDS,  # k-fold
    scoring="f1_macro",
    verbose=2,
    n_jobs=1
)

lr = LogisticRegression(
    random_state=SEED,
    max_iter=500,
    class_weight="balanced"
)



# Calculate total combinations for progress tracking
total_combinations = len(param_grid["C"]) * len(param_grid["solver"])
print(f"Grid search started (CV={CV_FOLDS}, params={total_combinations} combinations)...")

# Train model on reduced features
grid_search.fit(X_train_reduced, y_train)

train_time = time.time() - start_time
print(f"\nModel training completed (Time: {train_time:.2f}s)")
print(f"Best hyperparameters: {grid_search.best_params_}")
print(f"Best cross-validation score (F1-macro): {grid_search.best_score_:.4f}")
best_model = grid_search.best_estimator_

# -------------------------- 8. Model Evaluation --------------------------
print("\nStarting model evaluation on test set...")
start_time = time.time()

# Make predictions on test set (use reduced features)
y_test_pred = best_model.predict(tqdm(X_test_reduced, desc = "Predicting on test set"))
eval_time = time.time() - start_time

# Print comprehensive evaluation metrics
print(f"\n===== Test Set Evaluation Results (Time: {eval_time:.2f}s) =====")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

print("\nClassification Report (Precision/Recall/F1-Score):")
print(classification_report(y_test, y_test_pred, zero_division = 0))

# Generate and print confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
# Get sorted unique labels for consistent matrix formatting
unique_labels = sorted(Counter(y_test).keys())
cm_df = pd.DataFrame(
    cm,
    index = unique_labels,
    columns = unique_labels
)
print("\nConfusion Matrix:")
print(cm_df)


# -------------------------- 9. Prediction Function for New Utterances --------------------------
def predict_utterance(text, model, tfidf, svd, preprocess_func):
    """
    Predict dialogue act label for new utterance text

    Args:
        text (str): Raw input utterance
        model: Trained classification model
        tfidf: Fitted TF-IDF vectorizer
        svd: Fitted TruncatedSVD transformer
        preprocess_func: Text preprocessing function

    Returns:
        str: Predicted dialogue act label
    """
    # Complete preprocessing pipeline
    text_processed = preprocess_func(text)
    # Transform to TF-IDF features
    text_tfidf = tfidf.transform([text_processed])
    # Apply SVD dimensionality reduction
    text_reduced = svd.transform(text_tfidf)
    # Predict label
    pred_label = model.predict(text_reduced)[0]
    return pred_label


# -------------------------- 10. Example Predictions --------------------------
# Test prediction with sample utterances
test_examples = [
    "Can you pass the meeting notes to me?",
    "The project deadline is next Friday.",
    "Okay, let's move to the next topic."
]

print("\n===== Prediction Examples =====")
for example in test_examples:
    pred = predict_utterance(example, best_model, tfidf, svd, preprocess_text)
    print(f"Text: {example}")
    print(f"Predicted DA label: {pred}\n")