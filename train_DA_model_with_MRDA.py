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
from tqdm import tqdm
import joblib

# -------------------------- 1. Initialize NLTK Resources --------------------------
nltk.download('stopwords', quiet = True)
nltk.download('punkt_tab', quiet = True)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# -------------------------- 2. Configuration (Added Core Switch Parameter) --------------------------
DATA_DIR = "./data/MRDA"
LABEL_TYPE = "Full"  # Basic/General/Full hierarchy for MRDA dialogue act labels
SEED = 42  # Fixed random seed for reproducibility

# Core Switch: True = Load saved model, False = Retrain from scratch
USE_SAVED_MODEL = True

# Test/memory optimization configuration
TEST_SMALL_BATCH = False  # Enable for quick testing with limited samples
BATCH_SIZE = 1000  # Number of samples for small batch testing
TFIDF_MAX_FEATURES = 1000  # Reduced from 3000 to lower memory usage
SVD_COMPONENTS = 200  # Reduced from 500 to speed up dimensionality reduction
CV_FOLDS = 2  # Reduced from 3 to accelerate grid search

# Model saving paths (as requested: ./models/Logistic_DA_MRDA)
MODEL_BASE_DIR = "./models"
MODEL_SAVE_DIR = os.path.join(MODEL_BASE_DIR, "Logistic_DA_MRDA")
MODEL_FILENAME = f"logistic_da_mrda_{LABEL_TYPE.lower()}.pkl"
TFIDF_FILENAME = f"tfidf_mrda_{LABEL_TYPE.lower()}.pkl"
SVD_FILENAME = f"svd_mrda_{LABEL_TYPE.lower()}.pkl"
os.makedirs(MODEL_SAVE_DIR, exist_ok = True)  # Create directory if not exists

# Full file paths for model loading/saving
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME)
TFIDF_PATH = os.path.join(MODEL_SAVE_DIR, TFIDF_FILENAME)
SVD_PATH = os.path.join(MODEL_SAVE_DIR, SVD_FILENAME)


# -------------------------- 3. Model Loading/Training Control Logic (Added) --------------------------
def load_saved_models():
    """Load pre-saved logistic regression model, TF-IDF vectorizer, and SVD transformer"""
    try:
        print(f"Loading saved model from {MODEL_SAVE_DIR}...")
        model = joblib.load(MODEL_PATH)
        tfidf = joblib.load(TFIDF_PATH)
        svd = joblib.load(SVD_PATH)
        print("Saved models loaded successfully!")
        return model, tfidf, svd
    except FileNotFoundError as e:
        print(f"Saved model files not found: {e}")
        print("Will start new training instead...")
        return None, None, None
    except Exception as e:
        print(f"Failed to load saved models: {e}")
        print("Will start new training instead...")
        return None, None, None


# -------------------------- 4. Data Loading --------------------------
def load_mrda_data(file_path, label_type="Basic"):
    """
    Load and parse MRDA dataset files into structured text and label lists.

    Args:
        file_path (str): Path to MRDA dataset file (train_set.txt/val_set.txt/test_set.txt)
        label_type (str): Label hierarchy level (Basic/General/Full)

    Returns:
        tuple: (texts, labels) - List of utterance texts and corresponding DA labels
    """
    texts = []
    labels = []
    # Map label type to column index in MRDA pipe-separated format
    label_idx = {"Basic": 2, "General": 3, "Full": 4}[label_type]

    print(f"Starting to load {os.path.basename(file_path)}...")
    start_time = time.time()
    with open(file_path, "r", encoding = "utf-8") as f:
        for line in tqdm(f, desc = f"Loading {os.path.basename(file_path)}"):
            line = line.strip()
            # Skip empty lines or lines with invalid format (not 5 fields separated by |)
            if not line or len(line.split("|")) != 5:
                continue
            parts = line.split("|")
            text = parts[1].strip()  # Extract utterance text
            label = parts[label_idx].strip()  # Extract corresponding DA label
            texts.append(text)
            labels.append(label)

    # Limit samples for small batch testing
    if TEST_SMALL_BATCH:
        texts = texts[:BATCH_SIZE]
        labels = labels[:BATCH_SIZE]

    load_time = time.time() - start_time
    print(f"Loaded {os.path.basename(file_path)}: {len(texts)} texts, {len(labels)} labels (Time: {load_time:.2f}s)")
    print(f"Label distribution: {Counter(labels)}")
    return texts, labels


# Load train/validation/test datasets (required for evaluation even when loading saved model)
train_texts, train_labels = load_mrda_data(os.path.join(DATA_DIR, "train_set.txt"), LABEL_TYPE)
val_texts, val_labels = load_mrda_data(os.path.join(DATA_DIR, "val_set.txt"), LABEL_TYPE)
test_texts, test_labels = load_mrda_data(os.path.join(DATA_DIR, "test_set.txt"), LABEL_TYPE)

# -------------------------- 关键修复1：全局定义y_test变量 --------------------------
# Define target variables globally (available for both training and loading branches)
y_train = train_labels
y_val = val_labels
y_test = test_labels  # 核心：把y_test提到全局，无论是否训练都能访问

# -------------------------- 5. Text Preprocessing --------------------------
def preprocess_text(text):
    """
    Complete text preprocessing pipeline for MRDA utterances:
    1. Lowercasing
    2. Special character/number removal
    3. Tokenization
    4. Stopword removal
    5. Stemming
    6. Reconstruction of cleaned text string

    Args:
        text (str): Raw input utterance text

    Returns:
        str: Preprocessed and normalized text string
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters and numbers
    tokens = nltk.word_tokenize(text)  # Tokenization
    # Filter stopwords, short tokens and apply stemming
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words and len(token) > 1]
    return " ".join(tokens)  # Reconstruct preprocessed text


# Apply preprocessing (required for test set even when loading saved model)
print("\nStarting text preprocessing...")
start_time = time.time()
train_texts_processed = [preprocess_text(t) for t in tqdm(train_texts, desc = "Preprocessing train set")]
val_texts_processed = [preprocess_text(t) for t in tqdm(val_texts, desc = "Preprocessing val set")]
test_texts_processed = [preprocess_text(t) for t in tqdm(test_texts, desc = "Preprocessing test set")]
preprocess_time = time.time() - start_time
print(f"Text preprocessing completed (Total Time: {preprocess_time:.2f}s)\n")

# -------------------------- 6. Core Logic: Load Model or Train Model --------------------------
best_model = None
tfidf_vectorizer = None
svd = None

# Step 1: Attempt to load saved model if enabled
if USE_SAVED_MODEL:
    best_model, tfidf_vectorizer, svd = load_saved_models()

# Step 2: Execute full training pipeline if model loading failed or retraining is enabled
if best_model is None or tfidf_vectorizer is None or svd is None:
    # -------------------------- 6.1 TF-IDF Feature Extraction --------------------------
    print("Starting TF-IDF feature extraction...")
    start_time = time.time()
    tfidf_vectorizer = TfidfVectorizer(
        max_features = TFIDF_MAX_FEATURES,
        ngram_range = (1, 1)  # Use only unigrams for memory efficiency
    )
    # Fit TF-IDF on training data and transform all datasets
    X_train = tfidf_vectorizer.fit_transform(tqdm(train_texts_processed, desc = "Fitting TF-IDF on train set"))
    X_val = tfidf_vectorizer.transform(tqdm(val_texts_processed, desc = "Transforming val set"))
    X_test = tfidf_vectorizer.transform(tqdm(test_texts_processed, desc = "Transforming test set"))

    tfidf_time = time.time() - start_time
    print(f"TF-IDF feature extraction completed (Time: {tfidf_time:.2f}s)")
    print(f"TF-IDF dimensions: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}\n")

    # -------------------------- 关键修复2：删除训练分支中重复的y_test赋值（可选） --------------------------
    # 注释/删除这部分重复赋值，因为已经在全局定义过了
    # y_train = train_labels
    # y_val = val_labels
    # y_test = test_labels

    # -------------------------- 6.2 SVD Dimensionality Reduction --------------------------
    print("Starting dimensionality reduction with Truncated SVD...")
    start_time = time.time()
    svd = TruncatedSVD(
        n_components = SVD_COMPONENTS,
        random_state = SEED  # Reproducible dimensionality reduction
    )
    # Fit SVD only on training data (prevent data leakage) and transform all datasets
    X_train_reduced = svd.fit_transform(X_train)
    X_val_reduced = svd.transform(X_val)
    X_test_reduced = svd.transform(X_test)

    svd_time = time.time() - start_time
    print(f"SVD completed (Time: {svd_time:.2f}s)")
    print(f"Reduced dimensions: Train={X_train_reduced.shape}, Val={X_val_reduced.shape}, Test={X_test_reduced.shape}")
    print(f"Explained variance ratio (total): {svd.explained_variance_ratio_.sum():.4f}\n")

    # -------------------------- 6.3 Model Training --------------------------
    print("Starting Logistic Regression training with grid search...")
    start_time = time.time()

    # Initialize Logistic Regression (optimized for Full hierarchy multi-class classification)
    logistic_model = LogisticRegression(
        random_state = SEED,
        max_iter = 500,  # Increased iterations for convergence in multi-class scenario
        class_weight = "balanced",  # Address class imbalance in Full hierarchy
        solver = "saga"  # Optimized solver for multi-class classification with L1/L2 regularization
    )

    # Simplified grid search parameters (reduce computation time)
    param_grid = {
        "C": [1.0]  # Regularization strength (single value for faster training)
    }

    # Initialize Grid Search with cross-validation
    grid_search = GridSearchCV(
        estimator = logistic_model,
        param_grid = param_grid,
        cv = CV_FOLDS,
        scoring = "f1_macro",  # Appropriate metric for imbalanced multi-class classification
        verbose = 1,  # Reduced verbosity to save memory
        n_jobs = 1  # Single thread to prevent memory issues with old sklearn versions
    )

    print(f"Grid search started (CV={CV_FOLDS}, parameters={len(param_grid['C'])} combinations)...")
    grid_search.fit(X_train_reduced, y_train)

    train_time = time.time() - start_time
    print(f"\nModel training completed (Time: {train_time:.2f}s)")
    print(f"Best hyperparameters: {grid_search.best_params_}")
    print(f"Best cross-validation F1-macro score: {grid_search.best_score_:.4f}")
    best_model = grid_search.best_estimator_

    # -------------------------- 6.4 Save Model --------------------------
    print(f"\nSaving model to {MODEL_SAVE_DIR}...")
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(tfidf_vectorizer, TFIDF_PATH)
    joblib.dump(svd, SVD_PATH)
    print("Model saved successfully!")

    # Assign reduced test set features for evaluation
    X_test_reduced = svd.transform(tfidf_vectorizer.transform(test_texts_processed))
else:
    # Regenerate test set features with saved TF-IDF/SVD (ensure consistency)
    print("\nGenerating test set features with saved TF-IDF/SVD...")
    X_test_tfidf = tfidf_vectorizer.transform(test_texts_processed)
    X_test_reduced = svd.transform(X_test_tfidf)
    print(f"Test set reduced dimensions: {X_test_reduced.shape}")

# -------------------------- 7. Model Evaluation --------------------------
print("\nStarting test set evaluation...")
start_time = time.time()

# Generate predictions on test set (using reduced features)
y_test_pred = best_model.predict(X_test_reduced)
eval_time = time.time() - start_time

print(f"\n===== Test Set Evaluation Results (Time: {eval_time:.2f}s) =====")
print(f"Overall Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

# Print detailed classification report (zero_division=0 to avoid errors with rare labels)
print("\nClassification Report (Precision/Recall/F1-Score):")
print(classification_report(y_test, y_test_pred, zero_division = 0))

# Generate and print confusion matrix (show top 10 labels only to avoid excessive output)
cm = confusion_matrix(y_test, y_test_pred)
unique_labels = sorted(Counter(y_test).keys())
cm_df = pd.DataFrame(cm[:10, :10], index = unique_labels[:10], columns = unique_labels[:10])
print("\nConfusion Matrix (Top 10 Labels):")
print(cm_df)


# -------------------------- 8. Prediction Function --------------------------
def predict_dialogue_act(text, model, tfidf, svd):
    """
    Predict dialogue act label for new/unseen utterance text using saved model.

    Args:
        text (str): Raw input utterance text
        model: Trained Logistic Regression model
        tfidf: Fitted TF-IDF vectorizer
        svd: Fitted Truncated SVD transformer

    Returns:
        str: Predicted dialogue act label
    """
    processed_text = preprocess_text(text)  # Apply same preprocessing as training
    tfidf_vec = tfidf.transform([processed_text])  # Convert text to TF-IDF features
    svd_vec = svd.transform(tfidf_vec)  # Apply dimensionality reduction
    return model.predict(svd_vec)[0]  # Predict and return label


# Example predictions with sample utterances
test_utterances = [
    "Can you share the project docs?",
    "The deadline is tomorrow.",
    "Let's move to the next topic."
]
print("\n===== Example Predictions =====")
for utterance in test_utterances:
    predicted_label = predict_dialogue_act(utterance, best_model, tfidf_vectorizer, svd)
    print(f"Utterance: {utterance}\nPredicted DA Label: {predicted_label}\n")