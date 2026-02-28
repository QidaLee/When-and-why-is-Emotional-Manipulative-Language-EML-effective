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
import joblib  # For model serialization

# -------------------------- 1. Initialize NLTK Resources --------------------------
# Download required NLTK resources (quiet mode for first run)
nltk.download('stopwords', quiet = True)
nltk.download('punkt_tab', quiet = True)

# Initialize NLP tools
stop_words = set(stopwords.words('english'))  # English stopwords collection
stemmer = PorterStemmer()  # Porter Stemmer for word normalization

# -------------------------- 2. Configuration Parameters --------------------------
# Dataset configuration
DATA_DIR = "./data/MRDA"  # MRDA dataset directory path
LABEL_TYPE = "Full"  # Label hierarchy: Basic/General/Full
SEED = 42  # Fixed random seed for reproducibility

# Testing configuration
TEST_SMALL_BATCH = False  # Enable for quick testing with limited samples
BATCH_SIZE = 1000  # Number of samples for small batch testing

# Feature engineering configuration
TFIDF_MAX_FEATURES = 3000  # Max TF-IDF features (reduced for Full hierarchy efficiency)
SVD_COMPONENTS = 500  # SVD dimensionality reduction target (500 dimensions)
CV_FOLDS = 3  # Number of cross-validation folds

# Model saving configuration (UPDATED: Save to ./models/Logistic_DA_MRDA)
MODEL_BASE_DIR = "./models"
MODEL_SAVE_DIR = os.path.join(MODEL_BASE_DIR, "Logistic_DA_MRDA")
# Unique filenames with label type identifier
MODEL_FILENAME = f"logistic_da_mrda_{LABEL_TYPE.lower()}.pkl"
TFIDF_FILENAME = f"tfidf_mrda_{LABEL_TYPE.lower()}.pkl"
SVD_FILENAME = f"svd_mrda_{LABEL_TYPE.lower()}.pkl"

# Create model directory if it doesn't exist
os.makedirs(MODEL_SAVE_DIR, exist_ok = True)


# -------------------------- 3. Data Loading and Parsing Function --------------------------
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
    label_index_map = {"Basic": 2, "General": 3, "Full": 4}
    label_idx = label_index_map[label_type]

    print(f"Starting to load {os.path.basename(file_path)}...")
    start_time = time.time()

    # Read file with progress bar
    with open(file_path, "r", encoding = "utf-8") as f:
        for line in tqdm(f, desc = f"Loading {os.path.basename(file_path)}"):
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            parts = line.split("|")
            if len(parts) != 5:  # Validate MRDA format (5 fields separated by |)
                continue

            # Extract text and label from correct columns
            text = parts[1].strip()
            label = parts[label_idx].strip()

            texts.append(text)
            labels.append(label)

    # Small batch testing - limit samples for quick validation
    if TEST_SMALL_BATCH:
        texts = texts[:BATCH_SIZE]
        labels = labels[:BATCH_SIZE]

    # Calculate loading statistics
    load_time = time.time() - start_time
    print(f"Loaded {os.path.basename(file_path)}: {len(texts)} texts, {len(labels)} labels (Time: {load_time:.2f}s)")
    print(f"Label distribution: {Counter(labels)}")

    return texts, labels


# Load train/validation/test datasets
train_texts, train_labels = load_mrda_data(os.path.join(DATA_DIR, "train_set.txt"), LABEL_TYPE)
val_texts, val_labels = load_mrda_data(os.path.join(DATA_DIR, "val_set.txt"), LABEL_TYPE)
test_texts, test_labels = load_mrda_data(os.path.join(DATA_DIR, "test_set.txt"), LABEL_TYPE)


# -------------------------- 4. Text Preprocessing Pipeline --------------------------
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
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and numbers (keep only letters and spaces)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Filter stopwords, short tokens and apply stemming
    processed_tokens = [
        stemmer.stem(token)
        for token in tokens
        if token not in stop_words and len(token) > 1
    ]

    # Reconstruct preprocessed text
    return " ".join(processed_tokens)


# Apply preprocessing with progress tracking
print("\nStarting text preprocessing...")
start_time = time.time()

train_texts_processed = [preprocess_text(t) for t in tqdm(train_texts, desc = "Preprocessing train set")]
val_texts_processed = [preprocess_text(t) for t in tqdm(val_texts, desc = "Preprocessing val set")]
test_texts_processed = [preprocess_text(t) for t in tqdm(test_texts, desc = "Preprocessing test set")]

preprocess_time = time.time() - start_time
print(f"Text preprocessing completed (Total Time: {preprocess_time:.2f}s)\n")

# -------------------------- 5. TF-IDF Feature Extraction --------------------------
print("Starting TF-IDF feature extraction...")
start_time = time.time()

# Initialize TF-IDF vectorizer with memory constraints (optimized for Full hierarchy)
tfidf_vectorizer = TfidfVectorizer(
    max_features = TFIDF_MAX_FEATURES,  # Limit features to control memory usage
    ngram_range = (1, 1)  # Use only unigrams for Full hierarchy efficiency
)

# Fit TF-IDF on training data (learn vocabulary) and transform all datasets
X_train = tfidf_vectorizer.fit_transform(tqdm(train_texts_processed, desc = "Fitting TF-IDF on train set"))
X_val = tfidf_vectorizer.transform(tqdm(val_texts_processed, desc = "Transforming val set"))
X_test = tfidf_vectorizer.transform(tqdm(test_texts_processed, desc = "Transforming test set"))

tfidf_time = time.time() - start_time
print(f"TF-IDF feature extraction completed (Time: {tfidf_time:.2f}s)")
print(f"TF-IDF feature dimensions: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}\n")

# Assign labels to target variables
y_train = train_labels
y_val = val_labels
y_test = test_labels

# -------------------------- 6. Dimensionality Reduction with Truncated SVD --------------------------
print("Starting dimensionality reduction with Truncated SVD...")
start_time = time.time()

# Initialize SVD (fixed integer components - compatible with all sklearn versions)
svd = TruncatedSVD(
    n_components = SVD_COMPONENTS,
    random_state = SEED  # Reproducible dimensionality reduction
)

# Fit SVD only on training data (prevent data leakage) and transform all datasets
X_train_reduced = svd.fit_transform(X_train)
X_val_reduced = svd.transform(X_val)
X_test_reduced = svd.transform(X_test)

svd_time = time.time() - start_time
print(f"SVD dimensionality reduction completed (Time: {svd_time:.2f}s)")
print(
    f"Reduced feature dimensions: Train={X_train_reduced.shape}, Val={X_val_reduced.shape}, Test={X_test_reduced.shape}")
print(f"Total explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}\n")

# -------------------------- 7. Logistic Regression Model Training with Grid Search --------------------------
print("Starting Logistic Regression training with grid search...")
start_time = time.time()

# Initialize Logistic Regression (optimized for Full hierarchy multi-class classification)
logistic_model = LogisticRegression(
    random_state = SEED,
    max_iter = 500,  # Reduced iterations for Full hierarchy efficiency (prevents long training)
    class_weight = "balanced",  # Critical: Address class imbalance in Full hierarchy
    solver = "saga"  # Optimized solver for multi-class classification with L1/L2 regularization
)

# Grid search parameters (simplified for Full hierarchy efficiency)
param_grid = {
    "C": [1.0]  # Regularization strength (simplified to single value for faster training)
}

# Initialize Grid Search with cross-validation
grid_search = GridSearchCV(
    estimator = logistic_model,
    param_grid = param_grid,
    cv = CV_FOLDS,
    scoring = "f1_macro",  # Appropriate metric for imbalanced multi-class classification
    verbose = 2,
    n_jobs = 1  # Single thread to prevent memory issues with Full hierarchy
)

# Calculate total parameter combinations for progress tracking
total_combinations = len(param_grid["C"])
print(f"Grid search initialized (CV={CV_FOLDS}, parameters={total_combinations} combinations)...")

# Train model on reduced features
grid_search.fit(X_train_reduced, y_train)

# Calculate training time
train_time = time.time() - start_time
print(f"\nModel training completed (Total Time: {train_time:.2f}s)")
print(f"Best hyperparameters: {grid_search.best_params_}")
print(f"Best cross-validation F1-macro score: {grid_search.best_score_:.4f}")

# Extract best model from grid search
best_model = grid_search.best_estimator_

# -------------------------- 8. Save Trained Model and Transformers --------------------------
print(f"\nSaving trained model and transformers to {MODEL_SAVE_DIR}...")

# Save the best logistic regression model
model_save_path = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME)
joblib.dump(best_model, model_save_path)
print(f"✅ Trained model saved to: {model_save_path}")

# Save TF-IDF vectorizer (required for future predictions)
tfidf_save_path = os.path.join(MODEL_SAVE_DIR, TFIDF_FILENAME)
joblib.dump(tfidf_vectorizer, tfidf_save_path)
print(f"✅ TF-IDF vectorizer saved to: {tfidf_save_path}")

# Save SVD transformer (required for future predictions)
svd_save_path = os.path.join(MODEL_SAVE_DIR, SVD_FILENAME)
joblib.dump(svd, svd_save_path)
print(f"✅ SVD transformer saved to: {svd_save_path}")

# -------------------------- 9. Model Evaluation on Test Set --------------------------
print("\nStarting model evaluation on test set...")
start_time = time.time()

# Generate predictions on test set (using reduced features)
y_test_pred = best_model.predict(tqdm(X_test_reduced, desc = "Predicting on test set"))

eval_time = time.time() - start_time
print(f"\n===== Test Set Evaluation Results (Time: {eval_time:.2f}s) =====")
print(f"Overall Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

# Print detailed classification report
print("\nClassification Report (Precision/Recall/F1-Score):")
print(classification_report(y_test, y_test_pred, zero_division = 0))

# Generate and print confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
# Get sorted unique labels for consistent matrix formatting
unique_labels = sorted(Counter(y_test).keys())
conf_matrix_df = pd.DataFrame(
    conf_matrix,
    index = unique_labels,
    columns = unique_labels
)
print("\nConfusion Matrix:")
print(conf_matrix_df)


# -------------------------- 10. Prediction Function for New Utterances --------------------------
def predict_dialogue_act(text, model, tfidf, svd, preprocess_func):
    """
    Predict dialogue act label for new/ unseen utterance text.

    Args:
        text (str): Raw input utterance text
        model: Trained Logistic Regression model
        tfidf: Fitted TF-IDF vectorizer
        svd: Fitted Truncated SVD transformer
        preprocess_func: Text preprocessing function

    Returns:
        str: Predicted dialogue act label
    """
    # Complete preprocessing pipeline
    processed_text = preprocess_func(text)

    # Transform text to TF-IDF features
    text_tfidf = tfidf.transform([processed_text])

    # Apply SVD dimensionality reduction
    text_reduced = svd.transform(text_tfidf)

    # Predict dialogue act label
    predicted_label = model.predict(text_reduced)[0]

    return predicted_label


# -------------------------- 11. Example Predictions --------------------------
# Test prediction with sample utterances
test_utterances = [
    "Can you share the project documentation with me?",
    "The deadline for the report is tomorrow afternoon.",
    "Alright, let's discuss the next agenda item now."
]

print("\n===== Example Predictions =====")
for utterance in test_utterances:
    pred_label = predict_dialogue_act(utterance, best_model, tfidf_vectorizer, svd, preprocess_text)
    print(f"Utterance: {utterance}")
    print(f"Predicted DA Label: {pred_label}\n")


# -------------------------- 12. Model Loading Function (for future use) --------------------------
def load_trained_models():
    """
    Load saved Logistic Regression model, TF-IDF vectorizer and SVD transformer.

    Returns:
        tuple: (model, tfidf, svd) - Loaded model and transformers
    """
    model = joblib.load(os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME))
    tfidf = joblib.load(os.path.join(MODEL_SAVE_DIR, TFIDF_FILENAME))
    svd = joblib.load(os.path.join(MODEL_SAVE_DIR, SVD_FILENAME))
    return model, tfidf, svd

# Example usage (commented out - uncomment to test loading)
# loaded_model, loaded_tfidf, loaded_svd = load_trained_models()
# sample_prediction = predict_dialogue_act("What time is the meeting?", loaded_model, loaded_tfidf, loaded_svd, preprocess_text)
# print(f"Loaded model prediction: {sample_prediction}")