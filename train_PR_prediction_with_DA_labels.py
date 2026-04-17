# Import required libraries
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

# ===================== 1. Load Dataset =====================
DATA_PATH = "./output_data/all_turns_data_with_EML_DA_PR.xlsx"
df = pd.read_excel(DATA_PATH)

# Show basic dataset info
print("Dataset shape:", df.shape)
print("\nPR_label distribution:\n", df["PR_label"].value_counts())

# ===================== 2. Data Preprocessing =====================
# Select input features and target label
selected_features = ["Sentence", "EML_label", "DA_label"]
target = "PR_label"

# Clean EML_label (convert to numeric, remove non-digit characters)
df["EML_label"] = pd.to_numeric(df["EML_label"].astype(str).str.replace(r"[^0-9]", "", regex = True),
                                errors = "coerce").fillna(0).astype(int)

# Drop rows with missing critical values
df = df.dropna(subset = selected_features + [target])

# ===================== 3. Train-Test Split =====================
X = df[selected_features]
y = df[target]

# Encode target labels (string -> numeric)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Stratified split to preserve label distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size = 0.2, random_state = 42, stratify = y_encoded
)

# ===================== 4. Feature Engineering Pipeline =====================
# Text feature processing (Sentence)
text_processor = Pipeline(steps = [
    ("tfidf", TfidfVectorizer(
        max_features = 5000,
        stop_words = "english",
        ngram_range = (1, 2)
    ))
])

# Categorical feature processing (DA_label)
categorical_processor = Pipeline(steps = [
    ("imputer", SimpleImputer(strategy = "most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown = "ignore"))
])

# Numeric feature processing (EML_label)
numeric_processor = Pipeline(steps = [
    ("imputer", SimpleImputer(strategy = "constant", fill_value = 0))
])

# Combine all feature processors
preprocessor = ColumnTransformer(
    transformers = [
        ("text", text_processor, "Sentence"),
        ("cat", categorical_processor, ["DA_label"]),
        ("num", numeric_processor, ["EML_label"])
    ])

# ===================== 5. Build XGBoost Model =====================
# XGBoost classifier (optimized for classification)
xgb_model = XGBClassifier(
    n_estimators = 150,
    max_depth = 6,
    learning_rate = 0.1,
    random_state = 42,
    use_label_encoder = False,
    eval_metric = "mlogloss"
)

# Full training pipeline
model_pipeline = Pipeline(steps = [
    ("preprocessor", preprocessor),
    ("classifier", xgb_model)
])

# ===================== 6. Model Training =====================
print("\nStarting model training with XGBoost...")
model_pipeline.fit(X_train, y_train)

# ===================== 7. Model Evaluation =====================
y_pred = model_pipeline.predict(X_test)

print("\n" + "=" * 50)
print("MODEL EVALUATION RESULTS")
print("=" * 50)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names = label_encoder.classes_,
    zero_division = 0
))

# ===================== 8. Save Model & Encoder =====================
joblib.dump(model_pipeline, "pr_xgb_model.pkl")
joblib.dump(label_encoder, "pr_label_encoder.pkl")
print("\nModel saved as: pr_xgb_model.pkl")
print("Label encoder saved as: pr_label_encoder.pkl")


# ===================== 9. Single Sample Prediction =====================
def predict_pr(sentence: str, eml_label: int, da_label: str):
    """Predict PR label for a single dialogue turn"""
    input_data = pd.DataFrame({
        "Sentence": [sentence],
        "EML_label": [eml_label],
        "DA_label": [da_label]
    })
    pred_idx = model_pipeline.predict(input_data)[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    return pred_label


# Demo prediction
if __name__ == "__main__":
    test_sentence = "Hello. How are you?"
    test_eml = 0
    test_da = "o"
    pred_result = predict_pr(test_sentence, test_eml, test_da)

    print("\nPrediction Example:")
    print(f"Input Sentence: {test_sentence}")
    print(f"EML Label: {test_eml}, DA Label: {test_da}")
    print(f"Predicted PR Label: {pred_result}")