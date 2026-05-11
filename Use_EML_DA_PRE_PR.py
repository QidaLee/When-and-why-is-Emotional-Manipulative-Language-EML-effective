# Import required libraries
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import ast
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

# Set plot style for English display
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# ===================== Fixed 12 DA Categories =====================
FIXED_DA_LIST = ["ap", "ac", "dc", "do", "il", "ic",
                 "in", "pa", "pr", "sp", "sn", "o"]
DA_COLUMNS = [f"DA_{da}" for da in FIXED_DA_LIST]

# ===================== 1. Load Dataset =====================
DATA_PATH = "./output_data/all_turns_data_with_EML_DA_PR.xlsx"
df = pd.read_excel(DATA_PATH)

# Show basic dataset info
print("Dataset shape:", df.shape)
print("\nPR_label distribution:\n", df["PR_label"].value_counts())
print("\nMulti_DA_Labels raw sample:")
print(df["Multi_DA_Labels"].head(6).tolist())


# ===================== 2. Parse Multi_DA_Labels to Fixed 12-Dim 0/1 Matrix =====================
def parse_da_binary(da_str):
    """Convert list-string or comma-separated DA to fixed 12-dim 0/1 vector"""
    vec = [0] * len(FIXED_DA_LIST)
    if pd.isna(da_str):
        return vec

    s = str(da_str).strip()
    if s in ["", "[]", "nan"]:
        return vec

    # Parse list string like "['in','o','pa']"
    try:
        da_tags = ast.literal_eval(s)
    except:
        # Fallback for plain comma-separated format
        da_tags = [x.strip().strip("'\"[]") for x in s.split(",") if x.strip()]

    # Assign 1 to matched DA
    for tag in da_tags:
        tag = tag.strip().lower()
        if tag in FIXED_DA_LIST:
            idx = FIXED_DA_LIST.index(tag)
            vec[idx] = 1
    return vec


# Generate 12 DA binary columns
da_binary_arr = df["Multi_DA_Labels"].apply(parse_da_binary)
da_df = pd.DataFrame(da_binary_arr.tolist(), columns = DA_COLUMNS)

# Merge 12 DA columns back to original dataframe
df = pd.concat([df.reset_index(drop = True), da_df.reset_index(drop = True)], axis = 1)

# ===================== 3. Data Preprocessing =====================
target_col = "PR_label"

# Clean EML_label
df["EML_label"] = pd.to_numeric(
    df["EML_label"].astype(str).str.replace(r"[^0-9]", "", regex = True),
    errors = "coerce"
).fillna(0).astype(int)

# Drop rows with missing critical values
df = df.dropna(subset = ["Sentence", target_col])

# Define feature columns
text_feature = ["Sentence"]
numeric_features = ["EML_label"] + DA_COLUMNS
all_features = text_feature + numeric_features

X = df[all_features]
y = df[target_col]

# Encode target label
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ===================== 4. Train-Test Split =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size = 0.2, random_state = 42, stratify = y_encoded
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Fixed 12 DA columns generated: {len(DA_COLUMNS)}")

# ===================== 5. Feature Pipeline =====================
# Text processing pipeline
text_pipeline = Pipeline(steps = [
    ("tfidf", TfidfVectorizer(
        max_features = 5000,
        stop_words = "english",
        ngram_range = (1, 2)
    ))
])

# Numeric processing pipeline (EML + 12 DA 0/1 columns)
numeric_pipeline = Pipeline(steps = [
    ("imputer", SimpleImputer(strategy = "constant", fill_value = 0))
])

# Combine preprocessors
preprocessor = ColumnTransformer(
    transformers = [
        ("text", text_pipeline, "Sentence"),
        ("num", numeric_pipeline, numeric_features)
    ]
)

# ===================== 6. Build XGBoost Model =====================
xgb_model = XGBClassifier(
    n_estimators = 150,
    max_depth = 6,
    learning_rate = 0.1,
    random_state = 42,
    use_label_encoder = False,
    eval_metric = "mlogloss"
)

# Full pipeline
model_pipeline = Pipeline(steps = [
    ("preprocessor", preprocessor),
    ("classifier", xgb_model)
])

# ===================== 7. Model Training =====================
print("\nStarting XGBoost model training...")
try:
    model_pipeline.fit(X_train, y_train)
    print("Model training completed successfully!")
except Exception as e:
    print(f"Training error: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

# ===================== 8. Model Evaluation =====================
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

# ===================== 9. Save Model & Encoder =====================
joblib.dump(model_pipeline, "pr_xgb_model.pkl")
joblib.dump(label_encoder, "pr_label_encoder.pkl")
print("\nModel saved as: pr_xgb_model.pkl")
print("Label encoder saved as: pr_label_encoder.pkl")

# ===================== 10. Feature Importance =====================
print("\n" + "=" * 50)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 50)

feature_importance_df = None
try:
    preprocessor_fitted = model_pipeline.named_steps["preprocessor"]
    classifier = model_pipeline.named_steps["classifier"]

    # Get feature names
    text_feat_names = preprocessor_fitted.named_transformers_["text"].named_steps["tfidf"].get_feature_names_out()
    all_feat_names = list(text_feat_names) + numeric_features

    # Get importance
    if hasattr(classifier, "feature_importances_"):
        feat_imp = classifier.feature_importances_
        feature_importance_df = pd.DataFrame({
            "feature": all_feat_names[:len(feat_imp)],
            "importance": feat_imp
        }).sort_values("importance", ascending = False)

        # Show DA + EML importance
        da_eml_imp = feature_importance_df[
            feature_importance_df["feature"].isin(numeric_features)
        ].sort_values("importance", ascending = False)

        print("\nEML & 12 DA Features Importance:")
        print(da_eml_imp.to_string(index = False))

except Exception as e:
    print(f"Failed to compute feature importance: {e}")

# ===================== 11. Horizontal Bar Plot: 12 DA + EML =====================
print("\n" + "=" * 50)
print("PLOT 12 DA TYPES & EML FEATURE IMPORTANCE")
print("=" * 50)

try:
    if feature_importance_df is None:
        raise ValueError("Feature importance data not available")

    # Extract exactly 12 DA + EML
    plot_df = feature_importance_df[
        feature_importance_df["feature"].isin(numeric_features)
    ].sort_values("importance", ascending = True)

    # Plot horizontal bar chart
    plt.figure(figsize = (11, 7))
    colors = ["#1f77b4" if x == "EML_label" else "#ff7f0e" for x in plot_df["feature"]]
    plt.barh(plot_df["feature"], plot_df["importance"], color = colors)

    plt.xlabel("Feature Importance Score", fontsize = 12)
    plt.ylabel("Feature Name", fontsize = 12)
    plt.title("Feature Importance: 12 DA Types and EML Label", fontsize = 14, pad = 20)
    plt.grid(axis = "x", alpha = 0.3)
    plt.tight_layout()

    plt.savefig("DA_EML_Feature_Importance.png", dpi = 300, bbox_inches = "tight")
    plt.show()
    print("Plot saved as: DA_EML_Feature_Importance.png")

except Exception as e:
    print(f"Plot generation failed: {e}")


# ===================== 12. Prediction Demo =====================
def predict_pr(sentence: str, eml_label: int, multi_da_labels: str):
    """Predict PR label with raw input sentence, EML and multi-DA string"""
    # Build temporary input row
    temp_df = pd.DataFrame({"Sentence": [sentence], "EML_label": [eml_label]})
    # Parse DA to 12 binary columns
    da_vec = parse_da_binary(multi_da_labels)
    for idx, col in enumerate(DA_COLUMNS):
        temp_df[col] = da_vec[idx]
    pred_idx = model_pipeline.predict(temp_df)[0]
    return label_encoder.inverse_transform([pred_idx])[0]


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("PREDICTION DEMO EXAMPLES")
    print("=" * 50)

    try:
        # Demo 1
        pred1 = predict_pr("Hello. How are you?", 0, "['o','pa']")
        print("\nExample 1:")
        print("  Sentence: Hello. How are you?")
        print("  Predicted PR Label:", pred1)

        # Demo 2
        pred2 = predict_pr("I completely disagree with your proposal.", 1, "['pr','ic']")
        print("\nExample 2:")
        print("  Sentence: I completely disagree with your proposal.")
        print("  Predicted PR Label:", pred2)

    except Exception as e:
        print(f"Prediction demo failed: {e}")