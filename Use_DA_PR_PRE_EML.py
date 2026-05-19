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
print("\nEML_label distribution:\n", df["EML_label"].value_counts())
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
da_df = pd.DataFrame(da_binary_arr.tolist(), columns=DA_COLUMNS)

# Merge 12 DA columns back to original dataframe
df = pd.concat([df.reset_index(drop=True), da_df.reset_index(drop=True)], axis=1)

# ===================== 3. Data Preprocessing =====================
# NEW TARGET: Predict EML_label
target_col = "EML_label"

# Clean target (EML)
df[target_col] = pd.to_numeric(
    df[target_col].astype(str).str.replace(r"[^0-9]", "", regex=True),
    errors="coerce"
).fillna(0).astype(int)

# Drop rows with missing critical values
df = df.dropna(subset=["Sentence", "PR_label", target_col])

# FEATURES: Sentence + PR_label + 12 DA columns
text_feature = ["Sentence"]
categorical_features = ["PR_label"]
numeric_features = DA_COLUMNS
all_features = text_feature + categorical_features + numeric_features

X = df[all_features]
y = df[target_col]

# Encode PR_label (categorical)
pr_encoder = LabelEncoder()
X["PR_label"] = pr_encoder.fit_transform(X["PR_label"])

# Encode target (EML is already numeric)
print("\nEML classes:", sorted(y.unique()))
print("\nReal PR classes in dataset:", pr_encoder.classes_.tolist())

# ===================== 4. Train-Test Split =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Fixed 12 DA columns generated: {len(DA_COLUMNS)}")

# ===================== 5. Feature Pipeline =====================
# Text processing pipeline
text_pipeline = Pipeline(steps=[
    ("tfidf", TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2)
    ))
])

# Numeric pipeline: 12 DA binary features
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value=0))
])

# Categorical pipeline: PR_label
cat_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

# Combine preprocessors
preprocessor = ColumnTransformer(
    transformers=[
        ("text", text_pipeline, "Sentence"),
        ("cat", cat_pipeline, ["PR_label"]),
        ("num", numeric_pipeline, numeric_features)
    ]
)

# ===================== 6. Build XGBoost Model =====================
xgb_model = XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric="mlogloss"
)

# Full pipeline
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", xgb_model)
])

# ===================== 7. Model Training =====================
print("\nStarting XGBoost model training (Predict EML)...")
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
print("MODEL EVALUATION RESULTS (Predict EML)")
print("=" * 50)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    zero_division=0
))

# ===================== 9. Save Model & Encoders =====================
joblib.dump(model_pipeline, "eml_xgb_model.pkl")
joblib.dump(pr_encoder, "pr_label_encoder.pkl")
print("\nModel saved as: eml_xgb_model.pkl")
print("PR encoder saved as: pr_label_encoder.pkl")

# ===================== 10. FULL Feature Importance Analysis + Save to CSV =====================
print("\n" + "=" * 50)
print("FULL FEATURE IMPORTANCE ANALYSIS")
print("=" * 50)

full_feature_importance_df = None
try:
    # Get fitted pipeline components
    preprocessor_fitted = model_pipeline.named_steps["preprocessor"]
    classifier = model_pipeline.named_steps["classifier"]

    # Get all feature names
    text_transformer = preprocessor_fitted.named_transformers_["text"]
    text_feat_names = text_transformer.named_steps["tfidf"].get_feature_names_out()
    cat_feat_names = ["PR_label"]
    num_feat_names = numeric_features
    all_feat_names = list(text_feat_names) + cat_feat_names + num_feat_names

    # Get feature importance scores
    if hasattr(classifier, "feature_importances_"):
        feat_imp_scores = classifier.feature_importances_
        min_length = min(len(all_feat_names), len(feat_imp_scores))
        all_feat_names = all_feat_names[:min_length]
        feat_imp_scores = feat_imp_scores[:min_length]

        full_feature_importance_df = pd.DataFrame({
            "feature_name": all_feat_names,
            "importance_score": feat_imp_scores
        }).sort_values("importance_score", ascending=False).reset_index(drop=True)

        full_feature_importance_df["rank"] = full_feature_importance_df.index + 1

        # Save full feature importance to CSV
        full_feature_importance_df.to_csv("full_feature_importance_rank.csv", index=False, encoding="utf-8-sig")
        print("✅ Full feature importance saved to full_feature_importance_rank.csv")

        # Show top 20
        print("\n📊 Top 20 Most Important Features:")
        print(full_feature_importance_df.head(20).to_string(index=False))

        # Show PR + DA
        da_pr_df = full_feature_importance_df[
            full_feature_importance_df["feature_name"].isin(["PR_label"] + DA_COLUMNS)
        ].sort_values("importance_score", ascending=False)
        print("\n🔍 PR & DA Features Importance:")
        print(da_pr_df.to_string(index=False))

except Exception as e:
    print(f"❌ Feature importance error: {e}")

# ===================== 11. Plot =====================
try:
    da_pr_plot = full_feature_importance_df[
        full_feature_importance_df["feature_name"].isin(["PR_label"] + DA_COLUMNS)
    ].sort_values("importance_score", ascending=True)

    plt.figure(figsize=(11, 7))
    colors = ["#1f77b4" if x == "PR_label" else "#ff7f0e" for x in da_pr_plot["feature_name"]]
    plt.barh(da_pr_plot["feature_name"], da_pr_plot["importance_score"], color=colors)
    plt.xlabel("Feature Importance")
    plt.title("DA & PR Importance for EML Prediction")
    plt.tight_layout()
    plt.savefig("DA_PR_EML_Feature_Importance.png", dpi=300)
    print("✅ Plot saved")
except:
    print("⚠️ Plot skipped")

# ===================== 12. FIXED Prediction Demo =====================
def predict_eml(sentence: str, pr_label: str, multi_da_labels: str):
    temp_df = pd.DataFrame({"Sentence": [sentence], "PR_label": [pr_label]})
    temp_df["PR_label"] = pr_encoder.transform(temp_df["PR_label"])
    da_vec = parse_da_binary(multi_da_labels)
    for i, col in enumerate(DA_COLUMNS):
        temp_df[col] = da_vec[i]
    return int(model_pipeline.predict(temp_df)[0])

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("PREDICTION DEMO (Fixed with REAL PR labels)")
    print("=" * 50)

    # ✅ USE REAL PR LABELS: Compliance, Neutral, Resistance
    pred1 = predict_eml("Hello, good morning!", "Neutral", "['o','pa']")
    print("\nExample 1:")
    print("  Sentence: Hello, good morning!")
    print("  PR Label: Neutral")
    print("  DA Labels: ['o','pa']")
    print("  Predicted EML:", pred1)

    pred2 = predict_eml("I disagree with that.", "Resistance", "['pr','ic']")
    print("\nExample 2:")
    print("  Sentence: I disagree with that.")
    print("  PR Label: Resistance")
    print("  DA Labels: ['pr','ic']")
    print("  Predicted EML:", pred2)