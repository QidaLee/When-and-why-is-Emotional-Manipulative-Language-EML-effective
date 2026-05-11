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
from sklearn.multioutput import MultiOutputClassifier

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
print("\nMulti_DA_Labels raw sample:")
print(df["Multi_DA_Labels"].head(6).tolist())
print("\nUnique PR labels in dataset:", df["PR_label"].unique())


# ===================== 2. Parse Multi_DA_Labels to Fixed 12-Dim 0/1 Matrix (Target) =====================
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


# Generate 12 DA binary columns as multi-label target
da_binary_arr = df["Multi_DA_Labels"].apply(parse_da_binary)
da_target_df = pd.DataFrame(da_binary_arr.tolist(), columns = DA_COLUMNS)

# Merge DA target columns back to original dataframe
df = pd.concat([df.reset_index(drop = True), da_target_df.reset_index(drop = True)], axis = 1)

# ===================== 3. Data Preprocessing =====================
# Target: 12 DA multi-label columns
target_cols = DA_COLUMNS

# Clean EML_label
df["EML_label"] = pd.to_numeric(
    df["EML_label"].astype(str).str.replace(r"[^0-9]", "", regex = True),
    errors = "coerce"
).fillna(0).astype(int)

# Drop rows with missing critical values
df = df.dropna(subset = ["Sentence", "EML_label", "PR_label"]).copy()

# Features: Sentence + EML_label + PR_label
text_feature = ["Sentence"]
numeric_feature = ["EML_label"]
categorical_feature = ["PR_label"]
all_features = text_feature + numeric_feature + categorical_feature

# Create clean feature set
X = df[all_features].copy()
y = df[target_cols].copy()

# Encode PR_label
pr_encoder = LabelEncoder()
X["PR_label"] = pr_encoder.fit_transform(X["PR_label"])

# Save real PR classes for demo
pr_classes = pr_encoder.classes_.tolist()

# ===================== 4. Train-Test Split =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Multi-label target: {len(DA_COLUMNS)} DA types")

# ===================== 5. Feature Pipeline =====================
# Text processing pipeline
text_pipeline = Pipeline(steps = [
    ("tfidf", TfidfVectorizer(
        max_features = 5000,
        stop_words = "english",
        ngram_range = (1, 2)
    ))
])

# Numeric pipeline for EML_label
numeric_pipeline = Pipeline(steps = [
    ("imputer", SimpleImputer(strategy = "constant", fill_value = 0))
])

# Categorical pipeline for PR_label
cat_pipeline = Pipeline(steps = [
    ("imputer", SimpleImputer(strategy = "most_frequent"))
])

# Combine preprocessors
preprocessor = ColumnTransformer(
    transformers = [
        ("text", text_pipeline, "Sentence"),
        ("num", numeric_pipeline, numeric_feature),
        ("cat", cat_pipeline, categorical_feature)
    ]
)

# ===================== 6. Build Multi-Output XGBoost Model =====================
base_xgb = XGBClassifier(
    n_estimators = 150,
    max_depth = 6,
    learning_rate = 0.1,
    random_state = 42,
    eval_metric = "logloss"
)

# Multi-output for 12 DA labels
multi_xgb_model = MultiOutputClassifier(base_xgb)

# Full pipeline
model_pipeline = Pipeline(steps = [
    ("preprocessor", preprocessor),
    ("classifier", multi_xgb_model)
])

# ===================== 7. Model Training =====================
print("\nStarting XGBoost multi-label training (EML+PR -> DA)...")
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

print("\n" + "=" * 60)
print("MULTI-LABEL EVALUATION METRICS (STANDARD)")
print("=" * 60)

# 1. Label-wise Accuracy
label_acc = np.mean([accuracy_score(y_test.iloc[:,i], y_pred[:,i]) for i in range(len(DA_COLUMNS))])

# 2. Exact Match Accuracy
exact_match = accuracy_score(y_test.values, y_pred)


print(f"Label-wise Accuracy (12 DA avg): {label_acc:.4f}")
print(f"Exact Match Accuracy (all 12 correct): {exact_match:.4f}")


# ===================== 9. Save Model & Encoder =====================
joblib.dump(model_pipeline, "da_multi_xgb_model.pkl")
joblib.dump(pr_encoder, "pr_label_encoder.pkl")
print("\nModel saved as: da_multi_xgb_model.pkl")
print("PR label encoder saved as: pr_label_encoder.pkl")

# ===================== 10. Feature Importance =====================
print("\n" + "=" * 50)
print("FEATURE IMPORTANCE ANALYSIS (EML + PR)")
print("=" * 50)

feature_importance_df = None
try:
    preprocessor_fitted = model_pipeline.named_steps["preprocessor"]
    classifier = model_pipeline.named_steps["classifier"].estimators_[0]

    # Get feature names
    text_feat_names = preprocessor_fitted.named_transformers_["text"].named_steps["tfidf"].get_feature_names_out()
    all_feat_names = list(text_feat_names) + ["EML_label", "PR_label"]

    if hasattr(classifier, "feature_importances_"):
        feat_imp = classifier.feature_importances_
        feature_importance_df = pd.DataFrame({
            "feature": all_feat_names[:len(feat_imp)],
            "importance": feat_imp
        }).sort_values("importance", ascending = False)

        # Only show EML and PR
        main_feats = ["EML_label", "PR_label"]
        main_imp = feature_importance_df[
            feature_importance_df["feature"].isin(main_feats)
        ].sort_values("importance", ascending = False)

        print("\nEML & PR Feature Importance for DA Prediction:")
        print(main_imp.to_string(index = False))

except Exception as e:
    print(f"Failed to compute feature importance: {e}")

# ===================== 11. Horizontal Bar Plot: EML + PR Importance =====================
print("\n" + "=" * 50)
print("PLOT EML & PR FEATURE IMPORTANCE")
print("=" * 50)

try:
    if feature_importance_df is None:
        raise ValueError("Feature importance data not available")

    plot_df = feature_importance_df[
        feature_importance_df["feature"].isin(["EML_label", "PR_label"])
    ].sort_values("importance", ascending = True)

    plt.figure(figsize = (9, 4))
    colors = ["#2E86AB", "#A23B72"]
    plt.barh(plot_df["feature"], plot_df["importance"], color = colors)

    plt.xlabel("Feature Importance Score", fontsize = 12)
    plt.ylabel("Feature Name", fontsize = 12)
    plt.title("Feature Importance: EML & PR for DA Prediction", fontsize = 14, pad = 20)
    plt.grid(axis = "x", alpha = 0.3)
    plt.tight_layout()

    plt.savefig("EML_PR_DA_Feature_Importance.png", dpi = 300, bbox_inches = "tight")
    plt.show()
    print("Plot saved as: EML_PR_DA_Feature_Importance.png")

except Exception as e:
    print(f"Plot generation failed: {e}")


# ===================== 12. Prediction Demo =====================
def predict_da(sentence: str, eml_label: int, pr_label):
    """Predict 12 DA multi-labels from Sentence + EML + PR"""
    temp_df = pd.DataFrame({
        "Sentence": [sentence],
        "EML_label": [eml_label],
        "PR_label": [pr_label]
    })
    temp_df["PR_label"] = pr_encoder.transform(temp_df["PR_label"])

    pred_vec = model_pipeline.predict(temp_df)[0]
    pred_da_list = [FIXED_DA_LIST[i] for i, val in enumerate(pred_vec) if val == 1]
    return pred_da_list, pred_vec


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("DA MULTI-LABEL PREDICTION DEMO")
    print("=" * 50)

    try:
        # Use REAL PR label from dataset
        sample_pr = pr_classes[0]
        da_list1, _ = predict_da("Hello. How are you?", 0, sample_pr)
        print("\nExample 1:")
        print("  Sentence: Hello. How are you?")
        print(f"  EML: 0 , PR: {sample_pr}")
        print("  Predicted DA Labels:", da_list1)

        # Second example
        if len(pr_classes) > 1:
            sample_pr2 = pr_classes[1]
            da_list2, _ = predict_da("I disagree with this.", 1, sample_pr2)
            print("\nExample 2:")
            print("  Sentence: I disagree with this.")
            print(f"  EML: 1 , PR: {sample_pr2}")
            print("  Predicted DA Labels:", da_list2)

    except Exception as e:
        print(f"Prediction demo failed: {e}")