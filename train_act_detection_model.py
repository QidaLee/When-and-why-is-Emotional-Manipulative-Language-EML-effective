import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
import torch
from torch import nn
import warnings

warnings.filterwarnings('ignore')

# ==================================================
# 1. Load Data
# ==================================================
print("=" * 50)
print("Loading Data")
print("=" * 50)

# TODO: Change to your actual file name
df = pd.read_csv('./data/300_dialog_with_result.csv')

print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# ==================================================
# 2. Select Features and Labels
# ==================================================
# Text column: 'Unit'
# Label column: 'persuasion_result'
texts = df['Unit'].astype(str).tolist()
labels_raw = df['persuasion_result'].tolist()

# ==================================================
# 3. Encode Labels
# ==================================================
unique_labels = sorted(list(set(labels_raw)))
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

labels = [label2id[label] for label in labels_raw]

print(f"\nLabel mapping: {label2id}")
print(f"Reverse mapping: {id2label}")

# ==================================================
# 4. Class Distribution
# ==================================================
print("\n" + "=" * 50)
print("Class Distribution")
print("=" * 50)
class_counts = pd.Series(labels_raw).value_counts()
print(class_counts)
print("\nPercentages:")
print(pd.Series(labels_raw).value_counts(normalize = True) * 100)

# ==================================================
# 5. Compute Class Weights (Handle Imbalance)
# ==================================================
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes = np.unique(labels),
    y = labels
)
class_weights = torch.tensor(class_weights, dtype = torch.float32)
print(f"\nClass weights: {class_weights}")

# ==================================================
# 6. Train/Validation Split
# ==================================================
print("\n" + "=" * 50)
print("Splitting Data")
print("=" * 50)

X_train, X_val, y_train, y_val = train_test_split(
    texts,
    labels,
    test_size = 0.2,
    random_state = 42,
    stratify = labels
)

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")

# ==================================================
# 7. Create Hugging Face Dataset
# ==================================================
train_df = pd.DataFrame({'text': X_train, 'label': y_train})
val_df = pd.DataFrame({'text': X_val, 'label': y_val})

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# ==================================================
# 8. Load Tokenizer and Model
# ==================================================
print("\n" + "=" * 50)
print("Loading Model")
print("=" * 50)

# Using smaller model for faster training, can be changed
model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels = len(unique_labels),
    id2label = id2label,
    label2id = label2id
)


# ==================================================
# 9. Tokenization Function
# ==================================================
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding = 'max_length',
        truncation = True,
        max_length = 128  # Adjust based on your text length
    )


print("\nTokenizing datasets...")
train_dataset = train_dataset.map(tokenize_function, batched = True)
val_dataset = val_dataset.map(tokenize_function, batched = True)

# Set format for PyTorch
train_dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask', 'label'])


# ==================================================
# 10. Custom Trainer with Class Weights
# ==================================================
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Get device from model parameters (works with DataParallel)
        device = next(model.parameters()).device
        loss_fct = nn.CrossEntropyLoss(weight = class_weights.to(device))

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ==================================================
# 11. Training Arguments
# ==================================================
training_args = TrainingArguments(
    output_dir = './results',
    num_train_epochs = 15,
    learning_rate = 1e-5,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 16,
    warmup_ratio = 0.2,
    weight_decay = 0.01,
    logging_steps = 50,
    eval_strategy = 'epoch',
    save_strategy = 'epoch',
    load_best_model_at_end = True,
    metric_for_best_model = 'eval_f1_macro',
    greater_is_better = True,
    save_total_limit = 2,
    remove_unused_columns = False,
    report_to = 'none'
)

callbacks = [EarlyStoppingCallback(early_stopping_patience = 5)]

# ==================================================
# 12. Evaluation Metrics
# ==================================================
from sklearn.metrics import accuracy_score, f1_score, classification_report


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis = 1)

    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average = 'macro')
    f1_weighted = f1_score(labels, predictions, average = 'weighted')

    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(labels, predictions, target_names = unique_labels))

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }


# ==================================================
# 13. Initialize Trainer
# ==================================================
trainer = WeightedTrainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,
    compute_metrics = compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience = 3)]
)

# ==================================================
# 14. Train
# ==================================================
print("\n" + "=" * 50)
print("Starting Training")
print("=" * 50)

trainer.train()

# ==================================================
# 15. Final Evaluation
# ==================================================
print("\n" + "=" * 50)
print("Final Evaluation")
print("=" * 50)

eval_results = trainer.evaluate()
print(f"\nEvaluation results: {eval_results}")

# ==================================================
# 16. Save Model
# ==================================================
model_save_path = './persuasion_model'
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"\nModel saved to: {model_save_path}")

# ==================================================
# 17. Prediction Examples
# ==================================================
print("\n" + "=" * 50)
print("Example Predictions")
print("=" * 50)


def predict(text):
    # Tokenize
    inputs = tokenizer(text, return_tensors = 'pt', truncation = True, max_length = 128, padding = True)

    # Get model device and move inputs to same device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim = -1)
        predicted_class_id = predictions.argmax().item()

    # Move to CPU for output
    predictions = predictions.cpu()

    return id2label[predicted_class_id], predictions[0].tolist()


# Test a few validation samples
for i in range(min(5, len(X_val))):
    text = X_val[i]
    true_label = id2label[y_val[i]]
    pred_label, probs = predict(text)
    print(f"\nText: {text[:100]}..." if len(text) > 100 else f"\nText: {text}")
    print(f"True label: {true_label}")
    print(f"Predicted label: {pred_label}")
    print(f"Probability distribution: {dict(zip(unique_labels, [round(p, 3) for p in probs]))}")
