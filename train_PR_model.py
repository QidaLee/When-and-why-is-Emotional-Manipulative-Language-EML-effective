import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    AdamW
)
from datasets import Dataset
import torch
from torch import nn
import warnings
import random
import os

warnings.filterwarnings('ignore')

# 固定随机种子，保证结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(42)

# ==================================================
# 1. Load Data
# ==================================================
print("=" * 50)
print("Loading Data")
print("=" * 50)

df = pd.read_excel('./data/Persuasion_For_Good/all_turns_data_with_PR_label.xlsx')

print(f"Data shape: {df.shape}")

# ==================================================
# 2. Select Features and Labels
# ==================================================
texts = df['Sentence'].astype(str).tolist()
labels_raw = df['persuasion_result'].tolist()

# ==================================================
# 3. Encode Labels
# ==================================================
unique_labels = sorted(list(set(labels_raw)))
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

labels = [label2id[label] for label in labels_raw]

print(f"\nLabel mapping: {label2id}")

# ==================================================
# 4. Class Weights
# ==================================================
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# ==================================================
# 6. Train/Validation Split
# ==================================================
X_train, X_val, y_train, y_val = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# ==================================================
# 7. Dataset
# ==================================================
train_df = pd.DataFrame({'text': X_train, 'label': y_train})
val_df = pd.DataFrame({'text': X_val, 'label': y_val})

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# ==================================================
# 8. Tokenizer & Model (更强的模型)
# ==================================================
print("\nLoading stronger model...")

# ✅ 换成更强的 base 模型（比 distilbert 强很多）
model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id
)

# ==================================================
# 9. Tokenization
# ==================================================
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=180  # ✅ 加长序列长度
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# ==================================================
# 10. 加权损失函数（解决类别不平衡）
# ==================================================
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        device = next(model.parameters()).device
        loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(device))

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ==================================================
# 11. 超参数全面优化（直接提升准确率）
# ==================================================
training_args = TrainingArguments(
    output_dir='./results/PR_model',
    num_train_epochs=20,
    learning_rate=2e-5,          # ✅ 最优学习率
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_steps=20,

    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1_macro',
    greater_is_better=True,

    save_total_limit=2,
    report_to='none',

    # ✅ 优化器 & 梯度裁剪（防止过拟合）
    optim="adamw_torch",
    max_grad_norm=1.0,
)

# ==================================================
# 12. Metrics
# ==================================================
from sklearn.metrics import accuracy_score, f1_score, classification_report

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')

    print("\nClassification Report:")
    print(classification_report(labels, predictions, target_names=unique_labels, digits=4))

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

# ==================================================
# 13. Trainer
# ==================================================
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
)

# ==================================================
# 14. Train
# ==================================================
print("\nStarting Training...")
trainer.train()

# ==================================================
# 15. Evaluate
# ==================================================
eval_results = trainer.evaluate()
print(f"\nBest Results: {eval_results}")

# ==================================================
# 16. Save
# ==================================================
model_save_path = './models/PR_model'
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"\nBest model saved to: {model_save_path}")

# ==================================================
# 17. Prediction Examples
# ==================================================
print("\nExample Predictions")

def predict(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=180, padding=True)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        pred_id = torch.argmax(outputs.logits, dim=-1).item()
    return id2label[pred_id]

for i in range(5):
    text = X_val[i]
    true = id2label[y_val[i]]
    pred = predict(text)
    print(f"\nText: {text[:80]}...")
    print(f"TRUE: {true}   |   PRED: {pred}")