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
)
from datasets import Dataset
import torch
from torch import nn
import warnings
import random

warnings.filterwarnings('ignore')

# 固定随机种子
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
texts = df['Sentence'].astype(str).tolist()
labels_raw = df['persuasion_result'].tolist()

# ==================================================
# 2. Label Mapping
# ==================================================
unique_labels = sorted(list(set(labels_raw)))
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}
labels = [label2id[label] for label in labels_raw]

# ==================================================
# 3. Class Weights
# ==================================================
class_weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(labels), y=labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# ==================================================
# 4. Train/Val Split
# ==================================================
X_train, X_val, y_train, y_val = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# ==================================================
# 5. Dataset
# ==================================================
train_df = pd.DataFrame({'text': X_train, 'label': y_train})
val_df = pd.DataFrame({'text': X_val, 'label': y_val})
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# ==================================================
# 6. Model & Tokenizer
# ==================================================
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ==================================================
# 🔥 关键：支持断点续训（从上次保存的模型继续训练）
# ==================================================
import os
model_save_path = "./models/PR_model"
if os.path.exists(model_save_path):
    print(f"✅ 找到已保存模型，从 {model_save_path} 继续断点训练...")
    model = AutoModelForSequenceClassification.from_pretrained(model_save_path)
else:
    print("✅ 首次训练，加载预训练 bert-base-uncased...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(unique_labels),
        id2label=id2label, label2id=label2id
    )

# ==================================================
# 7. Tokenization
# ==================================================
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=180
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# ==================================================
# 8. Weighted Trainer
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
# 🔥 最重要：训练参数（每轮保存 + 断点续训）
# ==================================================
training_args = TrainingArguments(
    output_dir='./results/PR_model',
    num_train_epochs=12,             # 减少一点，更快结束
    learning_rate=2e-5,
    per_device_train_batch_size=8,   # 减小batch，速度大幅提升
    per_device_eval_batch_size=16,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_steps=20,

    # ======================
    # ✅ 核心：自动保存 + 续训
    # ======================
    evaluation_strategy="epoch",     # 每轮评估
    save_strategy="epoch",           # ✅每轮都保存模型
    save_total_limit=3,              # 只保留最新3个模型
    load_best_model_at_end=True,     # ✅自动加载最好模型
    metric_for_best_model="f1_macro",
    greater_is_better=True,

    optim="adamw_torch",
    max_grad_norm=1.0,
    report_to='none',
)

# ==================================================
# 9. Metrics
# ==================================================
from sklearn.metrics import accuracy_score, f1_score, classification_report
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    f1_m = f1_score(labels, predictions, average='macro')
    f1_w = f1_score(labels, predictions, average='weighted')
    print("\n" + classification_report(labels, predictions, target_names=unique_labels, digits=4))
    return {'accuracy': acc, 'f1_macro': f1_m, 'f1_weighted': f1_w}

# ==================================================
# 10. Trainer
# ==================================================
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# ==================================================
# 11. Start Training
# ==================================================
print("\n🚀 开始训练（支持断点续训）...")
trainer.train()

# ==================================================
# 12. 保存最终 BEST 模型
# ==================================================
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"\n✅ 最优模型已保存到：{model_save_path}")

# ==================================================
# 13. 预测示例
# ==================================================
print("\n🔍 预测示例：")
def predict(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=180, padding=True)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        pred_id = torch.argmax(model(**inputs).logits, dim=-1).item()
    return id2label[pred_id]

for i in range(5):
    text = X_val[i]
    true = id2label[y_val[i]]
    pred = predict(text)
    print(f"\n{text[:70]}...")
    print(f"TRUE: {true} | PRED: {pred}")