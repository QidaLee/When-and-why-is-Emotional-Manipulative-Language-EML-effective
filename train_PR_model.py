import pandas as pd
import numpy as np
import os
import json
import csv
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
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
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# 固定随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

# ==================================================
# 📂 自动创建保存目录（所有训练结果都存在这里）
# ==================================================
base_dir = "./train_logs_PR"
os.makedirs(base_dir, exist_ok = True)

# 子目录
split_dir = os.path.join(base_dir, "dataset_split")  # 训练集/测试集划分
log_dir = os.path.join(base_dir, "training_logs")  # 训练日志
report_dir = os.path.join(base_dir, "reports")  # 指标报告
os.makedirs(split_dir, exist_ok = True)
os.makedirs(log_dir, exist_ok = True)
os.makedirs(report_dir, exist_ok = True)

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
    'balanced', classes = np.unique(labels), y = labels
)
class_weights = torch.tensor(class_weights, dtype = torch.float32)

# ==================================================
# ✅ 4. 训练集/验证集划分（带索引保存！）
# ==================================================
X_train, X_val, y_train, y_val, train_idx, val_idx = train_test_split(
    texts, labels, df.index.tolist(),
    test_size = 0.2, random_state = 42, stratify = labels
)

# ✅ 保存：划分索引 + 训练/验证样本
pd.DataFrame({'split': 'train', 'index': train_idx}).to_csv(os.path.join(split_dir, "train_indices.csv"), index = False)
pd.DataFrame({'split': 'val', 'index': val_idx}).to_csv(os.path.join(split_dir, "val_indices.csv"), index = False)

train_split = pd.DataFrame({'text': X_train, 'label': [id2label[y] for y in y_train]})
val_split = pd.DataFrame({'text': X_val, 'label': [id2label[y] for y in y_val]})
train_split.to_csv(os.path.join(split_dir, "train_data.csv"), index = False, encoding = 'utf-8-sig')
val_split.to_csv(os.path.join(split_dir, "val_data.csv"), index = False, encoding = 'utf-8-sig')

print(f"训练集大小: {len(X_train)} | 验证集大小: {len(X_val)}")
print(f"划分结果已保存到: {split_dir}")

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

model_save_path = "./models/PR_model"
if os.path.exists(model_save_path):
    print(f"✅ 从 {model_save_path} 继续断点训练...")
    model = AutoModelForSequenceClassification.from_pretrained(model_save_path)
else:
    print("✅ 首次训练 bert-base-uncased...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels = len(unique_labels),
        id2label = id2label, label2id = label2id
    )


# ==================================================
# 7. Tokenization
# ==================================================
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding = 'max_length',
        truncation = True,
        max_length = 180
    )


train_dataset = train_dataset.map(tokenize_function, batched = True)
val_dataset = val_dataset.map(tokenize_function, batched = True)
train_dataset.set_format('torch', columns = ['input_ids', 'attention_mask', 'label'])
val_dataset.set_format('torch', columns = ['input_ids', 'attention_mask', 'label'])


# ==================================================
# 8. Weighted Trainer
# ==================================================
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        device = next(model.parameters()).device
        loss_fct = nn.CrossEntropyLoss(weight = class_weights.to(device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ==================================================
# 9. 训练参数
# ==================================================
training_args = TrainingArguments(
    output_dir = './results/PR_model',
    num_train_epochs = 12,
    learning_rate = 2e-5,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 16,
    warmup_ratio = 0.1,
    weight_decay = 0.01,
    logging_steps = 20,

    eval_strategy = "epoch",
    save_strategy = "epoch",
    save_total_limit = 3,
    load_best_model_at_end = True,
    metric_for_best_model = "f1_macro",
    greater_is_better = True,

    logging_dir = log_dir,  # ✅ 日志保存
    logging_first_step = True,
)

# ==================================================
# ✅ 10. 计算指标 + 自动保存所有验证结果
# ==================================================
all_eval_results = []


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis = 1)

    acc = accuracy_score(labels, predictions)
    f1_m = f1_score(labels, predictions, average = 'macro')
    f1_w = f1_score(labels, predictions, average = 'weighted')
    report = classification_report(labels, predictions, target_names = unique_labels, digits = 4, output_dict = True)

    # 保存每轮结果
    all_eval_results.append({
        "step": len(all_eval_results) + 1,
        "accuracy": acc,
        "f1_macro": f1_m,
        "f1_weighted": f1_w
    })

    print("\n" + classification_report(labels, predictions, target_names = unique_labels, digits = 4))
    return {'accuracy': acc, 'f1_macro': f1_m, 'f1_weighted': f1_w}


# ==================================================
# 11. Trainer
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
# 12. 开始训练
# ==================================================
print("\n🚀 开始训练...")
trainer.train()

# ==================================================
# 13. 保存模型
# ==================================================
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"\n✅ 最优模型已保存到：{model_save_path}")

# ==================================================
# ✅ 14. 最终验证：保存完整报告 + 混淆矩阵
# ==================================================
print("\n📊 生成最终验证报告...")
model.eval()
val_preds = []
val_trues = []

for i in range(len(val_dataset)):
    batch = {k: v.unsqueeze(0).to(model.device) for k, v in val_dataset[i].items()}
    with torch.no_grad():
        logits = model(**batch).logits
    pred = torch.argmax(logits, dim = -1).item()
    val_preds.append(pred)
    val_trues.append(val_dataset[i]['label'].item())

# 保存最终详细报告
final_report = classification_report(val_trues, val_preds, target_names = unique_labels, digits = 4)
with open(os.path.join(report_dir, "final_classification_report.txt"), 'w', encoding = 'utf-8') as f:
    f.write(final_report)

# 保存每轮训练指标
pd.DataFrame(all_eval_results).to_csv(os.path.join(report_dir, "epoch_metrics.csv"), index = False)

# 保存混淆矩阵
cm = confusion_matrix(val_trues, val_preds)
cm_df = pd.DataFrame(cm, index = unique_labels, columns = unique_labels)
cm_df.to_csv(os.path.join(report_dir, "confusion_matrix.csv"))

# 画图
plt.figure(figsize = (10, 8))
sns.heatmap(cm_df, annot = True, fmt = 'd', cmap = 'Blues')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(report_dir, "confusion_matrix.png"), dpi = 300)

print(f"✅ 所有训练报告已保存到: {report_dir}")
print(f"✅ 训练日志已保存到: {log_dir}")

# ==================================================
# 15. 预测示例
# ==================================================
print("\n🔍 预测示例：")


def predict(text):
    inputs = tokenizer(text, return_tensors = 'pt', truncation = True, max_length = 180, padding = True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        pred_id = torch.argmax(model(**inputs).logits, dim = -1).item()
    return id2label[pred_id]


for i in range(5):
    text = X_val[i]
    true = id2label[y_val[i]]
    pred = predict(text)
    print(f"\n{text[:70]}...")
    print(f"TRUE: {true} | PRED: {pred}")