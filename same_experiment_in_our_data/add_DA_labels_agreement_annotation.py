import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from tqdm import tqdm

# ====================== 配置参数（你只需要改这里的路径） ======================
MODEL_PATH = "../models/bert-base-uncased_DA_MRDA_our_label"
DATA_PATH = "../data/our_data/agreement_annotations_processed.csv"  # 我们刚生成的文件
OUTPUT_PATH = "agreement_annotations_processed_with_DA.csv"  # 输出带DA标签的最终文件

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# ====================== 加载模型 ======================
def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()
    id2label = model.config.id2label
    label2id = model.config.label2id
    return tokenizer, model, id2label, label2id


# ====================== 预测DA标签 ======================
def predict_da_labels(texts, tokenizer, model, id2label, batch_size=32):
    all_predictions = []
    all_confidences = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="预测DA标签"):
            batch_texts = texts[i:i + batch_size]

            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(device)

            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            confidences, predictions = torch.max(probabilities, dim=1)

            batch_predictions = [id2label[int(pred)] for pred in predictions]
            batch_confidences = [float(conf) for conf in confidences]

            all_predictions.extend(batch_predictions)
            all_confidences.extend(batch_confidences)

    return all_predictions, all_confidences


# ====================== 主流程 ======================
def main():
    # 1. 加载模型
    print("加载模型和分词器...")
    tokenizer, model, id2label, label2id = load_model_and_tokenizer(MODEL_PATH)
    print(f"模型标签: {list(label2id.keys())}")

    # 2. 读取我们处理好的 CSV 文件
    print("读取处理好的数据...")
    df = pd.read_csv(DATA_PATH)

    # 3. 我们的文本列叫 text，不是 Sentence！
    texts = df["text"].fillna("").tolist()

    # 4. 预测
    print("开始预测对话行为(DA)标签...")
    da_predictions, da_confidences = predict_da_labels(
        texts, tokenizer, model, id2label
    )

    # 5. 把DA标签加入数据
    df["DA_label"] = da_predictions
    df["DA_confidence"] = da_confidences

    # 6. 保存最终文件
    print(f"保存带DA标签的最终文件: {OUTPUT_PATH}")
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    # 展示结果
    print("\n✅ DA标签预测完成！")
    print(df[["conversation_id", "turn", "speaker", "DA_label"]].head(10))


if __name__ == "__main__":
    main()