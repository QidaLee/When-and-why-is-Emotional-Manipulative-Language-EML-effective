import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from tqdm import tqdm  # 进度条，提升体验

# ====================== 配置参数 ======================
MODEL_PATH = "models/bert-base-uncased_DA_MRDA_our_label"
DATA_PATH = "data/Persuasion_For_Good/all_turns_data.xlsx"
OUTPUT_PATH = "data/Persuasion_For_Good/all_turns_data_with_DA.xlsx"

# 设置设备（GPU优先）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# ====================== 加载模型和分词器 ======================
def load_model_and_tokenizer(model_path):
    """加载训练好的模型和分词器"""
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 加载模型
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model = model.to(device)
    model.eval()  # 设置为评估模式

    # 获取标签映射（从模型配置中）
    id2label = model.config.id2label
    label2id = model.config.label2id

    return tokenizer, model, id2label, label2id


# ====================== 预测DA标签 ======================
def predict_da_labels(texts, tokenizer, model, id2label, batch_size=32):
    """
    批量预测文本的DA标签
    Args:
        texts: 文本列表
        tokenizer: 分词器
        model: 模型
        id2label: id到标签的映射
        batch_size: 批量大小
    Returns:
        预测的标签列表和对应的置信度
    """
    all_predictions = []
    all_confidences = []

    # 分批次处理
    with torch.no_grad():  # 禁用梯度计算，节省内存
        for i in tqdm(range(0, len(texts), batch_size), desc = "预测DA标签"):
            batch_texts = texts[i:i + batch_size]

            # 分词处理
            inputs = tokenizer(
                batch_texts,
                padding = True,
                truncation = True,
                max_length = 128,  # 根据你的模型调整
                return_tensors = "pt"
            ).to(device)

            # 模型预测
            outputs = model(**inputs)
            logits = outputs.logits

            # 计算概率和预测标签
            probabilities = torch.softmax(logits, dim = 1)
            confidences, predictions = torch.max(probabilities, dim = 1)

            # 转换为标签和置信度
            batch_predictions = [id2label[int(pred)] for pred in predictions]
            batch_confidences = [float(conf) for conf in confidences]

            all_predictions.extend(batch_predictions)
            all_confidences.extend(batch_confidences)

    return all_predictions, all_confidences


# ====================== 主流程 ======================
def main():
    # 1. 加载模型和分词器
    print("加载模型和分词器...")
    tokenizer, model, id2label, label2id = load_model_and_tokenizer(MODEL_PATH)
    print(f"模型标签列表: {list(label2id.keys())}")

    # 2. 读取Excel数据
    print("读取Excel数据...")
    df = pd.read_excel(DATA_PATH)

    # 检查必要的列是否存在
    if "Sentence" not in df.columns:
        raise ValueError("数据中缺少 'Sentence' 列！")

    # 获取需要预测的文本列表
    texts = df["Sentence"].fillna("").tolist()  # 处理空值

    # 3. 预测DA标签
    print("开始预测DA标签...")
    da_predictions, da_confidences = predict_da_labels(
        texts, tokenizer, model, id2label
    )

    # 4. 将预测结果添加到DataFrame
    df["DA_label"] = da_predictions
    df["DA_confidence"] = da_confidences

    # 5. 保存结果到新的Excel文件
    print(f"保存结果到: {OUTPUT_PATH}")
    df.to_excel(OUTPUT_PATH, index = False)

    # 打印统计信息
    print("\nDA标签预测统计:")
    print(df["DA_label"].value_counts())
    print("\n任务完成！")


if __name__ == "__main__":
    main()