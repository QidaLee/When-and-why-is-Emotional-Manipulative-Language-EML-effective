import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from tqdm import tqdm  # 进度条，方便查看预测进度

# ====================== 核心配置参数 ======================
MODEL_PATH = "models/PR_model"  # PR模型路径
DATA_PATH = "data/Persuasion_For_Good/all_turns_data.xlsx"  # 原始数据路径
OUTPUT_PATH = "data/Persuasion_For_Good/all_turns_data_with_PR.xlsx"  # 带PR标签的输出路径
BATCH_SIZE = 32  # 批量预测大小（可根据显存调整）
MAX_SEQ_LENGTH = 128  # 文本最大长度（适配模型输入）

# 设置计算设备（优先使用GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用计算设备: {device}")


# ====================== 加载模型和分词器 ======================
def load_pr_model(model_path):
    """
    加载训练好的PR模型和对应的分词器
    返回: 分词器、模型、标签映射（id->标签）、标签映射（标签->id）
    """
    print("开始加载PR模型和分词器...")
    # 加载分词器（自动适配模型的分词规则）
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 加载分类模型（设置为评估模式，禁用训练相关参数）
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model = model.to(device)  # 移到指定设备
    model.eval()  # 评估模式，避免dropout等训练层生效

    # 获取模型的标签映射（从模型配置中读取，无需手动定义）
    id2label = model.config.id2label
    label2id = model.config.label2id

    print(f"PR模型加载完成，包含标签: {list(label2id.keys())}")
    return tokenizer, model, id2label, label2id


# ====================== 批量预测PR标签 ======================
def predict_pr_labels(texts, tokenizer, model, id2label):
    """
    对文本列表批量预测PR标签，并返回标签和对应置信度
    修复：移除token_type_ids参数，适配DistilBERT模型
    Args:
        texts: 需要预测的文本列表
        tokenizer: 分词器
        model: PR模型
        id2label: id到PR标签的映射字典
    Returns:
        pr_labels: 每个文本的PR标签列表
        pr_confidences: 每个标签对应的置信度（0-1）
    """
    pr_labels = []
    pr_confidences = []

    # 禁用梯度计算，节省显存并加速预测
    with torch.no_grad():
        # 分批次处理文本，避免内存溢出
        for start_idx in tqdm(range(0, len(texts), BATCH_SIZE), desc = "PR标签预测中"):
            # 截取当前批次的文本
            batch_texts = texts[start_idx:start_idx + BATCH_SIZE]

            # 文本分词处理（自动padding、truncation，适配模型输入）
            # 关键修复：return_token_type_ids=False 禁止生成token_type_ids
            inputs = tokenizer(
                batch_texts,
                padding = True,  # 批量内填充到相同长度
                truncation = True,  # 超过max_length的文本截断
                max_length = MAX_SEQ_LENGTH,
                return_tensors = "pt",  # 返回PyTorch张量
                return_token_type_ids = False  # 核心修复：不生成token_type_ids
            ).to(device)  # 移到指定设备

            # 模型预测
            outputs = model(**inputs)
            logits = outputs.logits  # 获取模型原始输出

            # 计算每个标签的概率（softmax归一化）
            probabilities = torch.softmax(logits, dim = 1)
            # 获取最大概率的置信度和对应的标签id
            batch_confidences, batch_label_ids = torch.max(probabilities, dim = 1)

            # 将张量转换为Python列表，并映射为实际标签
            batch_confidences = [float(conf) for conf in batch_confidences]
            batch_labels = [id2label[int(label_id)] for label_id in batch_label_ids]

            # 添加到总结果中
            pr_labels.extend(batch_labels)
            pr_confidences.extend(batch_confidences)

    return pr_labels, pr_confidences


# ====================== 主执行流程 ======================
def main():
    # 1. 加载PR模型和分词器
    tokenizer, model, id2label, label2id = load_pr_model(MODEL_PATH)

    # 2. 读取Excel数据
    print("读取原始Excel数据...")
    df = pd.read_excel(DATA_PATH)

    # 检查关键列是否存在
    if "Sentence" not in df.columns:
        raise ValueError("数据文件中缺少 'Sentence' 列（待预测的文本列）！")

    # 处理空值（避免预测出错）
    texts = df["Sentence"].fillna("").tolist()
    print(f"共需预测 {len(texts)} 条文本")

    # 3. 批量预测PR标签
    pr_labels, pr_confidences = predict_pr_labels(texts, tokenizer, model, id2label)

    # 4. 将预测结果添加到DataFrame
    df["PR_label"] = pr_labels  # PR标签列
    df["PR_confidence"] = pr_confidences  # 标签置信度列

    # 5. 保存结果到新Excel文件
    print(f"保存预测结果到: {OUTPUT_PATH}")
    df.to_excel(OUTPUT_PATH, index = False)

    # 打印PR标签分布统计（方便快速查看结果）
    print("\nPR标签预测结果统计:")
    print(df["PR_label"].value_counts())
    print("\nPR标签添加完成！")


if __name__ == "__main__":
    main()