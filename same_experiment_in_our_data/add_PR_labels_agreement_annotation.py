import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from tqdm import tqdm
from collections import Counter

# ====================== 配置参数（和DA代码保持一致） ======================
MODEL_PATH = "../models/PR_model"  # 你的PR模型路径
DATA_PATH = "agreement_annotations_processed_with_DA.csv"  # 我们处理好的数据
OUTPUT_PATH = "agreement_annotations_processed_with_DA_PR.csv"  # 最终输出文件

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


# ====================== 预测PR标签 ======================
def predict_pr_labels(texts, tokenizer, model, id2label, batch_size=32):
    all_predictions = []
    all_confidences = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc = "预测PR标签"):
            batch_texts = texts[i:i + batch_size]

            inputs = tokenizer(
                batch_texts,
                padding = True,
                truncation = True,
                max_length = 128,
                return_tensors = "pt",
                return_token_type_ids = False  # 适配你的模型
            ).to(device)

            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim = 1)
            confidences, predictions = torch.max(probabilities, dim = 1)

            batch_predictions = [id2label[int(pred)] for pred in predictions]
            batch_confidences = [float(conf) for conf in confidences]

            all_predictions.extend(batch_predictions)
            all_confidences.extend(batch_confidences)

    return all_predictions, all_confidences


# ====================== 展示置信度统计（新增函数） ======================
def show_confidence_analysis(df):
    print("\n" + "=" * 80)
    print("                    PR 预测置信度统计分析")
    print("=" * 80)

    conf = df["PR_confidence"]

    # 1. 总体统计
    print(f"\n📊 总体置信度概况：")
    print(f"总数：{len(conf):,}")
    print(f"平均值：{conf.mean():.4f}")
    print(f"中位数：{conf.median():.4f}")
    print(f"最小值：{conf.min():.4f}")
    print(f"最大值：{conf.max():.4f}")
    print(f"标准差：{conf.std():.4f}")

    # 2. 分段统计
    print(f"\n📦 置信度分段统计：")
    low = (conf < 0.5).sum()
    mid1 = ((conf >= 0.5) & (conf < 0.8)).sum()
    mid2 = ((conf >= 0.8) & (conf < 0.9)).sum()
    high = (conf >= 0.9).sum()

    print(f"＜0.5（低置信）：{low} 例 ({low / len(conf) * 100:.2f}%)")
    print(f"0.5–0.8（中置信）：{mid1} 例 ({mid1 / len(conf) * 100:.2f}%)")
    print(f"0.8–0.9（高置信）：{mid2} 例 ({mid2 / len(conf) * 100:.2f}%)")
    print(f"≥0.9（极高置信）：{high} 例 ({high / len(conf) * 100:.2f}%)")

    # 3. 按标签分组统计置信度
    print(f"\n🏷️  各PR标签置信度：")
    for label in sorted(df["PR_label"].unique()):
        sub = df[df["PR_label"] == label]
        print(f"→ {label:12s}：{len(sub):4,} 例 | 平均置信度 = {sub['PR_confidence'].mean():.4f}")

    print("\n" + "=" * 80)


# ====================== 主流程 ======================
def main():
    # 1. 加载模型
    print("加载PR模型和分词器...")
    tokenizer, model, id2label, label2id = load_model_and_tokenizer(MODEL_PATH)
    print(f"模型标签: {list(label2id.keys())}")

    # 2. 读取我们处理好的 CSV 文件
    print("读取处理好的数据...")
    df = pd.read_csv(DATA_PATH)

    # 3. 文本列是 text
    texts = df["text"].fillna("").tolist()

    # 4. 预测
    print("开始预测PR标签...")
    pr_predictions, pr_confidences = predict_pr_labels(
        texts, tokenizer, model, id2label
    )

    # 5. 把PR标签加入数据
    df["PR_label"] = pr_predictions
    df["PR_confidence"] = pr_confidences

    # 6. 显示置信度分析（新增）
    show_confidence_analysis(df)

    # 7. 保存最终文件
    print(f"\n保存带PR标签的最终文件: {OUTPUT_PATH}")
    df.to_csv(OUTPUT_PATH, index = False, encoding = "utf-8-sig")

    # 展示结果
    print("\n✅ PR标签预测完成！前10行预览：")
    print(df[["conversation_id", "turn", "speaker", "PR_label", "PR_confidence"]].head(10))


if __name__ == "__main__":
    main()
