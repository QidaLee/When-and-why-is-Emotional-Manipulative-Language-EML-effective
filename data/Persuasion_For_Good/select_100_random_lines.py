import pandas as pd
import numpy as np
import random

# 设置随机种子以保证结果可复现
random.seed(42)
np.random.seed(42)

# 1. 读取原始数据
df = pd.read_excel('300_dialog.xlsx')

# 2. 数据清理：删除无用列、重命名、处理缺失值
df_clean = df.copy()

# 删除不需要的列（说明列和空列）
df_clean = df_clean.drop(['Unnamed: 0', 'Unnamed: 12', 'B2: Dialogue ID'], axis = 1)

# 重命名列以便理解（Dialogue_ID=对话ID, Role=角色, Turn=轮次）
df_clean.columns = ['Dialogue_ID', 'Role', 'Turn', 'Sentence', 'er_label_1', 'ee_label_1',
                    'er_label_2', 'ee_label_2', 'neg', 'neu', 'pos']

# 处理标签列缺失值（NaN替换为空字符串，便于后续去重）
label_columns = ['er_label_1', 'ee_label_1', 'er_label_2', 'ee_label_2']
for col in label_columns:
    df_clean[col] = df_clean[col].fillna('')

# 处理情感列缺失值（NaN替换为0，避免后续列表处理出错）
emotion_columns = ['neg', 'neu', 'pos']
for col in emotion_columns:
    df_clean[col] = df_clean[col].fillna(0.0)


# 3. 定义合并函数（修改：情感值也保存为去重后的列表）
def merge_sentences_and_labels(group):
    """
    合并同一Dialogue_ID、Role、Turn下的数据，保留原始行号用于排序
    所有标签列和情感列都保存为去重后的列表形式
    """
    # 合并句子（空格连接）
    merged_sentence = ' '.join(group['Sentence'].astype(str).tolist())

    # 收集所有非空标签，去重后排序
    all_er_label_1 = list(set([label for label in group['er_label_1'] if label != '']))
    all_ee_label_1 = list(set([label for label in group['ee_label_1'] if label != '']))
    all_er_label_2 = list(set([label for label in group['er_label_2'] if label != '']))
    all_ee_label_2 = list(set([label for label in group['ee_label_2'] if label != '']))

    # 核心修改：情感值也保存为去重后的列表（保留原始值，不取平均）
    all_neg = list(set([round(val, 3) for val in group['neg']]))  # 保留3位小数去重
    all_neu = list(set([round(val, 3) for val in group['neu']]))
    all_pos = list(set([round(val, 3) for val in group['pos']]))

    # 记录分组内最小原始行号（用于后续排序）
    min_original_index = group.index.min()

    return pd.Series({
        'Sentence': merged_sentence,
        'er_label_1': all_er_label_1 if all_er_label_1 else [],
        'ee_label_1': all_ee_label_1 if all_ee_label_1 else [],
        'er_label_2': all_er_label_2 if all_er_label_2 else [],
        'ee_label_2': all_ee_label_2 if all_ee_label_2 else [],
        'neg': all_neg if all_neg else [],  # 情感值存列表
        'neu': all_neu if all_neu else [],
        'pos': all_pos if all_pos else [],
        'min_original_index': min_original_index
    })


# 4. 按核心键分组合并
df_merged = df_clean.groupby(['Dialogue_ID', 'Role', 'Turn'], group_keys = False).apply(
    merge_sentences_and_labels
).reset_index()

# 5. 按原始顺序排序并添加index列（第一版）
df_merged_sorted = df_merged.sort_values('min_original_index').drop('min_original_index', axis = 1)
# 关键修复：先删除旧的index列（如果存在），再重新添加
if 'index' in df_merged_sorted.columns:
    df_merged_sorted = df_merged_sorted.drop('index', axis = 1)
df_merged_final = df_merged_sorted.reset_index(drop = True)
df_merged_final.index.name = 'index'
df_merged_final = df_merged_final.reset_index()

# 6. 随机选择100个turn（共200行）
print("=== 随机抽样逻辑执行 ===")
# 定义turn唯一标识
df_merged_final['turn_unique_id'] = df_merged_final['Dialogue_ID'] + '_turn_' + df_merged_final['Turn'].astype(str)

# 筛选包含完整角色的turn
turn_role_count = df_merged_final.groupby('turn_unique_id')['Role'].nunique().reset_index()
complete_turns = turn_role_count[turn_role_count['Role'] == 2]['turn_unique_id'].tolist()

print(f"总turn数量：{len(turn_role_count)} 个")
print(f"包含完整角色的turn数量：{len(complete_turns)} 个")

# 随机选择100个turn
selected_turns = random.sample(complete_turns, 100) if len(complete_turns) >= 100 else complete_turns

# 筛选抽样数据
df_sampled = df_merged_final[df_merged_final['turn_unique_id'].isin(selected_turns)].copy()

# 关键修复：重新生成index列（核心解决报错的部分）
# 步骤1：删除辅助列和旧的index列
df_sampled = df_sampled.drop(['turn_unique_id', 'index'], axis = 1, errors = 'ignore')

# 步骤2：按对话逻辑排序
df_sampled = df_sampled.sort_values(['Dialogue_ID', 'Turn', 'Role'])

# 步骤3：重新生成连续的index列（避免重复）
df_sampled = df_sampled.reset_index(drop = True)  # 重置索引，不保留旧索引
df_sampled.index.name = 'new_index'  # 先命名为临时名称
df_sampled = df_sampled.reset_index()  # 转为列

# 步骤4：将临时列名改为index（最终修复）
df_sampled.rename(columns = {'new_index': 'index'}, inplace = True)

# 7. 验证结果（新增：验证情感列是否为列表格式）
print(f"\n=== 抽样结果验证 ===")
print(f"抽样后数据行数：{len(df_sampled)} 行")
print(f"抽样后包含turn数量：{df_sampled.groupby(['Dialogue_ID', 'Turn']).ngroups} 个")

# 验证角色分布
role_distribution = df_sampled['Role'].value_counts().sort_index()
print(f"\n抽样数据角色分布：")
print(f"  Role=0（说服者）：{role_distribution.get(0, 0)} 行")
print(f"  Role=1（被说服者）：{role_distribution.get(1, 0)} 行")

# 验证情感列格式（展示前3行的情感列表）
print(f"\n情感列格式验证（前3行）：")
for i in range(3):
    row = df_sampled.iloc[i]
    print(f"index={row['index']}: neg={row['neg']}, neu={row['neu']}, pos={row['pos']}")

# 显示前5行核心信息
print(f"\n抽样数据前5行：")
display_cols = ['index', 'Dialogue_ID', 'Role', 'Turn', 'neg', 'neu', 'pos']
print(df_sampled[display_cols].head())

# 8. 保存结果
output_file = '100_sample_turns_data.xlsx'
df_sampled.to_excel(output_file, index = False, engine = 'openpyxl')

print(f"\n 数据处理完成！文件已保存为：{output_file}")
print(f"最终数据行数：{len(df_sampled)} 行，列数：{len(df_sampled.columns)} 列")
print(f"注意：neg/neu/pos列已改为去重后的列表格式，不再是平均值！")