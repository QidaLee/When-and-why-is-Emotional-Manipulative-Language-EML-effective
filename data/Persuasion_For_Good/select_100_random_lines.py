import pandas as pd
import numpy as np
import random

# 设置随机种子以保证结果可复现
random.seed(42)
np.random.seed(42)

# ============= 抽样选项控制 =============
# 设置是否进行抽样，True=进行抽样（选择100个turn），False=处理全部数据
SAMPLE_ENABLED = False  # 如果要处理全部数据，改为False
SAMPLE_SIZE = 100  # 抽样数量，可以根据需要修改
# ======================================

# 1. 读取原始数据
df = pd.read_excel('300_dialog.xlsx')

# 查看前几行数据确认
print("数据前5行：")
print(df.head())
print("\n数据列名：")
print(df.columns.tolist())
print("\n")

# 2. 数据清理：重命名列、处理缺失值
df_clean = df.copy()

# ============= 根据实际列名重命名 =============
# 实际列名：B2, B4, Turn, Unit, er_label_1, ee_label_1, er_label_2, ee_label_2, neg, neu, pos
# B2 = Dialogue_ID, B4 = Role, Unit = Sentence
rename_dict = {
    'B2': 'Dialogue_ID',
    'B4': 'Role',
    'Unit': 'Sentence'
}

df_clean = df_clean.rename(columns = rename_dict)
print("重命名后的列名：")
print(df_clean.columns.tolist())
print("\n")

# 处理标签列缺失值（NaN替换为空字符串，便于后续列表处理）
label_columns = ['er_label_1', 'ee_label_1', 'er_label_2', 'ee_label_2']
for col in label_columns:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].fillna('')

# 处理情感列缺失值（NaN替换为0，避免后续列表处理出错）
emotion_columns = ['neg', 'neu', 'pos']
for col in emotion_columns:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].fillna(0.0)


# ========== 解析字符串列表为实际列表 ==========
def parse_string_list(s):
    """
    解析原始标签列的字符串格式（如 "['greeting', 'task-related-inquiry']"）为实际列表
    处理空值、空字符串、纯空格等情况，不做去重
    """
    if pd.isna(s) or s == '' or s.strip() == '[]':
        return []

    # 清理字符串格式
    s = str(s).strip().strip('[]').replace("'", "").replace('"', '')
    if s == '':
        return []

    # 分割并清理每个标签（保留所有项，包括重复项）
    items = [item.strip() for item in s.split(',') if item.strip()]
    return items


# 3. 定义合并函数（完全不去重）
def merge_sentences_and_labels(group):
    """
    合并同一Dialogue_ID、Role、Turn下的数据，保留原始行号用于排序
    所有标签列和情感列都保存为原始顺序的列表形式（完全不去重）
    """
    # 按原始行号排序（保证合并顺序和原始数据一致）
    group_sorted = group.sort_index()

    # 合并句子（空格连接）
    merged_sentence = ' '.join(group_sorted['Sentence'].astype(str).tolist())

    # ========== 标签列处理：保留所有项，完全不去重 ==========
    # 先解析所有标签为列表，再合并（保留所有项和顺序）
    all_er_label_1 = []
    all_ee_label_1 = []
    all_er_label_2 = []
    all_ee_label_2 = []

    # 遍历排序后的分组数据，按原始顺序收集标签（保留所有项）
    for idx, row in group_sorted.iterrows():
        # 解析并合并er_label_1（保留所有项）
        er1_labels = parse_string_list(row['er_label_1'])
        all_er_label_1.extend(er1_labels)

        # 解析并合并ee_label_1（保留所有项）
        ee1_labels = parse_string_list(row['ee_label_1'])
        all_ee_label_1.extend(ee1_labels)

        # 解析并合并er_label_2（保留所有项）
        er2_labels = parse_string_list(row['er_label_2'])
        all_er_label_2.extend(er2_labels)

        # 解析并合并ee_label_2（保留所有项）
        ee2_labels = parse_string_list(row['ee_label_2'])
        all_ee_label_2.extend(ee2_labels)

    # ========== 情感值处理：保留所有项，完全不去重 ==========
    all_neg = []
    all_neu = []
    all_pos = []

    for idx, row in group_sorted.iterrows():
        # 处理情感值（转为浮点数，保留3位小数，保留所有项）
        neg_val = round(float(row['neg']) if pd.notna(row['neg']) else 0.0, 3)
        neu_val = round(float(row['neu']) if pd.notna(row['neu']) else 0.0, 3)
        pos_val = round(float(row['pos']) if pd.notna(row['pos']) else 0.0, 3)

        all_neg.append(neg_val)
        all_neu.append(neu_val)
        all_pos.append(pos_val)

    # 记录分组内最小原始行号（用于后续排序）
    min_original_index = group.index.min()

    return pd.Series({
        'Sentence': merged_sentence,
        'er_label_1': all_er_label_1 if all_er_label_1 else [],
        'ee_label_1': all_ee_label_1 if all_ee_label_1 else [],
        'er_label_2': all_er_label_2 if all_er_label_2 else [],
        'ee_label_2': all_ee_label_2 if all_ee_label_2 else [],
        'neg': all_neg if all_neg else [],
        'neu': all_neu if all_neu else [],
        'pos': all_pos if all_pos else [],
        'min_original_index': min_original_index
    })


# 4. 按核心键分组合并
print("开始合并数据...")
df_merged = df_clean.groupby(['Dialogue_ID', 'Role', 'Turn'], group_keys = False).apply(
    merge_sentences_and_labels
).reset_index()

print(f"合并后数据行数：{len(df_merged)}")
print("\n")

# 5. 按原始顺序排序并添加index列
df_merged_sorted = df_merged.sort_values('min_original_index').drop('min_original_index', axis = 1)
df_merged_final = df_merged_sorted.reset_index(drop = True)
df_merged_final.index.name = 'index'
df_merged_final = df_merged_final.reset_index()

# 6. 根据选项决定是否抽样
if SAMPLE_ENABLED:
    print(f"\n=== 抽样模式已开启 ===")
    print(f"将从完整数据中随机抽取 {SAMPLE_SIZE} 个turn")

    # 随机选择turn
    print("=== 随机抽样逻辑执行 ===")
    # 定义turn唯一标识
    df_merged_final['turn_unique_id'] = df_merged_final['Dialogue_ID'].astype(str) + '_turn_' + df_merged_final[
        'Turn'].astype(str)

    # 筛选包含完整角色的turn
    turn_role_count = df_merged_final.groupby('turn_unique_id')['Role'].nunique().reset_index()
    complete_turns = turn_role_count[turn_role_count['Role'] == 2]['turn_unique_id'].tolist()

    print(f"总turn数量：{len(turn_role_count)} 个")
    print(f"包含完整角色的turn数量：{len(complete_turns)} 个")

    # 随机选择指定数量的turn
    if len(complete_turns) >= SAMPLE_SIZE:
        selected_turns = random.sample(complete_turns, SAMPLE_SIZE)
    else:
        selected_turns = complete_turns
        print(f"警告：完整角色turn数量({len(complete_turns)})小于抽样数量({SAMPLE_SIZE})，将使用全部完整turn")

    # 筛选抽样数据
    df_final = df_merged_final[df_merged_final['turn_unique_id'].isin(selected_turns)].copy()

    # 重新生成index列
    df_final = df_final.drop(['turn_unique_id', 'index'], axis = 1, errors = 'ignore')
    df_final = df_final.sort_values(['Dialogue_ID', 'Turn', 'Role'])
    df_final = df_final.reset_index(drop = True)
    df_final.index.name = 'new_index'
    df_final = df_final.reset_index()
    df_final.rename(columns = {'new_index': 'index'}, inplace = True)

    print(f"\n=== 抽样完成 ===")
    print(f"抽样后数据行数：{len(df_final)} 行")
    print(f"抽样后包含turn数量：{df_final.groupby(['Dialogue_ID', 'Turn']).ngroups} 个")

else:
    print(f"\n=== 抽样模式已关闭 ===")
    print("将处理全部数据（不进行抽样）")

    # 不抽样，直接使用合并后的全部数据
    df_final = df_merged_final.copy()

    # 确保index列正确
    if 'index' in df_final.columns:
        df_final = df_final.drop('index', axis = 1)
    df_final = df_final.reset_index(drop = True)
    df_final.index.name = 'new_index'
    df_final = df_final.reset_index()
    df_final.rename(columns = {'new_index': 'index'}, inplace = True)

    print(f"全部数据行数：{len(df_final)} 行")
    print(f"全部数据包含turn数量：{df_final.groupby(['Dialogue_ID', 'Turn']).ngroups} 个")

# 7. 验证结果
print(f"\n=== 最终数据验证 ===")
print(f"最终数据行数：{len(df_final)} 行")
print(f"最终数据列数：{len(df_final.columns)} 列")
print(f"最终数据列名：{df_final.columns.tolist()}")

# 验证角色分布
role_distribution = df_final['Role'].value_counts().sort_index()
print(f"\n数据角色分布：")
print(f"  Role=0（说服者）：{role_distribution.get(0, 0)} 行")
print(f"  Role=1（被说服者）：{role_distribution.get(1, 0)} 行")

# 验证标签列（保留所有项，不去重）
print(f"\n标签列验证（前3行，保留所有项）：")
for i in range(min(3, len(df_final))):
    row = df_final.iloc[i]
    print(f"index={row['index']}: ee_label_1={row['ee_label_1']} (长度：{len(row['ee_label_1'])})")
    print(f"          er_label_1={row['er_label_1']} (长度：{len(row['er_label_1'])})")

# 验证情感列（保留所有项，不去重）
print(f"\n情感列验证（前3行，保留所有项）：")
for i in range(min(3, len(df_final))):
    row = df_final.iloc[i]
    print(f"index={row['index']}: neg={row['neg']} (长度：{len(row['neg'])})")
    print(f"          neu={row['neu']} (长度：{len(row['neu'])})")
    print(f"          pos={row['pos']} (长度：{len(row['pos'])})")

# 显示前5行核心信息
print(f"\n数据前5行：")
display_cols = ['index', 'Dialogue_ID', 'Role', 'Turn', 'ee_label_1', 'neg', 'neu', 'pos']
print(df_final[display_cols].head())

# 8. 保存结果
if SAMPLE_ENABLED:
    output_file = f'{SAMPLE_SIZE}_sample_turns_data.xlsx'
else:
    output_file = 'all_turns_data.xlsx'

df_final.to_excel(output_file, index = False, engine = 'openpyxl')

print(f"\n 数据处理完成！")
print(f"文件已保存为：{output_file}")
print(f"当前模式：{'抽样模式' if SAMPLE_ENABLED else '全部数据处理模式'}")
print(f"✅ 已完全移除所有去重逻辑（包括set()函数）")
print(f"✅ 标签列和情感列保留原始顺序和所有重复项")
print(f"✅ 合并后的列表和原始数据完全对应，无任何数据丢失")