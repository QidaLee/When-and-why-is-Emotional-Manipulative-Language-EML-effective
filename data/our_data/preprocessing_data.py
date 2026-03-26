import pandas as pd
import numpy as np

# 1. 读取数据
df = pd.read_csv('agreement_annotations.csv')

# 2. 保留需要的列
keep_cols = ['speaker', 'conversation_id', 'text',
             'manipulation_davide', 'manipulation_diletta',
             'manipulation_inga', 'manipulation_matias', 'manipulation']
df = df[keep_cols].copy()

# 3. 生成连续发言组（同说话人+同对话合并）
df['group'] = ((df['speaker'] != df['speaker'].shift()) |
               (df['conversation_id'] != df['conversation_id'].shift())).cumsum()

# 4. 计算 manipulation = 0/1（有1则1，全0则0）
df['manipulation_final'] = (df[['manipulation_davide', 'manipulation_diletta',
                                'manipulation_inga', 'manipulation_matias', 'manipulation']]
                            .sum(axis=1) > 0).astype(int)

# 5. 分组聚合（保留顺序）
result = df.groupby(['conversation_id', 'group', 'speaker'],
                    as_index=False, sort=False).agg({
    'text': ' '.join,
    'manipulation_final': 'max'
})

# --------------------------
# 核心：正确计算 turn（你的规则：turn = max(各speaker发言次数) - 1）
# --------------------------
def calculate_turn_correctly(conversation_df):
    speaker_counts = {}
    turn_list = []

    for _, row in conversation_df.iterrows():
        speaker = row['speaker']
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        current_turn = max(speaker_counts.values()) - 1
        turn_list.append(current_turn)

    conversation_df['turn'] = turn_list
    return conversation_df

result = result.groupby('conversation_id', group_keys=False, sort=False).apply(calculate_turn_correctly)

# 6. 最终整理
result.rename(columns={'manipulation_final': 'manipulation'}, inplace=True)
final_cols = ['conversation_id', 'turn', 'speaker', 'text', 'manipulation']
result = result[final_cols].reset_index(drop=True)

# 保存（已修复编码错误）
result.to_csv('agreement_annotations_processed.csv', index=False, encoding='utf-8-sig')

print("✅ 处理完成！文件已保存")
print("\n预览：")
print(result.head(10))