import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from collections import Counter

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# ===================== 配置 =====================
VALID_PR_LABELS = ['Compliance', 'Resistance']
DA_COLUMN = 'DA_label'
TURN_GAP_THRESHOLD = 3

# ===================== 1. 加载数据 =====================
df = pd.read_excel('output_data/all_turns_data_with_EML_DA_PR.xlsx')
df['Turn'] = pd.to_numeric(df['Turn'], errors = 'coerce')
df = df.dropna(subset = ['Dialogue_ID', 'Turn', 'Role', DA_COLUMN])
df = df.sort_values(['Dialogue_ID', 'Turn']).reset_index(drop = True)

# 标记是否有 EML
eml_dialogues = df[df['EML_label'] == 1]['Dialogue_ID'].unique()
df['EML_Presence'] = df['Dialogue_ID'].apply(lambda x: 'Yes' if x in eml_dialogues else 'No')

# 有效 PR
pr_valid = df[df['PR_label'].isin(VALID_PR_LABELS)].copy()
pr_valid = pr_valid.sort_values(['Dialogue_ID', 'Turn']).reset_index(drop = True)

# ===================== 2. 核心 PR 转移逻辑（完全不变） =====================
transitions = []

for dia_id, dia_df in pr_valid.groupby('Dialogue_ID'):
    dia_df = dia_df.sort_values('Turn').reset_index(drop = True)
    pr_list = dia_df.to_dict('records')
    eml_group = dia_df['EML_Presence'].iloc[0]

    for i in range(len(pr_list) - 1):
        prev_pr = pr_list[i]
        next_pr = pr_list[i + 1]

        # 同角色
        if prev_pr['Role'] != next_pr['Role']:
            continue
        # Turn 差 ≤3
        turn_gap = next_pr['Turn'] - prev_pr['Turn']
        if turn_gap > TURN_GAP_THRESHOLD:
            continue

        # 取紧邻前一句 DA
        between_df = df[
            (df['Dialogue_ID'] == dia_id) &
            (df['Turn'] > prev_pr['Turn']) &
            (df['Turn'] < next_pr['Turn'])
            ].sort_values('Turn')

        if between_df.empty:
            continue

        last_row = between_df.iloc[-1]
        da_before = last_row[DA_COLUMN]

        transitions.append({
            'EML_Presence': eml_group,
            'DA_Type': da_before,
            'PR_Transition': f"{prev_pr['PR_label']}→{next_pr['PR_label']}"
        })

trans_df = pd.DataFrame(transitions).dropna(subset = ['DA_Type'])

# ===================== 3. 生成你要的完整表格 =====================
result_rows = []
trans_types = ['Compliance→Compliance', 'Compliance→Resistance',
               'Resistance→Compliance', 'Resistance→Resistance']

for eml in ['Yes', 'No']:
    group = trans_df[trans_df['EML_Presence'] == eml]
    for da in group['DA_Type'].unique():
        da_sub = group[group['DA_Type'] == da]
        total = len(da_sub)
        for t in trans_types:
            cnt = len(da_sub[da_sub['PR_Transition'] == t])
            if cnt == 0:
                continue
            prob = cnt / total * 100 if total > 0 else 0
            result_rows.append({
                'EML_Presence': eml,
                'DA_Type': da,
                'PR_Transition': t,
                'Count': cnt,
                'DA_Total': total,
                'Probability': round(prob, 2)
            })

result_df = pd.DataFrame(result_rows)
result_df = result_df.sort_values(['EML_Presence', 'DA_Type', 'PR_Transition'])

# ===================== 4. 两张热力图：With EML / Without EML =====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 6))

# --- 左图：With EML ---
df_yes = trans_df[trans_df['EML_Presence'] == 'Yes']
pivot_yes = pd.crosstab(df_yes['PR_Transition'], df_yes['DA_Type'], normalize = 'index') * 100
sns.heatmap(pivot_yes, annot = True, fmt = '.1f', cmap = 'Oranges', ax = ax1, annot_kws = {"size": 11})
ax1.set_title('DA Before PR Transition (With EML)', fontsize = 14, pad = 15)
ax1.set_xlabel('DA Type')
ax1.set_ylabel('PR Transition')

# --- 右图：Without EML ---
df_no = trans_df[trans_df['EML_Presence'] == 'No']
pivot_no = pd.crosstab(df_no['PR_Transition'], df_no['DA_Type'], normalize = 'index') * 100
sns.heatmap(pivot_no, annot = True, fmt = '.1f', cmap = 'Blues', ax = ax2, annot_kws = {"size": 11})
ax2.set_title('DA Before PR Transition (Without EML)', fontsize = 14, pad = 15)
ax2.set_xlabel('DA Type')
ax2.set_ylabel('PR Transition')

plt.tight_layout()
plt.savefig('DA_Before_PR_Transition_2Heatmaps.png', dpi = 300, bbox_inches = 'tight')
plt.show()

# ===================== 输出 =====================
print("=" * 120)
print("    EML_Presence | DA_Type | PR_Transition | Count | DA_Total | Probability    ")
print("=" * 120)
print(result_df.to_string(index = False))

result_df.to_excel('DA_Before_PR_Transition_Result.xlsx', index = False)
print("\n✅ 结果已保存：Excel + 两张热力图")