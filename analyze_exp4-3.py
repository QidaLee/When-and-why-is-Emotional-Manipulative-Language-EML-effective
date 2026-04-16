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

# 有效 PR
pr_valid = df[df['PR_label'].isin(VALID_PR_LABELS)].copy()
pr_valid = pr_valid.sort_values(['Dialogue_ID', 'Turn']).reset_index(drop = True)

# ===================== 2. 核心：两个PR之间 → 取最后一条 EML 的 DA =====================
transitions = []

for dia_id, dia_df in pr_valid.groupby('Dialogue_ID'):
    dia_df = dia_df.sort_values('Turn').reset_index(drop = True)
    pr_list = dia_df.to_dict('records')

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

        # 两个 PR 之间的所有语句
        between_df = df[
            (df['Dialogue_ID'] == dia_id) &
            (df['Turn'] > prev_pr['Turn']) &
            (df['Turn'] < next_pr['Turn'])
            ].sort_values('Turn')

        if between_df.empty:
            continue

        # 取出中间所有 EML
        eml_between = between_df[between_df['EML_label'] == 1]
        if eml_between.empty:
            continue

        # ✅ 关键：只取 最后一条 EML
        last_eml = eml_between.iloc[-1]
        da_type = last_eml[DA_COLUMN]

        transitions.append({
            'DA_Type': da_type,
            'PR_Transition': f"{prev_pr['PR_label']}→{next_pr['PR_label']}"
        })

trans_df = pd.DataFrame(transitions).dropna(subset = ['DA_Type'])

# ===================== 3. 生成表格 =====================
result_rows = []
trans_types = ['Compliance→Compliance', 'Compliance→Resistance',
               'Resistance→Compliance', 'Resistance→Resistance']

for da in trans_df['DA_Type'].unique():
    da_sub = trans_df[trans_df['DA_Type'] == da]
    total = len(da_sub)
    for t in trans_types:
        cnt = len(da_sub[da_sub['PR_Transition'] == t])
        if cnt == 0:
            continue
        prob = cnt / total * 100 if total > 0 else 0
        result_rows.append({
            'EML_Presence': 'Yes',
            'DA_Type': da,
            'PR_Transition': t,
            'Count': cnt,
            'DA_Total': total,
            'Probability': round(prob, 2)
        })

result_df = pd.DataFrame(result_rows)
result_df = result_df.sort_values(['DA_Type', 'PR_Transition'])

# ===================== 4. 热力图 =====================
plt.figure(figsize = (12, 6))
pivot = pd.crosstab(trans_df['PR_Transition'], trans_df['DA_Type'], normalize = 'index') * 100
sns.heatmap(pivot, annot = True, fmt = '.1f', cmap = 'Purples', annot_kws = {"size": 12})
plt.title('Last EML Turn DA Between PR Transitions', fontsize = 14, pad = 20)
plt.xlabel('DA Type')
plt.ylabel('PR Transition')
plt.tight_layout()
plt.savefig('Last_EML_DA_Between_PR.png', dpi = 300, bbox_inches = 'tight')
plt.show()

# ===================== 输出 =====================
print("=" * 110)
print("              Last EML DA Between PR Transitions (Final)")
print("=" * 110)
print(result_df.to_string(index = False))

result_df.to_excel('Last_EML_DA_Between_PR_Result.xlsx', index = False)
print("\n✅ 结果已保存：Excel + 热力图")