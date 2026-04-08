import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# ----------------------
# 🔧 最新精细配置
# ----------------------
VALID_PR_LABELS = ['Compliance', 'Resistance']
OUTPUT_EXCEL = 'exp3_final_pr_transitions.xlsx'
TURN_GAP_THRESHOLD = 3  # 🔴 你要求：最大允许 turn 差 = 3

# ----------------------
# 1. 加载数据
# ----------------------
df = pd.read_excel('output_data/all_turns_data_with_EML_DA_PR.xlsx')

df['Turn'] = pd.to_numeric(df['Turn'], errors='coerce')
df['EML_label'] = pd.to_numeric(df['EML_label'], errors='coerce').fillna(0).astype(int)
df = df.dropna(subset=['Dialogue_ID', 'Turn', 'Role'])  #  FIX: speaker → Role
df = df.sort_values(['Dialogue_ID', 'Turn']).reset_index(drop=True)

# 只保留有效PR
pr_valid = df[df['PR_label'].isin(VALID_PR_LABELS)].copy()
pr_valid = pr_valid.sort_values(['Dialogue_ID', 'Turn']).reset_index(drop=True)

# ----------------------
# 2. 提取 PR 转移（✅ 双条件：Same Role + Turn Gap ≤3）
# ----------------------
transitions = []

for dia_id, dia_df in pr_valid.groupby('Dialogue_ID'):
    dia_df = dia_df.sort_values('Turn').reset_index(drop=True)
    pr_list = dia_df.to_dict('records')

    # 遍历连续PR对
    for i in range(len(pr_list) - 1):
        prev = pr_list[i]
        next_pr = pr_list[i+1]

        # ===================== 核心条件 =====================
        # 1. 必须是同一个 Role（FIX: speaker → Role）
        if prev['Role'] != next_pr['Role']:
            continue

        # 2. 必须满足 turn 差 ≤ 3
        turn_gap = next_pr['Turn'] - prev['Turn']
        if turn_gap > TURN_GAP_THRESHOLD:
            continue
        # ====================================================

        # 检查两个PR之间是否有EML
        between_df = df[
            (df['Dialogue_ID'] == dia_id) &
            (df['Turn'] > prev['Turn']) &
            (df['Turn'] < next_pr['Turn'])
        ]
        has_eml_between = int((between_df['EML_label'] == 1).any())

        transitions.append({
            'Dialogue_ID': dia_id,
            'Role': prev['Role'],  # FIX: Speaker → Role
            'Turn_Prev': prev['Turn'],
            'Turn_Next': next_pr['Turn'],
            'Turn_Gap': turn_gap,
            'PR_Prev': prev['PR_label'],
            'PR_Next': next_pr['PR_label'],
            'Transition': f"{prev['PR_label']}→{next_pr['PR_label']}",
            'Has_EML_Between': 'With_EML' if has_eml_between else 'Without_EML'
        })

trans_df = pd.DataFrame(transitions)

# ----------------------
# 3. 统计 With / Without EML
# ----------------------
transition_types = [
    'Compliance→Compliance',
    'Compliance→Resistance',
    'Resistance→Compliance',
    'Resistance→Resistance'
]

def count_group(trans_df, group_name):
    counts = []
    group = trans_df[trans_df['Has_EML_Between'] == group_name]
    for t in transition_types:
        cnt = (group['Transition'] == t).sum()
        counts.append(cnt)
    return counts

with_eml_counts = count_group(trans_df, 'With_EML')
without_eml_counts = count_group(trans_df, 'Without_EML')

summary = pd.DataFrame({
    'Transition': transition_types,
    'With_EML': with_eml_counts,
    'Without_EML': without_eml_counts
})

total_with = summary['With_EML'].sum()
total_without = summary['Without_EML'].sum()

summary['With_EML_Ratio'] = (summary['With_EML'] / total_with * 100) if total_with > 0 else 0
summary['Without_EML_Ratio'] = (summary['Without_EML'] / total_without * 100) if total_without > 0 else 0

# ----------------------
# 4. 转移矩阵
# ----------------------
def build_matrix(counts):
    mat = np.array([
        [counts[0], counts[1]],
        [counts[2], counts[3]]
    ], dtype=float)
    ratio = mat / mat.sum() if mat.sum() > 0 else mat
    labels = ['Compliance', 'Resistance']
    return pd.DataFrame(ratio, index=labels, columns=labels), pd.DataFrame(mat, index=labels, columns=labels)

mat_with_ratio, mat_with_abs = build_matrix(with_eml_counts)
mat_without_ratio, mat_without_abs = build_matrix(without_eml_counts)

# ----------------------
# 5. 热力图
# ----------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(mat_with_ratio, annot=True, fmt='.2f', cmap='Blues', ax=ax1, vmin=0, vmax=1)
ax1.set_title('PR Transition (With EML between PRs)')
ax1.set_xlabel('Next PR')
ax1.set_ylabel('Current PR')

sns.heatmap(mat_without_ratio, annot=True, fmt='.2f', cmap='Oranges', ax=ax2, vmin=0, vmax=1)
ax2.set_title('PR Transition (Without EML between PRs)')
ax2.set_xlabel('Next PR')
ax2.set_ylabel('Current PR')

plt.tight_layout()
plt.savefig('exp3_final_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------
# 6. 有向图
# ----------------------
def plot_graph(counts, title, save_path):
    G = nx.DiGraph()
    G.add_nodes_from(['Compliance', 'Resistance'])
    edges = [
        ('Compliance', 'Compliance', counts[0]),
        ('Compliance', 'Resistance', counts[1]),
        ('Resistance', 'Compliance', counts[2]),
        ('Resistance', 'Resistance', counts[3])
    ]
    edges = [(u, v, w) for u, v, w in edges if w > 0]
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=['#1f77b4','#ff7f0e'])
    max_w = max([w for _,_,w in edges]) if edges else 1
    widths = [w/max_w*5 for _,_,w in edges]
    nx.draw_networkx_edges(G, pos, width=widths, arrowstyle='->', arrowsize=20, alpha=0.6)
    nx.draw_networkx_edge_labels(G, pos, {(u,v): str(w) for u,v,w in edges}, font_size=10)
    nx.draw_networkx_labels(G, pos, font_weight='bold')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if total_with > 0:
    plot_graph(with_eml_counts, 'With EML between PRs', 'exp3_final_graph_with.png')
if total_without > 0:
    plot_graph(without_eml_counts, 'Without EML between PRs', 'exp3_final_graph_without.png')

# ----------------------
# 7. 柱状图（带百分比）
# ----------------------
plt.figure(figsize=(12,6))
x = np.arange(len(summary))
w = 0.35

bars1 = plt.bar(x - w/2, summary['With_EML_Ratio'], w, label='With EML', color='#1f77b4')
bars2 = plt.bar(x + w/2, summary['Without_EML_Ratio'], w, label='Without EML', color='#ff7f0e')

plt.xticks(x, summary['Transition'], rotation=15, ha='right')
plt.ylabel('Ratio (%)')
plt.title('PR Transition (Same Role, Turn Gap ≤3)')
plt.legend()

# 显示百分比
for bar in bars1:
    height = bar.get_height()
    if height > 0:
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
for bar in bars2:
    height = bar.get_height()
    if height > 0:
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('exp3_final_bar.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------
# 8. 稳定性 & 切换率
# ----------------------
def stability(counts):
    total = sum(counts)
    if total == 0:
        return 0, 0
    stable = counts[0] + counts[3]
    switch = total - stable
    return (stable/total*100), (switch/total*100)

stab_with, switch_with = stability(with_eml_counts)
stab_without, switch_without = stability(without_eml_counts)

# ----------------------
# 9. 输出最终结果
# ----------------------
print("=" * 80)
print("              Experiment 3 FINAL: Same Role + Turn Gap ≤3")
print("=" * 80)
print("\nTransition Counts & Ratios:")
print(summary.round(2))

print("\nWith EML Matrix:")
print(mat_with_abs.astype(int))

print("\nWithout EML Matrix:")
print(mat_without_abs.astype(int))

print("\nStability Analysis:")
print(f"With EML : Stability={stab_with:.2f}%, Switch={switch_with:.2f}%")
print(f"Without EML : Stability={stab_without:.2f}%, Switch={switch_without:.2f}%")

print("\nSample Size:")
print(f"Total valid transitions WITH EML: {total_with}")
print(f"Total valid transitions WITHOUT EML: {total_without}")

# ----------------------
# 10. 保存Excel
# ----------------------
with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as f:
    summary.to_excel(f, sheet_name='Summary', index=False)
    mat_with_abs.to_excel(f, sheet_name='With_EML_Matrix')
    mat_without_abs.to_excel(f, sheet_name='Without_EML_Matrix')
    trans_df.to_excel(f, sheet_name='All_Valid_Transitions', index=False)

print(f"\n✅ Final results saved to: {OUTPUT_EXCEL}")