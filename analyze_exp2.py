import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import warnings

warnings.filterwarnings('ignore')

# Set plot font and style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# ----------------------
# 1. Data Loading and Preprocessing
# ----------------------
# Load dataset
df = pd.read_excel('output_data/all_turns_data_with_EML_DA_PR.xlsx')

# Ensure key columns are numeric
df['Turn'] = pd.to_numeric(df['Turn'], errors = 'coerce')
df['EML_label'] = pd.to_numeric(df['EML_label'], errors = 'coerce')

# Filter valid PR records (only Compliance/Resistance, exclude Neutral)
pr_valid_labels = ['Compliance', 'Resistance']
pr_data = df[df['PR_label'].isin(pr_valid_labels)].copy()

# ----------------------
# 2. Reclassify Dialogues (Fix Core Logic)
# ----------------------
# Step 1: Mark dialogues with/without EML (simplified logic)
eml_dialogues = df[df['EML_label'] == 1]['Dialogue_ID'].unique().tolist()
pr_data['has_eml'] = pr_data['Dialogue_ID'].apply(lambda x: 'With_EML' if x in eml_dialogues else 'Without_EML')

# Step 2: Filter dialogues with ≥2 PR occurrences
dialogue_pr_counts = pr_data.groupby('Dialogue_ID')['PR_label'].count().reset_index()
dialogue_pr_counts.columns = ['Dialogue_ID', 'pr_count']
qualified_dialogues = dialogue_pr_counts[dialogue_pr_counts['pr_count'] >= 2]['Dialogue_ID'].tolist()

# Filter PR data to qualified dialogues only
pr_data_qualified = pr_data[pr_data['Dialogue_ID'].isin(qualified_dialogues)].copy()


# ----------------------
# 3. Extract PR Transitions (Fixed)
# ----------------------
def extract_pr_transitions(dialogue_ids, pr_data, group_flag):
    """Extract PR transition patterns for given dialogues"""
    transitions = {
        'Compliance→Compliance': 0,
        'Compliance→Resistance': 0,
        'Resistance→Compliance': 0,
        'Resistance→Resistance': 0
    }
    all_sequences = []

    for dialogue_id in dialogue_ids:
        # Get sorted PR records for the dialogue
        dialogue_pr = pr_data[
            (pr_data['Dialogue_ID'] == dialogue_id) &
            (pr_data['has_eml'] == group_flag)
            ].sort_values('Turn')
        pr_labels = dialogue_pr['PR_label'].tolist()

        # Extract transitions (t → t+1)
        for i in range(len(pr_labels) - 1):
            current = pr_labels[i]
            next_label = pr_labels[i + 1]
            transition_key = f"{current}→{next_label}"
            if transition_key in transitions:
                transitions[transition_key] += 1

        # Store full sequence (for reference)
        all_sequences.append({
            'dialogue_id': dialogue_id,
            'pr_sequence': pr_labels,
            'has_eml': group_flag
        })

    return transitions, all_sequences


# Split into two valid groups
with_eml_dialogues = pr_data_qualified[pr_data_qualified['has_eml'] == 'With_EML']['Dialogue_ID'].unique().tolist()
without_eml_dialogues = pr_data_qualified[pr_data_qualified['has_eml'] == 'Without_EML'][
    'Dialogue_ID'].unique().tolist()

# Extract transitions for both groups
transitions_with_eml, sequences_with_eml = extract_pr_transitions(with_eml_dialogues, pr_data_qualified, 'With_EML')
transitions_without_eml, sequences_without_eml = extract_pr_transitions(without_eml_dialogues, pr_data_qualified,
                                                                        'Without_EML')

# ----------------------
# 4. Calculate Transition Frequencies/Ratios (Fixed)
# ----------------------
# Convert to DataFrame for analysis
transition_df = pd.DataFrame({
    'Transition': list(transitions_with_eml.keys()),
    'With_EML': list(transitions_with_eml.values()),
    'Without_EML': list(transitions_without_eml.values())
})

# Calculate total transitions for ratio calculation
total_with_eml = transition_df['With_EML'].sum()
total_without_eml = transition_df['Without_EML'].sum()

# Avoid division by zero
transition_df['With_EML_Ratio'] = transition_df['With_EML'].apply(
    lambda x: (x / total_with_eml * 100) if total_with_eml > 0 else 0)
transition_df['Without_EML_Ratio'] = transition_df['Without_EML'].apply(
    lambda x: (x / total_without_eml * 100) if total_without_eml > 0 else 0)


# ----------------------
# 5. Transition Diagram 1: Transition Matrix Heatmap (核心)
# ----------------------
def create_transition_matrix(transition_dict):
    """Convert transition dict to matrix"""
    # Define order of labels
    labels = ['Compliance', 'Resistance']
    matrix = np.zeros((2, 2))

    # Fill matrix
    matrix[0, 0] = transition_dict['Compliance→Compliance']
    matrix[0, 1] = transition_dict['Compliance→Resistance']
    matrix[1, 0] = transition_dict['Resistance→Compliance']
    matrix[1, 1] = transition_dict['Resistance→Resistance']

    # Normalize to ratio (0-1) for heatmap
    matrix_ratio = matrix / matrix.sum() if matrix.sum() > 0 else matrix
    return pd.DataFrame(matrix_ratio, index = labels, columns = labels), pd.DataFrame(matrix, index = labels,
                                                                                      columns = labels)


# Create matrices for both groups
matrix_with_eml_ratio, matrix_with_eml_abs = create_transition_matrix(transitions_with_eml)
matrix_without_eml_ratio, matrix_without_eml_abs = create_transition_matrix(transitions_without_eml)

# Plot heatmap (two subplots for comparison)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (14, 6))

# Heatmap for With EML
sns.heatmap(matrix_with_eml_ratio, annot = True, fmt = '.2f', cmap = 'Blues',
            cbar = True, ax = ax1, vmin = 0, vmax = 1,
            annot_kws = {"size": 12}, cbar_kws = {'label': 'Transition Ratio'})
ax1.set_title('PR Transition Matrix (With EML)', fontsize = 14, pad = 20)
ax1.set_xlabel('Next PR Label', fontsize = 12)
ax1.set_ylabel('Current PR Label', fontsize = 12)

# Heatmap for Without EML
sns.heatmap(matrix_without_eml_ratio, annot = True, fmt = '.2f', cmap = 'Oranges',
            cbar = True, ax = ax2, vmin = 0, vmax = 1,
            annot_kws = {"size": 12}, cbar_kws = {'label': 'Transition Ratio'})
ax2.set_title('PR Transition Matrix (Without EML)', fontsize = 14, pad = 20)
ax2.set_xlabel('Next PR Label', fontsize = 12)
ax2.set_ylabel('Current PR Label', fontsize = 12)

plt.tight_layout()
plt.savefig('pr_transition_matrix_heatmap.png', dpi = 300, bbox_inches = 'tight')
plt.show()


# ----------------------
# 6. Transition Diagram 2: Directed Graph (有向转移图)
# ----------------------
def plot_transition_graph(transition_dict, title, filename):
    """Plot directed graph for PR transitions"""
    G = nx.DiGraph()
    labels = ['Compliance', 'Resistance']
    G.add_nodes_from(labels)

    # Add edges with weights
    edges = [
        ('Compliance', 'Compliance', transition_dict['Compliance→Compliance']),
        ('Compliance', 'Resistance', transition_dict['Compliance→Resistance']),
        ('Resistance', 'Compliance', transition_dict['Resistance→Compliance']),
        ('Resistance', 'Resistance', transition_dict['Resistance→Resistance'])
    ]

    # Filter out zero-weight edges
    edges = [(u, v, w) for u, v, w in edges if w > 0]
    for u, v, w in edges:
        G.add_edge(u, v, weight = w)

    # Set layout
    pos = nx.spring_layout(G, seed = 42, k = 3)  # Fixed seed for consistency

    # Plot nodes
    nx.draw_networkx_nodes(G, pos, node_size = 3000, node_color = ['#1f77b4', '#ff7f0e'],
                           alpha = 0.8, ax = None)

    # Plot edges with width proportional to weight
    max_weight = max([w for _, _, w in edges]) if edges else 1
    edge_widths = [w / max_weight * 5 for _, _, w in edges]
    nx.draw_networkx_edges(G, pos, width = edge_widths, alpha = 0.6,
                           arrowstyle = '->', arrowsize = 20)

    # Add edge labels (transition count)
    edge_labels = {(u, v): f'{w}' for u, v, w in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels,
                                 font_size = 10, label_pos = 0.3)

    # Add node labels
    nx.draw_networkx_labels(G, pos, font_size = 12, font_weight = 'bold')

    # Set title and save
    plt.title(title, fontsize = 14, pad = 20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi = 300, bbox_inches = 'tight')
    plt.show()


# Plot directed graph for With EML (if data exists)
if total_with_eml > 0:
    plot_transition_graph(transitions_with_eml,
                          'PR Transition Directed Graph (With EML)',
                          'pr_transition_graph_with_eml.png')

# Plot directed graph for Without EML (if data exists)
if total_without_eml > 0:
    plot_transition_graph(transitions_without_eml,
                          'PR Transition Directed Graph (Without EML)',
                          'pr_transition_graph_without_eml.png')

# ----------------------
# 7. Fixed Transition Frequency Bar Chart
# ----------------------
plt.figure(figsize = (12, 7))
x = np.arange(len(transition_df['Transition']))
width = 0.35

# Create grouped bar chart
bars1 = plt.bar(x - width / 2, transition_df['With_EML_Ratio'], width,
                label = 'With EML', color = '#1f77b4', alpha = 0.8)
bars2 = plt.bar(x + width / 2, transition_df['Without_EML_Ratio'], width,
                label = 'Without EML', color = '#ff7f0e', alpha = 0.8)

# Add labels and title
plt.xlabel('PR Transition Pattern', fontsize = 12)
plt.ylabel('Transition Ratio (%)', fontsize = 12)
plt.title('PR Transition Frequencies: EML vs No EML', fontsize = 14, pad = 20)
plt.xticks(x, transition_df['Transition'], rotation = 15, ha = 'right')
plt.legend(fontsize = 10)

# Add value labels on bars (avoid empty labels)
for bar in bars1:
    height = bar.get_height()
    if height > 0:
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{height:.1f}%', ha = 'center', va = 'bottom', fontsize = 9)

for bar in bars2:
    height = bar.get_height()
    if height > 0:
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{height:.1f}%', ha = 'center', va = 'bottom', fontsize = 9)

plt.tight_layout()
plt.savefig('pr_transition_frequency.png', dpi = 300, bbox_inches = 'tight')
plt.show()


# ----------------------
# 8. Stability Analysis (Fixed)
# ----------------------
def calculate_stability(transition_dict):
    total = sum(transition_dict.values())
    if total == 0:
        return 0
    stable = transition_dict['Compliance→Compliance'] + transition_dict['Resistance→Resistance']
    return (stable / total) * 100


stability_with_eml = calculate_stability(transitions_with_eml)
stability_without_eml = calculate_stability(transitions_without_eml)

# Calculate switch rate
switch_with_eml = 100 - stability_with_eml
switch_without_eml = 100 - stability_without_eml

# ----------------------
# 9. Output Results Summary (Fixed)
# ----------------------
print("=== Experiment 2: PR Sequence Patterns Results (Fixed) ===")
print("\n1. Transition Frequencies:")
print(transition_df.round(2))

print("\n2. Transition Matrix (With EML) - Absolute Counts:")
print(matrix_with_eml_abs)

print("\n3. Transition Matrix (Without EML) - Absolute Counts:")
print(matrix_without_eml_abs)

print("\n4. Stability Analysis (% of same-label transitions):")
print(f"With EML: {stability_with_eml:.2f}%")
print(f"Without EML: {stability_without_eml:.2f}%")

print("\n5. Switch Rate (% of opposite-label transitions):")
print(f"With EML: {switch_with_eml:.2f}%")
print(f"Without EML: {switch_without_eml:.2f}%")

print("\n6. Sample Size:")
print(f"Dialogues with EML (≥2 PRs): {len(with_eml_dialogues)}")
print(f"Dialogues without EML (≥2 PRs): {len(without_eml_dialogues)}")
print(f"Total qualified dialogues: {len(qualified_dialogues)}")
print(f"Total transitions (With EML): {total_with_eml}")
print(f"Total transitions (Without EML): {total_without_eml}")

# Save results to Excel
result_summary = pd.DataFrame({
    'Metric': ['Stability_Rate (%)', 'Switch_Rate (%)', 'Total_Transitions', 'Qualified_Dialogues'],
    'With_EML': [stability_with_eml, switch_with_eml, total_with_eml, len(with_eml_dialogues)],
    'Without_EML': [stability_without_eml, switch_without_eml, total_without_eml, len(without_eml_dialogues)]
})
# Add transition matrix to Excel
with pd.ExcelWriter('pr_sequence_patterns_results_fixed.xlsx', engine = 'openpyxl') as writer:
    result_summary.to_excel(writer, sheet_name = 'Summary', index = False)
    matrix_with_eml_abs.to_excel(writer, sheet_name = 'Transition_Matrix_With_EML')
    matrix_without_eml_abs.to_excel(writer, sheet_name = 'Transition_Matrix_Without_EML')
    transition_df.to_excel(writer, sheet_name = 'Transition_Frequencies', index = False)