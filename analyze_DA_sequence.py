import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import warnings
import os
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

warnings.filterwarnings('ignore')

OUTPUT_DIR = 'output_data'
os.makedirs(OUTPUT_DIR, exist_ok = True)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_and_preprocess_results(file_path):
    df = pd.read_csv(file_path)

    df = df[df['persuasion_result'].notna()].copy()
    df = df[df['persuasion_result'].isin(['Compliance', 'Resistance'])].copy()

    def str_to_list(str_repr):
        try:
            if pd.isna(str_repr):
                return []
            clean_str = str_repr.strip('[]').strip()
            if not clean_str:
                return []
            elements = [elem.strip().strip("'\"").strip() for elem in clean_str.split(', ')]
            elements = [elem for elem in elements if elem]
            return elements
        except Exception as e:
            print(f"Error parsing DA sequence: {e}, Input: {str_repr}")
            return []

    df['da_sequence_list'] = df['da_sequence'].apply(str_to_list)

    df = df[df['da_sequence_list'].apply(len) > 0].copy()

    df = df.reset_index(drop = True)

    print(f"Data loaded successfully (Only Compliance/Resistance)")
    print(f"   Total valid conversations: {len(df)}")
    print(f"   Persuasion result distribution:")
    print(df['persuasion_result'].value_counts())

    if len(df) < 5:
        print(f"Insufficient data (need at least 5 conversations for analysis)")

    return df


def analyze_da_features_by_result(df):
    da_by_result = defaultdict(list)
    result_groups = df.groupby('persuasion_result')

    for result_type, group in result_groups:
        for da_list in group['da_sequence_list']:
            da_by_result[result_type].extend(da_list)

    da_stats_by_result = {}
    for result_type, da_list in da_by_result.items():
        da_counter = Counter(da_list)
        total_da = len(da_list)
        da_stats = {
            'count': da_counter,
            'percentage': {da: (count / total_da) * 100 for da, count in da_counter.items()},
            'total': total_da
        }
        da_stats_by_result[result_type] = da_stats

    print(f"\nTop 10 DA Labels by Persuasion Result")
    for result_type, stats in da_stats_by_result.items():
        print(f"\n{result_type} (Total DA occurrences: {stats['total']}):")
        top_da = stats['count'].most_common(10)
        for i, (da, count) in enumerate(top_da, 1):
            percentage = stats['percentage'][da]
            print(f"  {i:2d}. {da}: {count} occurrences ({percentage:.1f}%)")

    return da_stats_by_result


def extract_sequence_features(df):
    feature_data = []

    all_da = []
    for da_list in df['da_sequence_list']:
        all_da.extend(da_list)
    common_da = [da for da, count in Counter(all_da).most_common(15)]

    for idx, row in df.iterrows():
        da_list = row['da_sequence_list']
        result = row['persuasion_result']
        conv_id = row['conversation_id'] if 'conversation_id' in df.columns else idx
        seq_len = len(da_list)

        features = {
            'conversation_id': conv_id,
            'persuasion_result': result,
            'sequence_length': seq_len,
            'first_da': da_list[0] if seq_len > 0 else 'None',
            'last_da': da_list[-1] if seq_len > 0 else 'None',
            'da_diversity': len(set(da_list)) / seq_len if seq_len > 0 else 0
        }

        da_counter = Counter(da_list)
        for da in common_da:
            count = da_counter.get(da, 0)
            features[f'da_{da}_count'] = count
            features[f'da_{da}_ratio'] = count / seq_len if seq_len > 0 else 0

        features['has_request_inform'] = 1 if ('request' in da_list and 'inform' in da_list) else 0
        features['has_greeting_closing'] = 1 if ('greeting' in da_list and 'closing' in da_list) else 0
        features['starts_with_greeting'] = 1 if (seq_len > 0 and da_list[0] == 'greeting') else 0
        features['ends_with_thanks'] = 1 if (seq_len > 0 and da_list[-1] in ['thanks', 'greeting, thanks']) else 0

        feature_data.append(features)

    feature_df = pd.DataFrame(feature_data)

    print(f"\nSequence Feature Extraction Complete")
    print(f"   Number of features: {len(feature_df.columns) - 2} (excluding ID and label)")
    print(f"   Number of common DA labels: {len(common_da)}")

    return feature_df, common_da


def perform_association_rule_analysis(df):
    print(f"\n=== Association Rule Mining (DA Patterns -> Compliance/Resistance) ===")
    if len(df) < 5:
        print(f"Skip association rule mining: Insufficient data (need >=5 conversations)")
        return None

    transactions = []
    for _, row in df.iterrows():
        da_set = set(row['da_sequence_list'])
        result = row['persuasion_result']
        transaction = list(da_set) + [result]
        transactions.append(transaction)

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns = te.columns_)

    target_consequents = ['Compliance', 'Resistance']
    df_encoded = df_encoded[[col for col in df_encoded.columns if col in te.columns_]]

    min_support = 0.2
    frequent_itemsets = apriori(
        df_encoded,
        min_support = min_support,
        use_colnames = True,
        verbose = 0
    )
    if len(frequent_itemsets) == 0:
        print(f"No frequent itemsets found (min support: {min_support}), try lower support")
        return None

    min_confidence = 0.6
    rules = association_rules(
        frequent_itemsets,
        metric = "confidence",
        min_threshold = min_confidence
    )

    rules = rules[rules['consequents'].apply(
        lambda x: len(x) == 1 and list(x)[0] in target_consequents
    )]
    rules = rules[rules['antecedents'].apply(
        lambda x: not any(item in target_consequents for item in x)
    )]

    if len(rules) == 0:
        print(f"No valid rules found (DA -> Compliance/Resistance)")
        print(f"   Try reducing min_support (current: {min_support}) or min_confidence (current: {min_confidence})")
        return None

    rules = rules.sort_values('lift', ascending = False).reset_index(drop = True)
    rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    rules['antecedents'] = rules['antecedents'].apply(lambda x: set(x))
    rules['consequents'] = rules['consequents'].apply(lambda x: set(x))

    rule_dict = {
        'Compliance': rules[rules['consequents'].apply(lambda x: 'Compliance' in x)],
        'Resistance': rules[rules['consequents'].apply(lambda x: 'Resistance' in x)]
    }

    for result_type, rule_df in rule_dict.items():
        if len(rule_df) > 0:
            print(f"\nTop 10 Rules for {result_type} (DA Pattern -> {result_type}):")
            print(
                f"   Based on {len(df)} conversations | min_support: {min_support} | min_confidence: {min_confidence}")
            top_rules = rule_df.head(10)
            for i, (_, row) in enumerate(top_rules.iterrows(), 1):
                print(f"  {i:2d}. {row['antecedents']} -> {row['consequents']}")
                print(
                    f"     Support: {row['support']:.3f}, Confidence: {row['confidence']:.3f}, Lift: {row['lift']:.3f}")
            save_path = os.path.join(OUTPUT_DIR, f'association_rules_{result_type.lower()}_target.csv')
            rule_df.to_csv(save_path, index = False, encoding = 'utf-8')
            print(f"   Rules saved to: {save_path}")
        else:
            print(f"\nNo rules found for {result_type} (DA -> {result_type})")

    create_association_rule_visualization(rule_dict)

    return rule_dict


def create_association_rule_visualization(rule_dict):
    plot_data = []
    colors = {'Compliance': '#2E8B57', 'Resistance': '#DC143C'}
    result_types = ['Compliance', 'Resistance']

    for result_type in result_types:
        rule_df = rule_dict.get(result_type, pd.DataFrame())
        if len(rule_df) > 0:
            top_rules = rule_df.head(8)
            for _, row in top_rules.iterrows():
                ant_str = ', '.join(row['antecedents'])
                con_str = ', '.join(row['consequents'])
                rule_label = f"{ant_str[:25]}..." if len(ant_str) > 25 else ant_str
                plot_data.append({
                    'result_type': result_type,
                    'rule': rule_label,
                    'support': row['support'],
                    'confidence': row['confidence'],
                    'lift': row['lift']
                })

    if not plot_data:
        print(f"\nNo association rules to visualize")
        return

    plot_df = pd.DataFrame(plot_data)
    fig, axes = plt.subplots(1, 2, figsize = (22, 10))
    fig.suptitle('Association Rules: DA Patterns -> Persuasion Result (Compliance/Resistance)',
                 fontsize = 18, fontweight = 'bold', y = 0.98)

    ax1 = axes[0]
    for result_type in result_types:
        type_data = plot_df[plot_df['result_type'] == result_type]
        if len(type_data) > 0:
            ax1.scatter(
                type_data['confidence'],
                type_data['lift'],
                s = type_data['support'] * 1500,
                c = colors[result_type],
                label = result_type,
                alpha = 0.7,
                edgecolors = 'black',
                linewidth = 1,
                marker = 'o'
            )
    ax1.set_xlabel('Confidence (Probability of Result given DA Pattern)', fontsize = 12)
    ax1.set_ylabel('Lift (Strength of Association)', fontsize = 12)
    ax1.set_title('Confidence vs Lift (Bubble Size = Support)', fontsize = 14, fontweight = 'bold')
    ax1.legend(loc = 'upper right')
    ax1.grid(True, alpha = 0.3)
    ax1.axhline(y = 1, color = 'red', linestyle = '--', alpha = 0.5, label = 'No Association (Lift=1)')

    ax2 = axes[1]
    top_12_rules = plot_df.nlargest(12, 'lift')
    y_pos = np.arange(len(top_12_rules))
    lifts = top_12_rules['lift'].values
    rule_labels = top_12_rules['rule'].values
    bar_colors = [colors[rt] for rt in top_12_rules['result_type']]

    ax2.barh(y_pos, lifts, color = bar_colors, alpha = 0.8, edgecolor = 'black', linewidth = 1)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(rule_labels, fontsize = 9)
    ax2.set_xlabel('Lift Score (Higher = Stronger Association)', fontsize = 12)
    ax2.set_title('Top Association Rules by Lift Score', fontsize = 14, fontweight = 'bold')
    ax2.grid(True, alpha = 0.3, axis = 'x')
    ax2.invert_yaxis()
    for i, v in enumerate(lifts):
        ax2.text(v + 0.05, i, f'{v:.2f}', va = 'center', fontsize = 8, fontweight = 'bold')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'association_rules_target_analysis.png')
    plt.savefig(save_path, dpi = 300, bbox_inches = 'tight', facecolor = 'white')
    plt.close()
    print(f"\nAssociation rule visualization saved to: {save_path}")


def create_pattern_visualizations(df, da_stats, feature_df):
    fig, axes = plt.subplots(2, 2, figsize = (18, 14))
    fig.suptitle('DA Sequence Pattern Analysis (Compliance vs Resistance)',
                 fontsize = 18, fontweight = 'bold', y = 0.98)
    colors = {'Compliance': '#2E8B57', 'Resistance': '#DC143C'}
    result_types = ['Compliance', 'Resistance']
    width = 0.35

    ax1 = axes[0, 0]
    data_for_box = [feature_df[feature_df['persuasion_result'] == rt]['sequence_length'] for rt in result_types]
    bp1 = ax1.boxplot(data_for_box, labels = result_types, patch_artist = True)
    for patch, color in zip(bp1['boxes'], [colors[rt] for rt in result_types]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_title('DA Sequence Length Distribution', fontsize = 14, fontweight = 'bold')
    ax1.set_ylabel('Sequence Length (Number of Utterances)', fontsize = 12)
    ax1.grid(True, alpha = 0.3)

    ax2 = axes[0, 1]
    all_top_da = []
    for rt in result_types:
        if rt in da_stats:
            top_da = [da for da, _ in da_stats[rt]['count'].most_common(5)]
            all_top_da.extend(top_da)
    common_top_da = [da for da, _ in Counter(all_top_da).most_common(5)]
    x = np.arange(len(common_top_da))
    for i, rt in enumerate(result_types):
        if rt in da_stats:
            percentages = [da_stats[rt]['percentage'].get(da, 0) for da in common_top_da]
            ax2.bar(x + i * width / 2, percentages, width / 2, label = rt, color = colors[rt], alpha = 0.8)
    ax2.set_title('Top 5 DA Label Percentage', fontsize = 14, fontweight = 'bold')
    ax2.set_ylabel('Percentage (%)', fontsize = 12)
    ax2.set_xticks(x + width / 4)
    ax2.set_xticklabels(common_top_da, rotation = 45, ha = 'right')
    ax2.legend()
    ax2.grid(True, alpha = 0.3)

    ax3 = axes[1, 0]
    first_da_data = defaultdict(dict)
    for rt in result_types:
        rt_data = feature_df[feature_df['persuasion_result'] == rt]
        if len(rt_data) > 0:
            first_da_counter = Counter(rt_data['first_da'])
            total = len(rt_data)
            for da, count in first_da_counter.most_common(6):
                first_da_data[da][rt] = (count / total) * 100
    da_labels = list(first_da_data.keys())[:6]
    x = np.arange(len(da_labels))
    for i, rt in enumerate(result_types):
        percentages = [first_da_data[da].get(rt, 0) for da in da_labels]
        ax3.bar(x + i * width / 2, percentages, width / 2, label = rt, color = colors[rt], alpha = 0.8)
    ax3.set_title('First DA Label Distribution (Top 6)', fontsize = 14, fontweight = 'bold')
    ax3.set_ylabel('Percentage (%)', fontsize = 12)
    ax3.set_xticks(x + width / 4)
    ax3.set_xticklabels(da_labels, rotation = 45, ha = 'right')
    ax3.legend()
    ax3.grid(True, alpha = 0.3)

    ax4 = axes[1, 1]
    diversity_data = [feature_df[feature_df['persuasion_result'] == rt]['da_diversity'] for rt in result_types]
    bp4 = ax4.boxplot(diversity_data, labels = result_types, patch_artist = True)
    for patch, color in zip(bp4['boxes'], [colors[rt] for rt in result_types]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax4.set_title('DA Diversity Distribution', fontsize = 14, fontweight = 'bold')
    ax4.set_ylabel('DA Diversity (Unique DA / Total Length)', fontsize = 12)
    ax4.grid(True, alpha = 0.3)

    save_path = os.path.join(OUTPUT_DIR, 'da_sequence_pattern_analysis.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi = 300, bbox_inches = 'tight', facecolor = 'white')
    plt.close()
    print(f"\nDA pattern visualization saved to: {save_path}")


def build_prediction_model(feature_df):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score

    if len(feature_df) < 10:
        print(f"\nSkip model building: Insufficient data (need >=10 conversations)")
        return None, None

    feature_cols = [col for col in feature_df.columns if col not in ['conversation_id', 'persuasion_result']]
    le = LabelEncoder()
    feature_df_encoded = feature_df.copy()
    for col in ['first_da', 'last_da']:
        feature_df_encoded[f'{col}_encoded'] = le.fit_transform(feature_df_encoded[col])
    feature_cols = [col for col in feature_cols if col not in ['first_da', 'last_da']]
    feature_cols.extend(['first_da_encoded', 'last_da_encoded'])

    X = feature_df_encoded[feature_cols]
    y = feature_df_encoded['persuasion_result']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.3, random_state = 42, stratify = y
    )

    rf_model = RandomForestClassifier(
        n_estimators = 100, random_state = 42, max_depth = 10, class_weight = 'balanced'
    )
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    print(f"\nPrediction Model Evaluation (Compliance vs Resistance)")
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names = ['Compliance', 'Resistance']))

    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending = False).reset_index(drop = True)

    print(f"\nTop 15 Predictive Features:")
    top_features = feature_importance.head(15)
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        print(f"  {i:2d}. {row['feature']}: {row['importance']:.4f}")

    save_path = os.path.join(OUTPUT_DIR, 'key_predictive_features.csv')
    top_features.to_csv(save_path, index = False, encoding = 'utf-8')
    print(f"Top features saved to: {save_path}")

    plt.figure(figsize = (12, 8))
    top_10 = feature_importance.head(10)
    plt.barh(range(len(top_10)), top_10['importance'], color = '#4CAF50', alpha = 0.8, edgecolor = 'black')
    plt.yticks(range(len(top_10)), top_10['feature'], fontsize = 10)
    plt.xlabel('Feature Importance Score', fontsize = 12)
    plt.title('Top 10 Predictive Features (Compliance vs Resistance)', fontsize = 14, fontweight = 'bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha = 0.3, axis = 'x')
    plt.tight_layout()
    imp_save_path = os.path.join(OUTPUT_DIR, 'feature_importance_analysis.png')
    plt.savefig(imp_save_path, dpi = 300, bbox_inches = 'tight', facecolor = 'white')
    plt.close()
    print(f"Feature importance plot saved to: {imp_save_path}")

    return rf_model, feature_importance


def generate_pattern_summary(df, feature_df, da_stats, association_rules, feature_importance=None):
    total_conv = len(df)
    comp_conv = len(df[df['persuasion_result'] == 'Compliance'])
    res_conv = len(df[df['persuasion_result'] == 'Resistance'])
    comp_seq_len = feature_df[feature_df['persuasion_result'] == 'Compliance']['sequence_length'].mean()
    res_seq_len = feature_df[feature_df['persuasion_result'] == 'Resistance']['sequence_length'].mean()

    summary = f"""# DA Sequence Patterns -> Persuasion Result Analysis Report
## Focus: Compliance vs Resistance
*Report generated automatically from DA sequence mining analysis*

## Data Overview
- **Total Valid Conversations**: {total_conv}
- **Compliance Conversations**: {comp_conv} ({(comp_conv / total_conv) * 100:.1f}%)
- **Resistance Conversations**: {res_conv} ({(res_conv / total_conv) * 100:.1f}%)
- **Average Sequence Length (Compliance)**: {comp_seq_len:.2f} utterances
- **Average Sequence Length (Resistance)**: {res_seq_len:.2f} utterances

## Key DA Features by Persuasion Result
### 2.1 Compliance
- **Primary DA Labels (Top 5)**:
"""
    if 'Compliance' in da_stats:
        top_da = da_stats['Compliance']['count'].most_common(5)
        for da, count in top_da:
            pct = da_stats['Compliance']['percentage'][da]
            summary += f"  - {da}: {count} occurrences ({pct:.1f}%)\n"
    comp_first_da = Counter(feature_df[feature_df['persuasion_result'] == 'Compliance']['first_da']).most_common(3)
    summary += f"- **Common Opening DA (Top 3)**: \n"
    for da, count in comp_first_da:
        pct = (count / comp_conv) * 100
        summary += f"  - {da}: {pct:.1f}%\n"

    summary += f"""
### 2.2 Resistance
- **Primary DA Labels (Top 5)**:
"""
    if 'Resistance' in da_stats:
        top_da = da_stats['Resistance']['count'].most_common(5)
        for da, count in top_da:
            pct = da_stats['Resistance']['percentage'][da]
            summary += f"  - {da}: {count} occurrences ({pct:.1f}%)\n"
    res_first_da = Counter(feature_df[feature_df['persuasion_result'] == 'Resistance']['first_da']).most_common(3)
    summary += f"- **Common Opening DA (Top 3)**: \n"
    for da, count in res_first_da:
        pct = (count / res_conv) * 100
        summary += f"  - {da}: {pct:.1f}%\n"

    summary += f"""
## Core Finding: DA Patterns -> Persuasion Result Association Rules
### Key Metrics Definition
- **Support**: Proportion of conversations with the DA pattern (0-1)
- **Confidence**: Probability of the result given the DA pattern (0-1)
- **Lift**: Strength of association (**>1 = positive association**, <1 = negative association)

### 3.1 DA Patterns Leading to Compliance
"""
    if association_rules and 'Compliance' in association_rules and len(association_rules['Compliance']) > 0:
        comp_rules = association_rules['Compliance'].head(6)
        for i, (_, row) in enumerate(comp_rules.iterrows(), 1):
            ant = row['antecedents']
            con = row['consequents']
            s = row['support']
            c = row['confidence']
            l = row['lift']
            summary += f"{i}. **{ant}** -> {con}\n"
            summary += f"   Support: {s:.3f} | Confidence: {c:.3f} | Lift: {l:.3f}\n"
    else:
        summary += "No significant DA patterns found for Compliance (adjust min_support/min_confidence)\n"

    summary += f"""
### 3.2 DA Patterns Leading to Resistance
"""
    if association_rules and 'Resistance' in association_rules and len(association_rules['Resistance']) > 0:
        res_rules = association_rules['Resistance'].head(6)
        for i, (_, row) in enumerate(res_rules.iterrows(), 1):
            ant = row['antecedents']
            con = row['consequents']
            s = row['support']
            c = row['confidence']
            l = row['lift']
            summary += f"{i}. **{ant}** -> {con}\n"
            summary += f"   Support: {s:.3f} | Confidence: {c:.3f} | Lift: {l:.3f}\n"
    else:
        summary += "No significant DA patterns found for Resistance (adjust min_support/min_confidence)\n"

    if feature_importance is not None and len(feature_importance) > 0:
        summary += f"""
## Top Predictive Features (Top 10)
"""
        top_10 = feature_importance.head(10)
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            summary += f"{i:2d}. {row['feature']}: Importance Score = {row['importance']:.4f}\n"

    summary += f"""
## Practical Recommendations
### For Improving Compliance Rate
1. Prioritize DA patterns with **Lift > 1.2** for Compliance (identified in association rules)
2. Control DA sequence length to the optimal range ({comp_seq_len:.0f}±1 utterances)
3. Use the top opening DA for Compliance (e.g., {comp_first_da[0][0]}) as conversation start
4. Combine high-frequency DA labels for Compliance (e.g., {[da for da, _ in da_stats['Compliance']['count'].most_common(2)]})

### For Avoiding Resistance
1. Avoid DA patterns with **Lift > 1.2** for Resistance (identified in association rules)
2. Avoid overly long sequences (> {res_seq_len:.0f} utterances) which correlate with Resistance
3. Minimize use of high-frequency DA labels for Resistance (e.g., {[da for da, _ in da_stats['Resistance']['count'].most_common(2)]})
4. Monitor DA diversity - high diversity is a key indicator of potential Resistance

### Parameter Tuning Tips (if few rules found)
- Reduce `min_support` (current: 0.2) for small datasets (<=50 conversations)
- Reduce `min_confidence` (current: 0.6) to capture more weak but meaningful patterns
- Keep only **unique DA** in each conversation (avoids overcounting duplicate DA)

---
*Report generated at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Based on {total_conv} Compliance/Resistance conversations analysis*
"""

    save_path = os.path.join(OUTPUT_DIR, 'da_pattern_to_result_analysis_report.md')
    with open(save_path, 'w', encoding = 'utf-8') as f:
        f.write(summary)
    print(f"\nComprehensive analysis report saved to: {save_path}")


if __name__ == "__main__":

    INPUT_FILE = 'output_data/conversation_analysis_results.csv'

    try:
        conversation_df = load_and_preprocess_results(INPUT_FILE)
        if len(conversation_df) == 0:
            raise Exception("No valid Compliance/Resistance data found")

        da_statistics = analyze_da_features_by_result(conversation_df)

        sequence_features_df, common_da_labels = extract_sequence_features(conversation_df)

        association_rules_results = perform_association_rule_analysis(conversation_df)

        create_pattern_visualizations(conversation_df, da_statistics, sequence_features_df)

        model, feature_importance = build_prediction_model(sequence_features_df)

        generate_pattern_summary(
            conversation_df,
            sequence_features_df,
            da_statistics,
            association_rules_results,
            feature_importance
        )

        print(f"\nAll analysis completed successfully!")
        print(f"\nAll output files saved to: {os.path.abspath(OUTPUT_DIR)}")
        print(f"\nKey output files:")
        print(f"  1. association_rules_compliance_target.csv - DA->Compliance rules")
        print(f"  2. association_rules_resistance_target.csv - DA->Resistance rules")
        print(f"  3. association_rules_target_analysis.png - Rule visualization")
        print(f"  4. da_pattern_to_result_analysis_report.md - Full analysis report")

    except FileNotFoundError:
        print(f"\nError: Input file not found - {INPUT_FILE}")
        print(f"   Please check the file path and ensure the CSV exists")
    except Exception as e:
        print(f"\nError occurred during execution: {str(e)}")
        raise