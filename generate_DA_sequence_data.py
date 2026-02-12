import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


def load_and_preprocess_data(file_path):
    """
    Load the conversation data and perform basic preprocessing

    Args:
        file_path (str): Path to the input CSV file

    Returns:
        pd.DataFrame: Preprocessed dataframe with clean data
    """
    # Read CSV file with error handling for encoding issues
    try:
        df = pd.read_csv(file_path, encoding = 'utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding = 'latin-1')

    # Verify key columns exist
    required_columns = ['conversation_id', 'prediction label', 'prediction_label']
    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(
        f"✅ Successfully loaded data with {len(df)} utterances and {df['conversation_id'].nunique()} unique conversations")
    return df


def process_conversation_group(group):
    """
    Process a single conversation group to generate DA sequence and persuasion result

    Args:
        group (pd.DataFrame): DataFrame containing one conversation's utterances

    Returns:
        dict: Processed results with conversation ID, DA sequence, and persuasion result
    """
    # Sort by original index to maintain utterance order
    group_sorted = group.sort_index()

    # Generate DA (Dialogue Act) sequence in original order
    da_sequence = group_sorted['prediction label'].tolist()

    # Get unique persuasion labels in this conversation
    persuasion_labels = group_sorted['prediction_label'].unique()

    # Determine persuasion result based on label combinations
    if len(persuasion_labels) == 1:
        # Single label type
        persuasion_result = persuasion_labels[0]
        # Mark as None if all are Neutral (no need to record)
        if persuasion_result == 'Neutral':
            persuasion_result = None
    else:
        # Multiple label types - check for Compliance/Resistance combinations
        has_compliance = 'Compliance' in persuasion_labels
        has_resistance = 'Resistance' in persuasion_labels

        if has_compliance and has_resistance:
            # Both Compliance and Resistance exist
            persuasion_result = 'Conflict'
        elif has_compliance:
            # Only Compliance (may include Neutral)
            persuasion_result = 'Compliance'
        elif has_resistance:
            # Only Resistance (may include Neutral)
            persuasion_result = 'Resistance'
        else:
            # Only Neutral combinations
            persuasion_result = None

    # Return structured results
    return {
        'conversation_id': group_sorted['conversation_id'].iloc[0],
        'da_sequence': da_sequence,
        'persuasion_result': persuasion_result,
        'utterance_count': len(group_sorted),
        'unique_persuasion_labels': list(persuasion_labels)
    }


def analyze_conversation_data(df):
    """
    Main function to analyze conversation data by grouping and processing

    Args:
        df (pd.DataFrame): Input dataframe with conversation data

    Returns:
        pd.DataFrame: Processed results with DA sequences and persuasion results
    """
    # Group data by conversation ID
    conversation_groups = df.groupby('conversation_id')

    # Initialize list to store results
    processed_results = []

    # Process each conversation group with progress tracking
    total_groups = len(conversation_groups)
    print(f"\n🔄 Processing {total_groups} conversation groups...")

    for idx, (conv_id, group) in enumerate(conversation_groups, 1):
        # Process individual conversation group
        result = process_conversation_group(group)
        processed_results.append(result)

        # Print progress every 50 groups
        if idx % 50 == 0:
            print(f"   Processed {idx}/{total_groups} groups ({idx / total_groups * 100:.1f}%)")

    # Convert results to DataFrame
    results_df = pd.DataFrame(processed_results)

    # Print summary statistics
    print(f"\n📊 Processing complete! Summary:")
    result_distribution = results_df['persuasion_result'].value_counts(dropna = False)
    for result_type, count in result_distribution.items():
        percentage = count / len(results_df) * 100
        if result_type is None:
            print(f"   • All Neutral (not recorded): {count} conversations ({percentage:.1f}%)")
        else:
            print(f"   • {result_type}: {count} conversations ({percentage:.1f}%)")

    return results_df


def save_results(results_df, output_path):
    """
    Save processed results to CSV file

    Args:
        results_df (pd.DataFrame): Processed conversation results
        output_path (str): Path to save the output CSV file
    """
    # Save to CSV with UTF-8 encoding
    results_df.to_csv(output_path, index = False, encoding = 'utf-8')
    print(f"\n💾 Results saved to: {output_path}")
    print(f"   Output columns: {results_df.columns.tolist()}")


def create_visualizations(results_df, output_image_path):
    """
    Create visualizations for conversation analysis results

    Args:
        results_df (pd.DataFrame): Processed conversation results
        output_image_path (str): Path to save the visualization image
    """

    # Convert string representation of lists back to actual lists
    def str_to_list(str_repr):
        try:
            if pd.isna(str_repr):
                return []
            clean_str = str_repr.strip('[]').strip()
            if not clean_str:
                return []
            elements = [elem.strip().strip("'\"") for elem in clean_str.split(', ')]
            return elements
        except:
            return []

    # Process DA sequences for visualization
    if 'da_sequence' in results_df.columns:
        results_df['da_sequence_list'] = results_df['da_sequence'].apply(str_to_list)

        # Collect all DA labels
        all_da_labels = []
        for da_list in results_df['da_sequence_list']:
            all_da_labels.extend(da_list)

        # Get top 15 most common DA labels
        da_counter = Counter(all_da_labels)
        top_15_da = da_counter.most_common(15)
        da_labels_top15 = [item[0] for item in top_15_da]
        da_counts_top15 = [item[1] for item in top_15_da]

        # Create visualization figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (16, 12))
        fig.suptitle('Conversation Data Analysis Report', fontsize = 20, fontweight = 'bold', y = 0.98)

        # Color scheme
        colors_persuasion = ['#8dd3c7', '#fb8072', '#bebada', '#80b1d3']
        colors_da = plt.cm.Set3(np.linspace(0, 1, len(da_labels_top15)))

        # Subplot 1: Persuasion result distribution (pie chart)
        persuasion_counts = results_df['persuasion_result'].value_counts(dropna = False)
        persuasion_labels = ['All Neutral (not recorded)' if x is None else x for x in persuasion_counts.index]
        persuasion_values = persuasion_counts.values

        ax1.pie(persuasion_values, labels = persuasion_labels, autopct = '%1.1f%%',
                colors = colors_persuasion, startangle = 90, textprops = {'fontsize': 10})
        ax1.set_title('Persuasion Result Distribution', fontsize = 14, fontweight = 'bold', pad = 20)

        # Subplot 2: Persuasion result counts (bar chart)
        bars = ax2.bar(persuasion_labels, persuasion_values, color = colors_persuasion,
                       alpha = 0.8, edgecolor = 'black', linewidth = 0.5)
        ax2.set_title('Persuasion Result Count', fontsize = 14, fontweight = 'bold', pad = 20)
        ax2.set_ylabel('Number of Conversations', fontsize = 12)
        ax2.tick_params(axis = 'x', rotation = 45)

        # Add value labels to bars
        for bar, value in zip(bars, persuasion_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 2,
                     f'{int(value)}', ha = 'center', va = 'bottom', fontsize = 11, fontweight = 'bold')

        # Subplot 3: Top 15 DA labels (horizontal bar chart)
        bars3 = ax3.barh(range(len(da_labels_top15)), da_counts_top15,
                         color = colors_da, alpha = 0.8, edgecolor = 'black', linewidth = 0.5)
        ax3.set_yticks(range(len(da_labels_top15)))
        ax3.set_yticklabels(da_labels_top15, fontsize = 10)
        ax3.set_title('Top 15 Most Common Dialogue Act (DA) Labels', fontsize = 14, fontweight = 'bold', pad = 20)
        ax3.set_xlabel('Frequency', fontsize = 12)

        # Add value labels to horizontal bars
        for i, (bar, count) in enumerate(zip(bars3, da_counts_top15)):
            width = bar.get_width()
            ax3.text(width + 2, bar.get_y() + bar.get_height() / 2.,
                     f'{count}', ha = 'left', va = 'center', fontsize = 9, fontweight = 'bold')

        # Subplot 4: Average utterances per persuasion result
        non_none_df = results_df[results_df['persuasion_result'].notna()].copy()
        avg_utterances = non_none_df.groupby('persuasion_result')['utterance_count'].mean().sort_values(
            ascending = False)

        bars4 = ax4.bar(avg_utterances.index, avg_utterances.values,
                        color = colors_persuasion[1:], alpha = 0.8, edgecolor = 'black', linewidth = 0.5)
        ax4.set_title('Average Utterances per Persuasion Result', fontsize = 14, fontweight = 'bold', pad = 20)
        ax4.set_ylabel('Average Number of Utterances', fontsize = 12)
        ax4.set_xlabel('Persuasion Result Type', fontsize = 12)

        # Add value labels to bars
        for bar, value in zip(bars4, avg_utterances.values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'{value:.2f}', ha = 'center', va = 'bottom', fontsize = 11, fontweight = 'bold')

        # Save visualization
        plt.tight_layout()
        plt.savefig(output_image_path, dpi = 300, bbox_inches = 'tight', facecolor = 'white')
        plt.close()

        print(f"\n📸 Visualization saved to: {output_image_path}")


# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    # Configuration
    INPUT_FILE = 'agreement_with_our_predictions.csv'  # Update with your input path
    OUTPUT_CSV = './output_data/conversation_analysis_results.csv'  # Output results path
    OUTPUT_IMAGE = './output_data/conversation_visualization.png'  # Output visualization path

    try:
        # Step 1: Load and preprocess data
        df = load_and_preprocess_data(INPUT_FILE)

        # Step 2: Analyze conversation data
        results_df = analyze_conversation_data(df)

        # Step 3: Save processed results
        save_results(results_df, OUTPUT_CSV)

        # Step 4: Create visualizations (optional)
        create_visualizations(results_df, OUTPUT_IMAGE)

        print("\nAll processing completed successfully!")

        # Show sample results
        print("\nSample Results (First 3 Conversations):")
        sample_df = results_df.head(3)
        for idx, row in sample_df.iterrows():
            print(f"\nConversation ID: {row['conversation_id']}")
            print(f"  DA Sequence: {row['da_sequence']}")
            print(f"  Persuasion Result: {row['persuasion_result']}")
            print(f"  Number of Utterances: {row['utterance_count']}")

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise