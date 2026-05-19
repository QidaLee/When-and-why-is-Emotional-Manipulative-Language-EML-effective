import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Read the TXT file
data = []
with open('full_set_merged.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:  # skip empty lines
            parts = line.split('|')
            if len(parts) >= 3:  # ensure enough columns
                da_label = parts[2]  # last column is DA label
                data.append(da_label)

# Process DA labels (comma-separated)
all_da_labels = []
for da_str in data:
    # Split comma-separated labels, strip whitespace
    labels = [label.strip() for label in da_str.split(',')]
    all_da_labels.extend(labels)

# Calculate distribution
da_counter = Counter(all_da_labels)
total_count = len(all_da_labels)

# Print statistics
print("="*60)
print("DA Label Distribution Statistics")
print("="*60)
print(f"\nTotal label occurrences: {total_count}")
print(f"Number of unique labels: {len(da_counter)}")
print("\nLabel Distribution:")
print("-"*40)
print(f"{'DA Label':<15} {'Frequency':<10} {'Percentage':<10}")
print("-"*40)

for label, count in sorted(da_counter.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / total_count) * 100
    print(f"{label:<15} {count:<10} {percentage:>6.2f}%")

# Calculate label count per row distribution
labels_per_row = [len(da_str.split(',')) for da_str in data]
print("\n" + "="*60)
print("Labels Per Row Statistics")
print("="*60)
row_counter = Counter(labels_per_row)
for num_labels, count in sorted(row_counter.items()):
    percentage = (count / len(data)) * 100
    print(f"Rows with {num_labels} label(s): {count:6d} ({percentage:5.2f}%)")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart
labels = list(da_counter.keys())
counts = list(da_counter.values())
colors = plt.cm.Set3(range(len(labels)))

bars = ax1.bar(range(len(labels)), counts, color=colors)
ax1.set_xlabel('DA Labels', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('DA Label Distribution (Bar Chart)', fontsize=14)
ax1.set_xticks(range(len(labels)))
ax1.set_xticklabels(labels, rotation=45, ha='right')

# Add values on top of bars
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{count}', ha='center', va='bottom')

# Pie chart
wedges, texts, autotexts = ax2.pie(counts, labels=labels, autopct='%1.1f%%',
                                     colors=colors, startangle=90)
ax2.set_title('DA Label Distribution (Pie Chart)', fontsize=14)

plt.tight_layout()
plt.savefig('DA_label_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Save detailed statistics to CSV
stats_df = pd.DataFrame({
    'DA_Label': list(da_counter.keys()),
    'Count': list(da_counter.values()),
    'Percentage': [f"{(c/total_count)*100:.2f}%" for c in da_counter.values()]
})
stats_df = stats_df.sort_values('Count', ascending=False)
stats_df.to_csv('DA_label_statistics.csv', index=False)

print("\n✓ Statistics saved to 'DA_label_statistics.csv'")
print("✓ Chart saved to 'DA_label_distribution.png'")

# Show first 10 rows as example
print("\n" + "="*60)
print("Data Example (first 10 rows):")
print("="*60)
for i in range(min(10, len(data))):
    print(f"Row {i+1}: {data[i]}")