import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import re
from collections import Counter

# 1. Load the Excel file (update path to your local file)
df = pd.read_excel("output_data/all_turns_data_with_EML_DA_PR.xlsx")


# 2. Text preprocessing function (cleans English text)
def preprocess_text(text):
    """Clean raw sentences: handle NaN, remove punctuation/stopwords, lowercase"""
    # Step 1: Handle missing values or non-string inputs
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # Step 2: Remove special characters, punctuation, and lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())

    # Step 3: Remove stopwords (common words with no meaningful context)
    stopwords = set(STOPWORDS)
    # Add custom stopwords (adjust based on your data, e.g., domain-specific terms)
    custom_stopwords = {"i", "you", "me", "my", "your", "we", "us", "it", "is", "are", "was", "were", "that", "this"}
    all_stopwords = stopwords.union(custom_stopwords)

    # Step 4: Keep only words longer than 2 characters (filter short noise)
    words = [word.strip() for word in text.split() if word.strip() not in all_stopwords and len(word.strip()) > 2]

    return " ".join(words)


# 3. Safe filtering function for PR_label (handles non-string values)
def safe_filter_pr(df, target_pr):
    """Filter df to get sentences for a target PR label (handles non-string PR_label values)"""
    # Convert PR_label to string (force non-strings to "NaN" string for safe comparison)
    pr_label_str = df["PR_label"].astype(str).str.strip().str.lower()
    # Filter rows where PR_label matches the target (e.g., "compliance")
    filtered_df = df[pr_label_str == target_pr.lower()]
    # Preprocess and concatenate sentences
    return filtered_df["Sentence"].apply(preprocess_text).str.cat(sep = " ")


# 4. Extract text for each target category
## a. EML = 1 (filter numeric EML_label safely)
eml_text = df[df["EML_label"] == 1]["Sentence"].apply(preprocess_text).str.cat(sep = " ")

## b. PR = Compliance (use safe_filter_pr to handle non-string PR_label)
pr_compliance_text = safe_filter_pr(df, target_pr = "Compliance")

## c. PR = Resistance (same safe filtering)
pr_resistance_text = safe_filter_pr(df, target_pr = "Resistance")

# 5. Check text availability (avoid empty word clouds)
print("=== Text Availability Check ===")
print(f"EML=1 text: {len(eml_text.split())} valid words")
print(f"PR=Compliance text: {len(pr_compliance_text.split())} valid words")
print(f"PR=Resistance text: {len(pr_resistance_text.split())} valid words")


# 6. Reusable word cloud generator
def generate_wordcloud(text, title, save_path, color_scheme="viridis"):
    """Generate high-quality word cloud; skip if text is too short"""
    if len(text.split()) < 5:  # Skip if <5 valid words (avoids empty plots)
        print(f"⚠️ Skipping '{title}': Not enough valid text (needs ≥5 words)")
        return

    # Configure word cloud (optimized for academic use)
    wordcloud = WordCloud(
        width = 800,
        height = 600,
        background_color = "white",
        max_words = 150,  # Limit to top 150 frequent words
        contour_width = 1,
        contour_color = "#333333",  # Subtle gray border for clarity
        colormap = color_scheme,
        font_path = None  # Uses system default font (works for English)
    ).generate(text)

    # Plot and save
    plt.figure(figsize = (10, 7))
    plt.imshow(wordcloud, interpolation = "bilinear")  # Smooth image rendering
    plt.title(title, fontsize = 16, pad = 20, fontweight = "bold")
    plt.axis("off")  # Hide axes for clean visualization
    plt.tight_layout()  # Prevent title cutoff
    plt.savefig(save_path, dpi = 300, bbox_inches = "tight")  # 300 DPI = print-quality
    plt.close()  # Free memory
    print(f"✅ Saved: {save_path}")


# 7. Generate and save all 3 word clouds
## a. EML=1 (plasma color scheme)
generate_wordcloud(
    text = eml_text,
    title = "Word Cloud: Sentences with EML = 1",
    save_path = "D:/Master_study/master_thesis_programs/my_project/eml_1_wordcloud.png",
    color_scheme = "plasma"
)

## b. PR=Compliance (Blues color scheme)
generate_wordcloud(
    text = pr_compliance_text,
    title = "Word Cloud: Sentences with PR = Compliance",
    save_path = "D:/Master_study/master_thesis_programs/my_project/pr_compliance_wordcloud.png",
    color_scheme = "Blues"
)

## c. PR=Resistance (Reds color scheme)
generate_wordcloud(
    text = pr_resistance_text,
    title = "Word Cloud: Sentences with PR = Resistance",
    save_path = "D:/Master_study/master_thesis_programs/my_project/pr_resistance_wordcloud.png",
    color_scheme = "Reds"
)


# 8. Optional: Print top 10 frequent words (for quick analysis)
def get_top_words(text, top_n=10):
    words = text.split()
    return Counter(words).most_common(top_n)


print("\n=== Top 10 Frequent Words ===")
print(f"EML=1: {get_top_words(eml_text)}")
print(f"PR=Compliance: {get_top_words(pr_compliance_text)}")
print(f"PR=Resistance: {get_top_words(pr_resistance_text)}")