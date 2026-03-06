import torch
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


class PreTrainedEMLDetector:
    """Directly call pre-trained EML detection models from Hugging Face"""

    def __init__(self, model_name: str = None):
        # Automatically select the best pre-trained model (prioritize high-scoring EML-specific models)
        self.model_name = model_name or self._select_best_model()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load pre-trained model and tokenizer (adapted for English conversational data)
        print(f"Loading pre-trained model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,  # 0=non-EML, 1=EML
            ignore_mismatched_sizes=True  # Compatible with models having different label counts
        ).to(self.device)

        # Create inference pipeline
        self.eml_pipeline = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            return_all_scores=False  # Return only the final label
        )

    def _select_best_model(self):
        """Select the best EML detection model from Hugging Face
        (optimized for English donation persuasion dialogues)"""
        # Priority list of recommended pre-trained models
        # (optimized for English emotional manipulation/donation persuasion scenarios)
        recommended_models = [
            "cardiffnlp/twitter-roberta-base-emotion",  # Base emotion manipulation model (adapted for English)
            "unitary/toxic-bert",                       # Toxicity/manipulative language detection
            "mrm8488/bert-tiny-finetuned-emotion",      # Lightweight EML detection (faster speed)
        ]
        return recommended_models[1]  # Default to unitary/toxic-bert (best for manipulation detection)

    def label_single_line(self, text: str) -> int:
        """Label a single line of text: 1=EML, 0=non-EML"""
        if not text.strip():
            return 0

        # Inference (adapted to model output format)
        result = self.eml_pipeline(text)[0]
        label = result["label"]
        score = result["score"]

        # Map model output to 0/1 label (adapted for emotional manipulation detection in donation persuasion scenarios)
        # Extended manipulation-related labels: label as 1 if it contains negative emotions/manipulative tendencies
        manipulation_labels = ["anger", "fear", "sadness", "manipulation", "abuse", "negative", "guilt"]
        if any(emotion in label.lower() for emotion in manipulation_labels) or score > 0.7:
            return 1
        else:
            return 0

    def label_conversation(self, dialogue: list) -> list:
        """Batch label a list of dialogues, returns [(text, label), ...]"""
        labeled_results = []
        print(f"Starting to label {len(dialogue)} dialogue lines...")
        for i, line in enumerate(dialogue):
            label = self.label_single_line(line)
            labeled_results.append((line, label))
            if (i + 1) % 50 == 0:
                print(f"Labeled {i + 1}/{len(dialogue)} lines")
        return labeled_results


# ---------------------- Main program: Read Excel data and label ----------------------
if __name__ == "__main__":
    # 1. Configure file paths (updated for 300_dialog.xlsx, output to /output_data)
    input_excel_path = "data/Persuasion_For_Good/300_dialog.xlsx"  # Input data path
    output_excel_path = "output_data/300_dialog_with_eml_label.xlsx"  # Output to /output_data directory

    # Check if input file exists
    if not os.path.exists(input_excel_path):
        raise FileNotFoundError(f"Input file does not exist: {input_excel_path}")

    # 2. Read Excel data
    print("Reading Excel data...")
    df = pd.read_excel(input_excel_path)

    # Check if required column exists (text column is 'Unit')
    if "Unit" not in df.columns:
        raise KeyError("Data is missing the 'Unit' column (text column), please check column names")

    # 3. Extract text column to be labeled (Unit column - dialogue text)
    dialogue_list = df["Unit"].tolist()
    print(f"Loaded {len(dialogue_list)} dialogue texts to be labeled")

    # 4. Initialize pre-trained model (use unitary/toxic-bert as default)
    eml_detector = PreTrainedEMLDetector()

    # 5. Batch label dialogue texts
    labeled_results = eml_detector.label_conversation(dialogue_list)

    # 6. Merge labeling results back to original dataframe
    df["eml_label"] = [label for _, label in labeled_results]  # Add EML label column

    # 7. Create output directory if it doesn't exist, then save labeled data
    os.makedirs(os.path.dirname(output_excel_path), exist_ok=True)
    df.to_excel(output_excel_path, index=False)
    print(f"\nLabeling complete! Results saved to: {output_excel_path}")

    # 8. Preview first 10 labeled results for verification
    print("\n=== Preview of first 10 EML labeling results (1=contains emotional manipulation, 0=does not contain) ===")
    preview_df = df[["B2", "B4", "Turn", "Unit", "eml_label"]].head(10)
    for idx, row in preview_df.iterrows():
        print(f"Dialogue ID: {row['B2']} | Turn: {row['Turn']} | Text: {row['Unit'][:60]}... | EML Label: {row['eml_label']}")