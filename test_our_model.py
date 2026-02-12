import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define device as global variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict_texts(texts, model, tokenizer, id2label, batch_size=16):
    """
    Batch prediction for texts

    Args:
        texts: pandas Series or list of texts
        model: pretrained model
        tokenizer: corresponding tokenizer
        id2label: mapping from ID to original label string
        batch_size: batch size for prediction
    """
    predictions = []
    probabilities = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size].tolist()

        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding = True,
            truncation = True,
            max_length = 128,  # Match training config
            return_tensors = 'pt'
        ).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim = -1)
            preds = torch.argmax(logits, dim = -1)

            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())

    return predictions, probabilities


def main():
    # Load data
    df = pd.read_csv('./our_data/agreement_annotations_with_labels.csv')
    print(f"Loaded {len(df)} data samples")

    # Load trained model
    model_path = './persuasion_model'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Get label mappings from the model config
    id2label = model.config.id2label  # This is loaded from saved model
    label2id = model.config.label2id

    print(f"\nModel label mapping: {id2label}")
    print(f"Number of classes: {len(id2label)}")

    model.to(device)
    model.eval()

    # Predict
    texts = df['text'].fillna('')  # Handle NaN values
    predictions, probabilities = predict_texts(texts, model, tokenizer, id2label)

    # Map predictions to original labels using model's id2label
    df['prediction_id'] = predictions
    df['prediction_label'] = [id2label.get(p, f'UNKNOWN_{p}') for p in predictions]

    # Add confidence scores
    df['confidence'] = [max(probs) for probs in probabilities]

    # Compare with existing annotations if available
    # if 'labels_from_GPT' in df.columns:
    #     df['same_as_gpt'] = df['labels_from_GPT'] == df['prediction_label']
    #     gpt_agreement = df['same_as_gpt'].mean()
    #     print(f"\nAgreement with GPT: {gpt_agreement:.2%}")

    if 'lables_from_our_CaBert' in df.columns:
        df['same_as_cabert'] = df['lables_from_our_CaBert'] == df['prediction_label']
        cabert_agreement = df['same_as_cabert'].mean()
        print(f"Agreement with CaBERT: {cabert_agreement:.2%}")

    # Save results
    df.to_csv('./our_data/agreement_with_our_predictions.csv', index = False)
    print("\nPrediction completed, results saved")

    # Output statistics
    print("\n=== Prediction Results Summary ===")
    print("\nPredicted label distribution:")
    label_dist = df['prediction_label'].value_counts()
    for label, count in label_dist.items():
        percentage = count / len(df) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")

    # Average confidence per label
    print("\nAverage confidence per predicted label:")
    conf_by_label = df.groupby('prediction_label')['confidence'].mean()
    for label, conf in conf_by_label.items():
        print(f"  {label}: {conf:.3f}")

    # Show sample
    print("\nSample predictions (first 10 rows):")
    sample_cols = ['speaker', 'text', 'prediction_label', 'confidence']
    if 'labels_from_GPT' in df.columns:
        sample_cols.append('labels_from_GPT')
    if 'lables_from_our_CaBert' in df.columns:
        sample_cols.append('lables_from_our_CaBert')

    print(df[sample_cols].head(10).to_string())


if __name__ == "__main__":
    main()