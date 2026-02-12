import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import os
from datetime import datetime
import sys


# ==================================================
# Setup Logging
# ==================================================
def setup_logging():
    """Setup logging configuration"""
    # Create log directory if not exists
    log_dir = './log'
    os.makedirs(log_dir, exist_ok = True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'prediction_{timestamp}.log')

    # Configure logging
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(levelname)s - %(message)s',
        handlers = [
            logging.FileHandler(log_file, encoding = 'utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


# Define device as global variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict_texts(texts, model, tokenizer, id2label, logger, batch_size=16):
    """
    Batch prediction for texts

    Args:
        texts: pandas Series or list of texts
        model: pretrained model
        tokenizer: corresponding tokenizer
        id2label: mapping from ID to original label string
        logger: logger instance
        batch_size: batch size for prediction
    """
    predictions = []
    probabilities = []

    logger.info(f"Starting batch prediction with batch size: {batch_size}")
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch_num = i // batch_size + 1
        batch_texts = texts[i:i + batch_size].tolist() if hasattr(texts, 'tolist') else texts[i:i + batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding = True,
            truncation = True,
            max_length = 128,
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

        if batch_num % 10 == 0:
            logger.info(f"Processed batch {batch_num}/{total_batches}")

    logger.info(f"Batch prediction completed. Total samples: {len(predictions)}")
    return predictions, probabilities


def save_results_to_log(df, logger, log_dir='./log'):
    """Save detailed results to separate log file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save detailed results
    details_file = os.path.join(log_dir, f'prediction_details_{timestamp}.csv')
    df.to_csv(details_file, index = False)
    logger.info(f"Detailed predictions saved to: {details_file}")

    # Save summary statistics
    summary_file = os.path.join(log_dir, f'prediction_summary_{timestamp}.txt')
    with open(summary_file, 'w', encoding = 'utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("MODEL PREDICTION SUMMARY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        f.write("Model Information:\n")
        f.write(f"  Model path: ./persuasion_model\n")
        f.write(f"  Device: {device}\n")
        f.write(f"  Number of classes: {len(df['prediction_label'].unique())}\n\n")

        f.write("Prediction Statistics:\n")
        f.write(f"  Total samples: {len(df)}\n")
        f.write(f"  Average confidence: {df['confidence'].mean():.4f}\n")
        f.write(f"  Confidence std: {df['confidence'].std():.4f}\n")
        f.write(f"  Confidence min: {df['confidence'].min():.4f}\n")
        f.write(f"  Confidence max: {df['confidence'].max():.4f}\n\n")

        f.write("Predicted Label Distribution:\n")
        label_dist = df['prediction_label'].value_counts()
        for label, count in label_dist.items():
            percentage = count / len(df) * 100
            f.write(f"  {label}: {count} ({percentage:.2f}%)\n")

        f.write("\nAverage Confidence per Label:\n")
        conf_by_label = df.groupby('prediction_label')['confidence'].mean()
        for label, conf in conf_by_label.items():
            f.write(f"  {label}: {conf:.4f}\n")

        f.write("\n" + "=" * 60 + "\n")

    logger.info(f"Summary statistics saved to: {summary_file}")


def main():
    # Setup logging
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("Starting Prediction Pipeline")
    logger.info("=" * 60)

    try:
        # Load data
        data_path = './our_data/agreement_annotations_with_labels.csv'
        logger.info(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} data samples")
        logger.info(f"Columns: {df.columns.tolist()}")

        # Load trained model
        model_path = './persuasion_model'
        logger.info(f"Loading model from: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        # Get label mappings from the model config
        id2label = model.config.id2label
        label2id = model.config.label2id

        logger.info(f"Model label mapping: {id2label}")
        logger.info(f"Number of classes: {len(id2label)}")
        logger.info(f"Using device: {device}")

        model.to(device)
        model.eval()

        # Check CUDA memory if using GPU
        if device.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")

        # Predict
        logger.info("Starting batch prediction...")
        texts = df['text'].fillna('')
        predictions, probabilities = predict_texts(texts, model, tokenizer, id2label, logger)

        # Map predictions to original labels using model's id2label
        df['prediction_label'] = [id2label.get(p, f'UNKNOWN_{p}') for p in predictions]

        # Add confidence scores
        df['confidence'] = [max(probs) for probs in probabilities]

        # Add prediction timestamp
        # df['prediction_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Save results
        output_path = './our_data/agreement_with_our_predictions.csv'
        df.to_csv(output_path, index = False)
        logger.info(f"Predictions saved to: {output_path}")

        # Output statistics
        logger.info("\n" + "=" * 60)
        logger.info("PREDICTION RESULTS SUMMARY")
        logger.info("=" * 60)

        # Predicted label distribution
        label_dist = df['prediction_label'].value_counts()
        logger.info("\nPredicted label distribution:")
        for label, count in label_dist.items():
            percentage = count / len(df) * 100
            logger.info(f"  {label}: {count} ({percentage:.1f}%)")

        # Average confidence per label
        conf_by_label = df.groupby('prediction_label')['confidence'].mean()
        logger.info("\nAverage confidence per predicted label:")
        for label, conf in conf_by_label.items():
            logger.info(f"  {label}: {conf:.3f}")

        # Confidence statistics
        logger.info("\nConfidence Statistics:")
        logger.info(f"  Overall average: {df['confidence'].mean():.3f}")
        logger.info(f"  Overall median: {df['confidence'].median():.3f}")
        logger.info(f"  Overall std: {df['confidence'].std():.3f}")
        logger.info(f"  Min: {df['confidence'].min():.3f}")
        logger.info(f"  Max: {df['confidence'].max():.3f}")

        # Show sample predictions
        logger.info("\nSample predictions (first 5 rows):")
        sample_cols = ['speaker', 'text', 'prediction_label', 'confidence']

        for idx, row in df[sample_cols].head(5).iterrows():
            logger.info(f"\n  Row {idx}:")
            for col in sample_cols:
                if col == 'text' and len(str(row[col])) > 50:
                    logger.info(f"    {col}: {str(row[col])[:50]}...")
                else:
                    logger.info(f"    {col}: {row[col]}")

        # Save detailed results to log directory
        save_results_to_log(df, logger)

        logger.info("\n" + "=" * 60)
        logger.info("Prediction pipeline completed successfully!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info = True)
        raise


if __name__ == "__main__":
    main()