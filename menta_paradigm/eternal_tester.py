import joblib
import pandas as pd
import argparse
import os
from datetime import datetime
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)
import matplotlib.pyplot as plt
import seaborn as sns


def save_metrics_report(y_true, y_pred, output_dir, model_name, test_data_path):
    """Save classification metrics and confusion matrix visualization"""
    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
    }

    # Create classification report
    report = classification_report(y_true, y_pred, output_dict=True)

    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_file = os.path.join(output_dir, 'test_metrics.csv')
    metrics_df.to_csv(metrics_file, index=False)

    # Save full classification report
    report_df = pd.DataFrame(report).transpose()
    report_file = os.path.join(output_dir, 'classification_report.csv')
    report_df.to_csv(report_file, index=True)

    # Save confusion matrix visualization
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(set(y_true)),
                yticklabels=sorted(set(y_true)))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    cm_file = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_file, bbox_inches='tight')
    plt.close()

    # Create metadata file
    metadata = {
        'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_name': os.path.basename(model_name),
        'test_data': os.path.basename(test_data_path),
        'num_samples': len(y_true),
        'metrics_files': [metrics_file, report_file, cm_file]
    }

    metadata_file = os.path.join(output_dir, 'evaluation_metadata.txt')
    with open(metadata_file, 'w') as f:
        for k, v in metadata.items():
            f.write(f"{k}: {v}\n")

    print(f"Saved evaluation metrics to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Predict using a saved EEG model and calculate metrics')
    parser.add_argument('--model', type=str, default='ml_model_outputs/76%_random_forest_o1/best_randomforest_model.pkl',
                        help='Path to the saved model (e.g., best_randomforest_model.pkl)')
    parser.add_argument('--test_data', type=str, default='data/merged_features/2_sess_2+4_6_captures/2+4_6_captures_2_labels.csv',
                        help='Path to the test data CSV file')
    parser.add_argument('--output_dir', type=str, default='data/test_outputs/test_results',
                        help='Directory to save predictions and metrics')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the trained model
    model = joblib.load(args.model)

    # Load test data
    test_df = pd.read_csv(args.test_data)

    # Identify feature columns and labels
    metadata_columns = ['label', 'channel', 'session']
    feature_columns = [col for col in test_df.columns if col not in metadata_columns]

    if not feature_columns:
        raise ValueError("No feature columns found in the test data")

    X_test = test_df[feature_columns]

    # Generate predictions
    predictions = model.predict(X_test)

    # Create output DataFrame
    output_df = test_df.copy()
    output_df['prediction'] = predictions

    # Save predictions
    predictions_file = os.path.join(args.output_dir, 'predictions.csv')
    output_df.to_csv(predictions_file, index=False)
    print(f"Predictions saved to {predictions_file}")

    # Calculate metrics if labels are available
    if 'label' in test_df.columns:
        y_true = test_df['label']
        y_pred = predictions

        # Save metrics and visualizations
        save_metrics_report(y_true, y_pred, args.output_dir, args.model, args.test_data)
        print("\nClassification Metrics:")
        print(classification_report(y_true, y_pred))
    else:
        print("\nNo 'label' column found in test data - skipping metrics calculation")


if __name__ == '__main__':
    main()