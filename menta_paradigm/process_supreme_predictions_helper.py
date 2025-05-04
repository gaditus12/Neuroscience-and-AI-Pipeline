import os
import json
import pandas as pd
import numpy as np
from scipy.stats import entropy
from collections import defaultdict


def process_predictions(pred_path):
    with open(pred_path, "r") as f:
        data = json.load(f)

    y_true = data["y_true"]
    y_pred = data["y_pred"]
    y_proba = np.array(data["y_proba"])
    classes = data["classes"]

    confidences = y_proba.max(axis=1)
    predicted_indices = y_proba.argmax(axis=1)
    predicted_labels = [classes[i] for i in predicted_indices]
    entropies = [entropy(prob) for prob in y_proba]
    correctness = [yt == yp for yt, yp in zip(y_true, y_pred)]

    return pd.DataFrame(
        {
            "true_label": y_true,
            "predicted_label": y_pred,
            "confidence": confidences,
            "entropy": entropies,
            "correct": correctness,
        }
    )


def aggregate_by_channel(predictions_dir, run_id="unnamed_run"):
    output_dir = os.path.join("confidence_analysis", run_id)
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(predictions_dir) if f.endswith(".json")]
    channel_dfs = defaultdict(list)

    for file in files:
        file_path = os.path.join(predictions_dir, file)

        # Expect filename like 'o2_comBat_5_capture_preds.json'
        channel = file.split("_")[0]  # crude, adjust if needed
        df = process_predictions(file_path)
        df["session"] = file  # optional for tracking
        channel_dfs[channel].append(df)

    # Save one CSV per channel
    for channel, dfs in channel_dfs.items():
        combined = pd.concat(dfs, ignore_index=True)
        output_path = os.path.join(output_dir, f"{channel}_analysis.csv")
        combined.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    # Example usage
    run_id = "entropy_alpha(0.6)_o2_comBat_rf_tp8_comBat_rf_oz_comBat_knn_loso_1_run_1746285265"  # customize or extract from path
    predictions_dir = f"supreme_model_outputs/{run_id}/fold_predictions"
    aggregate_by_channel(predictions_dir, run_id)
