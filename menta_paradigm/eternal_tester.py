#!/usr/bin/env python
# evaluate_many_models.py  — v2 (saves model hyper‑parameters)

import os, glob, json, argparse, joblib
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report,
                             confusion_matrix)
from sklearn import __version__ as sklearn_version


# ─────────────────────────── helpers ──────────────────────────────────
def _write_metrics_and_figs(y_true, y_pred, out_dir,
                            model_name, test_csv_path):
    """Store metrics, report, CM figure & metadata inside *out_dir*."""
    metrics = dict(
        accuracy           = accuracy_score (y_true, y_pred),
        f1_macro           = f1_score       (y_true, y_pred, average='macro'),
        precision_macro    = precision_score(y_true, y_pred, average='macro'),
        recall_macro       = recall_score   (y_true, y_pred, average='macro'),
        f1_weighted        = f1_score       (y_true, y_pred, average='weighted'),
        precision_weighted = precision_score(y_true, y_pred, average='weighted'),
        recall_weighted    = recall_score   (y_true, y_pred, average='weighted'),
    )
    pd.DataFrame([metrics]).to_csv(
        os.path.join(out_dir, "test_metrics.csv"), index=False)

    pd.DataFrame(classification_report(
        y_true, y_pred, output_dict=True)
    ).T.to_csv(os.path.join(out_dir, "classification_report.csv"))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(set(y_true)),
                yticklabels=sorted(set(y_true)))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    # minimal metadata
    with open(os.path.join(out_dir, "evaluation_metadata.txt"), "w") as fh:
        fh.write(f"evaluation_date : "
                 f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write(f"model_name      : {model_name}\n")
        fh.write(f"test_data       : {os.path.basename(test_csv_path)}\n")
        fh.write(f"num_samples     : {len(y_true)}\n")
        fh.write(f"sklearn_version : {sklearn_version}\n")


def _dump_model_params(model, out_dir):
    """Serialize *all* hyper‑parameters to JSON (Pipeline‑aware)."""
    # A Pipeline exposes .get_params() just like any estimator – deep=True
    params = model.get_params(deep=True)

    # Convert any non‑JSON‑serialisable values to strings
    def make_serialisable(obj):
        try:
            json.dumps(obj); return obj
        except TypeError:
            return str(obj)

    params_serialisable = {k: make_serialisable(v) for k, v in params.items()}

    with open(os.path.join(out_dir, "model_params.json"), "w") as fh:
        json.dump(params_serialisable, fh, indent=2, sort_keys=True)


def _evaluate_one(model_path, X, y, out_root, test_csv):
    """Load *model_path*, predict on *X*, write artefacts under *out_root*."""
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    stamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir    = os.path.join(out_root, f"{model_name}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"→ Evaluating {model_name}")
    model      = joblib.load(model_path)
    y_pred     = model.predict(X)

    # predictions
    pd.DataFrame(dict(true=y, pred=y_pred)).to_csv(
        os.path.join(out_dir, "predictions.csv"), index=False)

    # metrics, figures & params
    _write_metrics_and_figs(y, y_pred, out_dir, model_name, test_csv)
    _dump_model_params(model, out_dir)

    print(f"   artefacts in {out_dir}\n")


# ───────────────────────────── main ──────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate every best*_model.pkl found in a directory")
    parser.add_argument("--models_dir", required=False, default='ml_model_outputs/comBat_100BO_kfolds',
                        help="Directory that contains the pickled models")
    parser.add_argument("--test_data", required=False, default="data/normalized_merges/14_test_comBat/normalized_imagery.csv",
                        help="CSV with test‑set features + ground‑truth labels")
    parser.add_argument("--output_root", default="data/test_outputs/model_evaluations",
                        help="Root folder where evaluation artefacts are stored")
    args = parser.parse_args()

    # discover models
    pattern     = os.path.join(os.path.abspath(args.models_dir),
                               "**", "best*_model.pkl")
    model_files = sorted(glob.glob(pattern, recursive=True))
    if not model_files:
        raise FileNotFoundError(f"No 'best*_model.pkl' under {args.models_dir}")

    # load test‑set once
    test_df      = pd.read_csv(args.test_data)
    meta_cols    = ['label', 'channel', 'session']
    feature_cols = [c for c in test_df.columns if c not in meta_cols]
    if not feature_cols:
        raise ValueError("Could not detect feature columns in the test CSV")
    X_test, y_test = test_df[feature_cols], test_df['label']

    # evaluate all models
    os.makedirs(args.output_root, exist_ok=True)
    for mf in model_files:
        try:
            _evaluate_one(mf, X_test, y_test, args.output_root, args.test_data)
        except Exception as exc:
            print(f"!! {os.path.basename(mf)} failed: {exc}\n")
