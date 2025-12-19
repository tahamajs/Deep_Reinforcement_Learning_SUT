"""Compute common metrics (MSE, RMSE, R^2, accuracy) and save metrics.json.

Usage:
    python scripts/compute_metrics.py --pred predictions.npz --out outputs/example_run

The predictions file is expected to be a numpy .npz with arrays:
- y_true (N,)
- y_pred (N,) or y_logits (N,C) for classification
- y_pred_labels (N,) optional for classification

This script writes `metrics.json` in the output folder and saves `confusion_matrix.png` in `out/pictures` for classification runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {"mse": mse, "rmse": rmse, "r2": r2}


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {"accuracy": acc, "precision": float(precision), "recall": float(recall), "f1_macro": float(f1), "confusion_matrix": cm.tolist()}


def save_confusion_matrix(cm, out_path: Path):
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion matrix')
    plt.tight_layout()
    out_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path / 'confusion_matrix.png', dpi=200)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", type=str, required=True, help=".npz file with predictions and targets")
    p.add_argument("--out", type=str, required=True, help="output folder to write metrics.json and pictures/")
    args = p.parse_args()

    pred_path = Path(args.pred)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(pred_path)
    y_true = data['y_true']

    metrics = {}
    # regression
    if 'y_pred' in data:
        y_pred = data['y_pred']
        metrics.update({"val_mse": None})
        m = compute_regression_metrics(y_true, y_pred)
        metrics.update({"val_mse": m['mse'], "val_rmse": m['rmse'], "val_r2": m['r2']})

    # classification
    if 'y_pred_labels' in data:
        y_pred_labels = data['y_pred_labels']
        cm = compute_classification_metrics(y_true, y_pred_labels)
        metrics.update({"accuracy": cm['accuracy'], "precision_macro": cm['precision'], "recall_macro": cm['recall'], "f1_macro": cm['f1_macro']})
        # save confusion matrix
        save_confusion_matrix(np.array(cm['confusion_matrix']), out_dir / 'pictures')

    # write metrics.json
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote metrics.json to {out_dir / 'metrics.json'}")


if __name__ == '__main__':
    main()
