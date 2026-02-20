"""Compare predictions CSV with ground truth (from features CSV) and report accuracy and other metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(
    predictions_path: str = "outputs/predictions.csv",
    features_path: str = "outputs/features.csv",
    label_column: str = "label",
) -> None:
    pred_df = pd.read_csv(predictions_path)
    feat_df = pd.read_csv(features_path)
    if "subject_id" not in pred_df.columns or "subject_id" not in feat_df.columns:
        raise ValueError("Both CSVs must have subject_id column")
    if label_column not in feat_df.columns:
        raise ValueError(f"Features CSV must have '{label_column}' column")
    if "predicted" not in pred_df.columns:
        raise ValueError("Predictions CSV must have 'predicted' column")

    merged = pred_df[["subject_id", "predicted"]].merge(
        feat_df[["subject_id", label_column]],
        on="subject_id",
        how="inner",
    )
    y_true = merged[label_column].astype(int)
    y_pred = merged["predicted"].astype(int)

    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("Comparison: predictions vs ground truth")
    print("=" * 50)
    print(f"Subjects compared: {len(merged)}")
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"F1 (binary): {f1:.4f}")
    print("Confusion matrix (rows=true, cols=predicted):")
    print("              predicted=0  predicted=1")
    print(f"  true=0      {cm[0,0]:>10}  {cm[0,1]:>10}")
    print(f"  true=1      {cm[1,0]:>10}  {cm[1,1]:>10}")
    print()
    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=["control (0)", "svPPA (1)"]))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Compare predictions to ground truth")
    p.add_argument("--predictions", default="outputs/predictions.csv", help="Predictions CSV")
    p.add_argument("--features", default="outputs/features.csv", help="Features CSV (has labels)")
    p.add_argument("--label", default="label", help="Ground truth column name")
    args = p.parse_args()
    main(args.predictions, args.features, args.label)
