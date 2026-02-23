#!/usr/bin/env python3
"""
Multi-class classifier for PPA subtypes using S_sem.

Classes: control (0), svPPA (1), lvPPA (2), nfvPPA (3)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("outputs/ppa_dissociation_all.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/multiclass_results.json"))
    args = parser.parse_args()
    
    df = pd.read_csv(args.features)
    
    # Create numeric labels
    subtype_map = {"control": 0, "svppa": 1, "lvppa": 2, "nfvppa": 3}
    df["label_mc"] = df["subtype"].map(subtype_map)
    df = df[df["label_mc"].notna() & df["s_sem"].notna()]
    
    print(f"\nDataset: {len(df)} samples")
    print(f"By subtype: {df['subtype'].value_counts().to_dict()}")
    
    # Features
    X = df[["s_sem"]].values
    y = df["label_mc"].astype(int).values
    
    # Handle NaN
    X = np.nan_to_num(X, nan=0.0)
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Classifier
    clf = LogisticRegression(
        multi_class="multinomial",
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    
    # CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(clf, X_scaled, y, cv=cv)
    
    # Metrics
    acc = accuracy_score(y, y_pred_cv)
    f1_macro = f1_score(y, y_pred_cv, average="macro")
    f1_weighted = f1_score(y, y_pred_cv, average="weighted")
    
    print("\n" + "=" * 60)
    print("MULTI-CLASS CLASSIFICATION: S_sem Only")
    print("=" * 60)
    print(f"\nOverall Accuracy: {acc:.3f}")
    print(f"Macro F1: {f1_macro:.3f}")
    print(f"Weighted F1: {f1_weighted:.3f}")
    
    # Per-class report
    target_names = ["control", "svPPA", "lvPPA", "nfvPPA"]
    print("\nClassification Report:")
    print(classification_report(y, y_pred_cv, target_names=target_names))
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred_cv)
    print("Confusion Matrix:")
    print(f"{'':>10} " + " ".join(f"{n:>8}" for n in target_names))
    for i, row in enumerate(cm):
        print(f"{target_names[i]:>10} " + " ".join(f"{v:>8}" for v in row))
    
    # Binary: control vs all PPA
    print("\n" + "-" * 60)
    print("BINARY: Control vs All PPA")
    print("-" * 60)
    y_binary = (df["label_mc"] > 0).astype(int).values
    y_pred_binary = (y_pred_cv > 0).astype(int)
    acc_binary = accuracy_score(y_binary, y_pred_binary)
    f1_binary = f1_score(y_binary, y_pred_binary)
    print(f"Accuracy: {acc_binary:.3f}")
    print(f"F1: {f1_binary:.3f}")
    
    # Binary: svPPA vs other PPA
    print("\n" + "-" * 60)
    print("BINARY: svPPA vs Other PPA (lvPPA + nfvPPA)")
    print("-" * 60)
    df_ppa = df[df["label_mc"] > 0]
    y_svppa = (df_ppa["label_mc"] == 1).astype(int).values
    y_pred_svppa = (y_pred_cv[df["label_mc"] > 0] == 1).astype(int)
    if len(y_svppa) > 0:
        acc_svppa = accuracy_score(y_svppa, y_pred_svppa)
        f1_svppa = f1_score(y_svppa, y_pred_svppa)
        print(f"Accuracy: {acc_svppa:.3f}")
        print(f"F1: {f1_svppa:.3f}")
    
    # Save results
    results = {
        "n_samples": len(df),
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": cm.tolist(),
        "binary_control_vs_ppa": {"accuracy": float(acc_binary), "f1": float(f1_binary)},
    }
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
