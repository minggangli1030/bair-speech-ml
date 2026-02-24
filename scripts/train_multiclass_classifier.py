#!/usr/bin/env python3
"""
Multi-class classifier for PPA subtypes.

Uses composite S_sem (scene_graph + Layer 2 adequacy), S_ph, Δ.
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler


def train_multiclass(X: np.ndarray, y: np.ndarray, feature_names: list[str], cv_folds: int = 5):
    """Train multinomial logistic regression with CV."""
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        median = np.nanmedian(col)
        X[np.isnan(X[:, j]), j] = median if not np.isnan(median) else 0.0
    X = np.nan_to_num(X, nan=0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X_scaled, y, cv=cv)
    return y_pred, X_scaled, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("outputs/ppa_dissociation_all.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/multiclass_results.json"))
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()
    
    df = pd.read_csv(args.features)
    
    subtype_map = {"control": 0, "svppa": 1, "lvppa": 2, "nfvppa": 3}
    df["label_mc"] = df["subtype"].map(subtype_map)
    df = df[df["label_mc"].notna() & df["s_sem"].notna()]
    
    print(f"\nDataset: {len(df)} samples")
    print(f"By subtype: {df['subtype'].value_counts().to_dict()}")
    
    y = df["label_mc"].astype(int).values
    target_names = ["control", "svPPA", "lvPPA", "nfvPPA"]
    results = []
    preds = {}
    
    # Model 1: S_sem only (composite = scene_graph + Layer 2)
    X1 = df[["s_sem"]].values
    y_pred1, _, _ = train_multiclass(X1, y, ["s_sem"], args.cv_folds)
    preds["S_sem_only"] = (y_pred1, y)
    r1 = {
        "model": "S_sem_only",
        "n_samples": len(df),
        "accuracy": float(accuracy_score(y, y_pred1)),
        "precision_macro": float(precision_score(y, y_pred1, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, y_pred1, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, y_pred1, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, y_pred1, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, y_pred1).tolist(),
    }
    results.append(r1)
    print(f"\n1. S_sem only: Acc={r1['accuracy']:.3f} Prec={r1['precision_macro']:.3f} Rec={r1['recall_macro']:.3f} F1_macro={r1['f1_macro']:.3f}")
    
    # Model 2: S_ph + S_sem (subsample with S_ph)
    df_sph = df[df["s_ph"].notna()].dropna(subset=["s_ph", "s_sem"])
    if len(df_sph) >= 20 and df_sph["label_mc"].nunique() >= 2:
        y2 = df_sph["label_mc"].astype(int).values
        X2 = df_sph[["s_ph", "s_sem"]].values
        y_pred2, _, _ = train_multiclass(X2, y2, ["s_ph", "s_sem"], args.cv_folds)
        preds["S_ph_S_sem"] = (y_pred2, y2)
        r2 = {
            "model": "S_ph_S_sem",
            "n_samples": len(df_sph),
            "accuracy": float(accuracy_score(y2, y_pred2)),
            "precision_macro": float(precision_score(y2, y_pred2, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y2, y_pred2, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y2, y_pred2, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y2, y_pred2, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y2, y_pred2).tolist(),
        }
        results.append(r2)
        print(f"2. S_ph + S_sem: Acc={r2['accuracy']:.3f} Prec={r2['precision_macro']:.3f} Rec={r2['recall_macro']:.3f} F1_macro={r2['f1_macro']:.3f} (n={len(df_sph)})")
    
    # Model 3: S_ph + S_sem + Δ (full dissociation)
    df_delta = df[df["delta"].notna()].dropna(subset=["s_ph", "s_sem", "delta"])
    if len(df_delta) >= 20 and df_delta["label_mc"].nunique() >= 2:
        y3 = df_delta["label_mc"].astype(int).values
        X3 = df_delta[["s_ph", "s_sem", "delta"]].values
        y_pred3, _, _ = train_multiclass(X3, y3, ["s_ph", "s_sem", "delta"], args.cv_folds)
        preds["S_ph_S_sem_Delta"] = (y_pred3, y3)
        r3 = {
            "model": "S_ph_S_sem_Delta",
            "n_samples": len(df_delta),
            "accuracy": float(accuracy_score(y3, y_pred3)),
            "precision_macro": float(precision_score(y3, y_pred3, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y3, y_pred3, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y3, y_pred3, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y3, y_pred3, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y3, y_pred3).tolist(),
        }
        results.append(r3)
        print(f"3. S_ph + S_sem + Δ: Acc={r3['accuracy']:.3f} Prec={r3['precision_macro']:.3f} Rec={r3['recall_macro']:.3f} F1_macro={r3['f1_macro']:.3f} (n={len(df_delta)})")
    
    # Model 4: S_sem + components (s_sem_graph, s_sem_layer2_adequacy) if available
    comp_cols = [c for c in ["s_sem_graph", "s_sem_layer2_adequacy"] if c in df.columns]
    if comp_cols:
        feat4 = ["s_sem"] + comp_cols
        X4 = df[feat4].values
        y_pred4, _, _ = train_multiclass(X4, y, feat4, args.cv_folds)
        preds["S_sem_plus_components"] = (y_pred4, y)
        r4 = {
            "model": "S_sem_plus_components",
            "n_samples": len(df),
            "accuracy": float(accuracy_score(y, y_pred4)),
            "precision_macro": float(precision_score(y, y_pred4, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y, y_pred4, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y, y_pred4, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y, y_pred4, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y, y_pred4).tolist(),
        }
        results.append(r4)
        print(f"4. S_sem + components: Acc={r4['accuracy']:.3f} Prec={r4['precision_macro']:.3f} Rec={r4['recall_macro']:.3f} F1_macro={r4['f1_macro']:.3f}")
    
    best = max(results, key=lambda r: r["accuracy"])
    y_pred_best, y_report = preds[best["model"]]
    
    print("\n" + "=" * 60)
    print(f"MULTI-CLASS: Best = {best['model']} (Acc={best['accuracy']:.3f})")
    print("=" * 60)
    print("\nClassification Report:")
    print(classification_report(y_report, y_pred_best, target_names=target_names))
    print("Confusion Matrix:")
    cm = best["confusion_matrix"]
    print(f"{'':>10} " + " ".join(f"{n:>8}" for n in target_names))
    for i, row in enumerate(cm):
        print(f"{target_names[i]:>10} " + " ".join(f"{v:>8}" for v in row))
    
    # Binary: control vs all PPA (using S_sem model predictions)
    print("\n" + "-" * 60)
    print("BINARY: Control vs All PPA (from S_sem model)")
    print("-" * 60)
    y_binary = (df["label_mc"] > 0).astype(int).values
    y_pred_binary = (y_pred1 > 0).astype(int)
    print(f"Accuracy: {accuracy_score(y_binary, y_pred_binary):.3f}")
    print(f"F1: {f1_score(y_binary, y_pred_binary):.3f}")
    
    # Binary: svPPA vs other PPA
    print("\n" + "-" * 60)
    print("BINARY: svPPA vs Other PPA")
    print("-" * 60)
    mask_ppa = df["label_mc"] > 0
    y_svppa = (df.loc[mask_ppa, "label_mc"] == 1).astype(int).values
    y_pred_svppa = (y_pred1[mask_ppa] == 1).astype(int)
    if len(y_svppa) > 0:
        print(f"Accuracy: {accuracy_score(y_svppa, y_pred_svppa):.3f}")
        print(f"F1: {f1_score(y_svppa, y_pred_svppa):.3f}")
    
    # Save
    out = {
        "models": results,
        "best_model": best["model"],
        "n_samples": len(df),
        "binary_control_vs_ppa": {"accuracy": float(accuracy_score(y_binary, y_pred_binary)), "f1": float(f1_score(y_binary, y_pred_binary))},
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
