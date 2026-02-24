#!/usr/bin/env python3
"""
P1b + P2: Train classifier on dissociation features and run ablation study.

Models:
1. Baseline: S_sem only
2. S_ph + S_sem
3. S_ph + S_sem + Δ (main model)
4. Full: S_ph + S_sem + Δ + Layer2 features
5. Ablation: Layer2 only (no dissociation)

Reports accuracy, F1, ROC-AUC, confusion matrix.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.dissociation import cliffs_delta, interpret_cliffs_delta, mann_whitney_u


def load_features(dissociation_csv: Path, layer2_csv: Path | None = None) -> pd.DataFrame:
    """Load dissociation features and optionally merge with Layer 2."""
    df = pd.read_csv(dissociation_csv)
    
    # If we have Layer 2 features, merge them
    if layer2_csv and layer2_csv.exists():
        l2 = pd.read_csv(layer2_csv)
        # Merge on subject_id (take first row per subject from l2)
        l2_agg = l2.groupby("subject_id").first().reset_index()
        df = df.merge(l2_agg[["subject_id"] + [c for c in l2_agg.columns if c.startswith("reduced_") or c in [
            "agrammatism", "paragrammatism", "anomia", "empty_speech", 
            "semantic_paraphasias", "phonemic_paraphasias_neologisms",
            "self_correction", "overall_communication_impairment"
        ]]], on="subject_id", how="left")
    
    return df


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    model_name: str,
    cv_folds: int = 5,
) -> dict:
    """Train logistic regression with CV and return metrics."""
    # Impute NaN with median
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        median = np.nanmedian(col)
        X[np.isnan(X[:, j]), j] = median if not np.isnan(median) else 0.0
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )
    
    # Cross-validation predictions
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(clf, X_scaled, y, cv=cv)
    y_prob_cv = cross_val_predict(clf, X_scaled, y, cv=cv, method="predict_proba")[:, 1]
    
    # Fit on all data for coefficients
    clf.fit(X_scaled, y)
    
    # Metrics
    acc = accuracy_score(y, y_pred_cv)
    f1 = f1_score(y, y_pred_cv, average="binary", zero_division=0)
    prec = precision_score(y, y_pred_cv, average="binary", zero_division=0)
    rec = recall_score(y, y_pred_cv, average="binary", zero_division=0)
    try:
        roc = roc_auc_score(y, y_prob_cv)
    except Exception:
        roc = float("nan")
    cm = confusion_matrix(y, y_pred_cv)
    report = classification_report(y, y_pred_cv, target_names=["control", "svppa"], output_dict=True, zero_division=0)
    
    # Coefficients
    coef = clf.coef_.ravel()
    coef_dict = {feature_names[i]: float(coef[i]) for i in range(len(feature_names))}
    top_features = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    
    return {
        "model": model_name,
        "n_samples": len(y),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "coefficients": coef_dict,
        "top_features": top_features,
    }


def main():
    parser = argparse.ArgumentParser(description="Train dissociation classifier + ablation")
    parser.add_argument("--features", type=Path, default=Path("outputs/ppa_dissociation_50.csv"),
                        help="Dissociation features CSV (from run_full_analysis.py)")
    parser.add_argument("--layer2-features", type=Path, default=None,
                        help="Optional Layer 2 features CSV for ablation (from extract command)")
    parser.add_argument("--output", type=Path, default=Path("outputs/classifier_results.json"))
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--binary", action="store_true", default=True,
                        help="Binary classification: svPPA (1) vs rest (0)")
    args = parser.parse_args()
    
    df = load_features(args.features, args.layer2_features)
    
    # Filter to samples with valid labels and at least S_sem
    df = df[df["label"].notna() & df["s_sem"].notna()]
    
    # Binary: svPPA (label=1) vs control (label=0); for multi-class, use label directly
    if args.binary:
        # Keep only control (0) and svPPA (1)
        df = df[df["label"].isin([0, 1])]
        y = df["label"].astype(int).values
    else:
        y = df["label"].astype(int).values
    
    print(f"\nDataset: {len(df)} samples")
    print(f"Label distribution: {pd.Series(y).value_counts().to_dict()}")
    
    results = []
    
    # Model 1: S_sem only
    feat1 = ["s_sem"]
    X1 = df[feat1].values
    r1 = train_and_evaluate(X1, y, feat1, "S_sem_only", args.cv_folds)
    results.append(r1)
    print(f"\n1. S_sem only: Acc={r1['accuracy']:.3f} Prec={r1.get('precision',0):.3f} Rec={r1.get('recall',0):.3f} F1={r1['f1']:.3f} AUC={r1['roc_auc']:.3f}")
    
    # Model 2: S_ph + S_sem (only for samples with S_ph, requires both classes)
    df_with_sph = df[df["s_ph"].notna() & (df["s_ph"] > 0)]
    n_classes_sph = df_with_sph["label"].nunique() if len(df_with_sph) > 0 else 0
    if len(df_with_sph) >= 10 and n_classes_sph >= 2:
        feat2 = ["s_ph", "s_sem"]
        X2 = df_with_sph[feat2].values
        y2 = df_with_sph["label"].astype(int).values
        r2 = train_and_evaluate(X2, y2, feat2, "S_ph_S_sem", args.cv_folds)
        results.append(r2)
        print(f"2. S_ph + S_sem: Acc={r2['accuracy']:.3f} Prec={r2.get('precision',0):.3f} Rec={r2.get('recall',0):.3f} F1={r2['f1']:.3f} AUC={r2['roc_auc']:.3f} (n={len(df_with_sph)})")
    else:
        print(f"2. S_ph + S_sem: Skipped ({len(df_with_sph)} samples, {n_classes_sph} class(es) - need S_ph for controls)")
    
    # Model 3: S_ph + S_sem + Δ (requires both classes)
    df_with_delta = df[df["delta"].notna()]
    n_classes_delta = df_with_delta["label"].nunique() if len(df_with_delta) > 0 else 0
    if len(df_with_delta) >= 10 and n_classes_delta >= 2:
        feat3 = ["s_ph", "s_sem", "delta"]
        X3 = df_with_delta[feat3].values
        y3 = df_with_delta["label"].astype(int).values
        r3 = train_and_evaluate(X3, y3, feat3, "S_ph_S_sem_Delta", args.cv_folds)
        results.append(r3)
        print(f"3. S_ph + S_sem + Δ: Acc={r3['accuracy']:.3f} Prec={r3.get('precision',0):.3f} Rec={r3.get('recall',0):.3f} F1={r3['f1']:.3f} AUC={r3['roc_auc']:.3f} (n={len(df_with_delta)})")
    else:
        print(f"3. S_ph + S_sem + Δ: Skipped ({len(df_with_delta)} samples, {n_classes_delta} class(es) - need Δ for controls)")
    
    # Model 4: Dissociation + additional features (phoneme_error_rate, silence_ratio)
    extra_feats = ["phoneme_error_rate", "silence_ratio"]
    available_extra = [f for f in extra_feats if f in df.columns]
    if available_extra and len(df_with_delta) >= 10 and n_classes_delta >= 2:
        feat4 = ["s_ph", "s_sem", "delta"] + available_extra
        X4 = df_with_delta[feat4].values
        y4 = df_with_delta["label"].astype(int).values
        r4 = train_and_evaluate(X4, y4, feat4, "Dissociation_Plus", args.cv_folds)
        results.append(r4)
        print(f"4. Dissociation + extras: Acc={r4['accuracy']:.3f} Prec={r4.get('precision',0):.3f} Rec={r4.get('recall',0):.3f} F1={r4['f1']:.3f}")
    
    # Model 5: Layer 2 features only (ablation)
    l2_cols = [c for c in df.columns if c.startswith("reduced_") or c in [
        "agrammatism", "paragrammatism", "anomia", "empty_speech",
        "semantic_paraphasias", "phonemic_paraphasias_neologisms",
        "self_correction", "overall_communication_impairment"
    ]]
    if l2_cols:
        feat5 = l2_cols
        X5 = df[feat5].values
        r5 = train_and_evaluate(X5, y, feat5, "Layer2_Only", args.cv_folds)
        results.append(r5)
        print(f"5. Layer 2 only: Acc={r5['accuracy']:.3f} Prec={r5.get('precision',0):.3f} Rec={r5.get('recall',0):.3f} F1={r5['f1']:.3f}")
        
        # Model 6: Layer 2 + Dissociation (full ablation)
        if len(df_with_delta) >= 10 and n_classes_delta >= 2:
            feat6 = ["s_ph", "s_sem", "delta"] + l2_cols
            X6 = df_with_delta[feat6].values
            y6 = df_with_delta["label"].astype(int).values
            r6 = train_and_evaluate(X6, y6, feat6, "Layer2_Plus_Dissociation", args.cv_folds)
            results.append(r6)
            print(f"6. Layer 2 + Dissociation: Acc={r6['accuracy']:.3f} Prec={r6.get('precision',0):.3f} Rec={r6.get('recall',0):.3f} F1={r6['f1']:.3f}")
    
    # Statistical comparison: S_sem between groups
    print("\n" + "="*60)
    print("STATISTICAL COMPARISONS")
    print("="*60)
    
    for feat in ["s_sem", "s_ph", "delta"]:
        if feat not in df.columns:
            continue
        for label_val, label_name in [(0, "control"), (1, "svppa")]:
            vals = df[df["label"] == label_val][feat].dropna()
            if len(vals) > 0:
                print(f"{feat} ({label_name}): n={len(vals)}, mean={vals.mean():.3f}, std={vals.std():.3f}")
    
    # Pairwise comparison
    if 0 in df["label"].values and 1 in df["label"].values:
        for feat in ["s_sem", "s_ph", "delta"]:
            if feat not in df.columns:
                continue
            v0 = df[df["label"] == 0][feat].dropna().values
            v1 = df[df["label"] == 1][feat].dropna().values
            if len(v0) >= 2 and len(v1) >= 2:
                u, p = mann_whitney_u(v0, v1)
                d = cliffs_delta(v0, v1)
                print(f"\n{feat} (control vs svppa): U={u:.1f}, p={p:.6f}, δ={d:.3f} ({interpret_cliffs_delta(d)})")
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY: Classification Results (control vs svPPA)")
    print("="*70)
    print(f"{'Model':<28} {'N':>4} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
    print("-"*70)
    for r in results:
        prec = r.get("precision", float("nan"))
        rec = r.get("recall", float("nan"))
        print(f"{r['model']:<28} {r['n_samples']:>4} {r['accuracy']:>6.3f} {prec:>6.3f} {rec:>6.3f} {r['f1']:>6.3f} {r['roc_auc']:>6.3f}")


if __name__ == "__main__":
    main()
