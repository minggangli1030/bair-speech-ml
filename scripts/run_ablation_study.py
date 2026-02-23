#!/usr/bin/env python3
"""
P2: Ablation study comparing different feature sets.

Models tested:
1. S_sem only (baseline semantic)
2. Layer 2 constructs only (clinical features)  
3. S_sem + Layer 2 (combined)
4. When Δ available: S_ph + S_sem + Δ
5. When Δ available: S_ph + S_sem + Δ + Layer 2 (full model)

Reports accuracy, F1, AUC and feature importances.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    model_name: str,
    cv_folds: int = 5,
) -> dict:
    """Train logistic regression with CV and return metrics."""
    # Impute NaN with 0
    X = np.nan_to_num(X, nan=0.0)
    
    # Check we have at least 2 classes
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        return {
            "model": model_name,
            "n_samples": len(y),
            "n_features": len(feature_names),
            "accuracy": float("nan"),
            "f1": float("nan"),
            "roc_auc": float("nan"),
            "error": f"Only {len(unique_classes)} class(es) present",
        }
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    
    # Adjust CV folds if needed
    min_class_count = min(np.bincount(y))
    cv_folds = min(cv_folds, min_class_count)
    if cv_folds < 2:
        cv_folds = 2
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    try:
        y_pred_cv = cross_val_predict(clf, X_scaled, y, cv=cv)
        y_prob_cv = cross_val_predict(clf, X_scaled, y, cv=cv, method="predict_proba")[:, 1]
    except Exception as e:
        return {
            "model": model_name,
            "n_samples": len(y),
            "n_features": len(feature_names),
            "accuracy": float("nan"),
            "f1": float("nan"),
            "roc_auc": float("nan"),
            "error": str(e),
        }
    
    # Fit on all data for coefficients
    clf.fit(X_scaled, y)
    
    # Metrics
    acc = accuracy_score(y, y_pred_cv)
    f1 = f1_score(y, y_pred_cv, average="binary")
    try:
        roc = roc_auc_score(y, y_prob_cv)
    except:
        roc = float("nan")
    
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
        "f1": float(f1),
        "roc_auc": float(roc),
        "top_features": top_features,
    }


def main():
    parser = argparse.ArgumentParser(description="P2: Ablation study with Layer 1/2 features")
    parser.add_argument("--features", type=Path, default=Path("outputs/features.csv"),
                        help="Features CSV with Layer 1/2 columns")
    parser.add_argument("--dissociation", type=Path, default=None,
                        help="Optional dissociation CSV to merge (for S_sem, S_ph, Δ)")
    parser.add_argument("--output", type=Path, default=Path("outputs/ablation_results.json"))
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()
    
    df = pd.read_csv(args.features)
    
    # Merge dissociation features if provided
    if args.dissociation and args.dissociation.exists():
        diss = pd.read_csv(args.dissociation)
        df = df.merge(diss[["subject_id", "s_sem", "s_ph", "delta"]].dropna(subset=["s_sem"]), 
                      on="subject_id", how="left")
    
    # Get label
    if "label" not in df.columns:
        print("ERROR: No 'label' column found")
        return
    
    y = df["label"].astype(int).values
    
    print(f"\nDataset: {len(df)} samples")
    print(f"Label distribution: {pd.Series(y).value_counts().to_dict()}")
    
    # Define feature groups
    layer2_cols = [c for c in df.columns if c.startswith("reduced_") or c in [
        "agrammatism", "paragrammatism", "anomia", "empty_speech",
        "semantic_paraphasias", "phonemic_paraphasias_neologisms",
        "self_correction", "overall_communication_impairment"
    ]]
    
    layer1_cols = [c for c in df.columns if c in [
        "wpm", "articulation_rate", "articulation_rate_phones_per_sec",
        "mean_pause_duration", "pause_to_speech_ratio", "mlu",
        "type_token_ratio", "content_word_density", "function_word_omission_rate"
    ]]
    
    results = []
    
    # Model 1: S_sem only (if available)
    if "s_sem" in df.columns and df["s_sem"].notna().sum() > 10:
        df_sem = df[df["s_sem"].notna()]
        feat1 = ["s_sem"]
        X1 = df_sem[feat1].values
        y1 = df_sem["label"].astype(int).values
        r1 = train_and_evaluate(X1, y1, feat1, "S_sem_only", args.cv_folds)
        results.append(r1)
        print(f"\n1. S_sem only: Acc={r1.get('accuracy', 'N/A'):.3f}, F1={r1.get('f1', 'N/A'):.3f}, AUC={r1.get('roc_auc', 'N/A'):.3f}")
    else:
        print("\n1. S_sem only: Skipped (no S_sem data)")
    
    # Model 2: Layer 2 constructs only
    if layer2_cols:
        feat2 = layer2_cols
        X2 = df[feat2].values
        r2 = train_and_evaluate(X2, y, feat2, "Layer2_only", args.cv_folds)
        results.append(r2)
        print(f"2. Layer 2 only ({len(layer2_cols)} feats): Acc={r2['accuracy']:.3f}, F1={r2['f1']:.3f}, AUC={r2['roc_auc']:.3f}")
    
    # Model 3: Layer 1 only
    if layer1_cols:
        feat3 = layer1_cols
        X3 = df[feat3].values
        r3 = train_and_evaluate(X3, y, feat3, "Layer1_only", args.cv_folds)
        results.append(r3)
        print(f"3. Layer 1 only ({len(layer1_cols)} feats): Acc={r3['accuracy']:.3f}, F1={r3['f1']:.3f}, AUC={r3['roc_auc']:.3f}")
    
    # Model 4: Layer 1 + Layer 2
    all_layers = layer1_cols + layer2_cols
    if all_layers:
        feat4 = all_layers
        X4 = df[feat4].values
        r4 = train_and_evaluate(X4, y, feat4, "Layer1_plus_Layer2", args.cv_folds)
        results.append(r4)
        print(f"4. Layer 1 + Layer 2 ({len(all_layers)} feats): Acc={r4['accuracy']:.3f}, F1={r4['f1']:.3f}, AUC={r4['roc_auc']:.3f}")
    
    # Model 5: S_sem + Layer 2 (if S_sem available)
    if "s_sem" in df.columns and df["s_sem"].notna().sum() > 10 and layer2_cols:
        df_sem = df[df["s_sem"].notna()]
        feat5 = ["s_sem"] + layer2_cols
        X5 = df_sem[feat5].values
        y5 = df_sem["label"].astype(int).values
        r5 = train_and_evaluate(X5, y5, feat5, "S_sem_plus_Layer2", args.cv_folds)
        results.append(r5)
        print(f"5. S_sem + Layer 2: Acc={r5['accuracy']:.3f}, F1={r5['f1']:.3f}, AUC={r5['roc_auc']:.3f}")
    
    # Model 6: Full dissociation (if Δ available)
    if "delta" in df.columns and df["delta"].notna().sum() > 10:
        df_delta = df[df["delta"].notna()]
        n_classes = df_delta["label"].nunique()
        if n_classes >= 2:
            feat6 = ["s_ph", "s_sem", "delta"]
            X6 = df_delta[feat6].values
            y6 = df_delta["label"].astype(int).values
            r6 = train_and_evaluate(X6, y6, feat6, "Dissociation_S_ph_S_sem_Delta", args.cv_folds)
            results.append(r6)
            print(f"6. Dissociation (Δ): Acc={r6['accuracy']:.3f}, F1={r6['f1']:.3f}, AUC={r6['roc_auc']:.3f}")
            
            # Model 7: Dissociation + Layer 2
            if layer2_cols:
                feat7 = ["s_ph", "s_sem", "delta"] + layer2_cols
                X7 = df_delta[feat7].values
                r7 = train_and_evaluate(X7, y6, feat7, "Dissociation_plus_Layer2", args.cv_folds)
                results.append(r7)
                print(f"7. Dissociation + Layer 2: Acc={r7['accuracy']:.3f}, F1={r7['f1']:.3f}, AUC={r7['roc_auc']:.3f}")
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary table
    print("\n" + "=" * 70)
    print("ABLATION STUDY SUMMARY")
    print("=" * 70)
    print(f"{'Model':<35} {'N':>5} {'Feats':>6} {'Acc':>6} {'F1':>6} {'AUC':>6}")
    print("-" * 70)
    for r in results:
        acc = r.get('accuracy', float('nan'))
        f1 = r.get('f1', float('nan'))
        auc = r.get('roc_auc', float('nan'))
        print(f"{r['model']:<35} {r['n_samples']:>5} {r['n_features']:>6} {acc:>6.3f} {f1:>6.3f} {auc:>6.3f}")
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
