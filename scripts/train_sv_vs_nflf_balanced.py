#!/usr/bin/env python3
"""
Task 2 (svPPA vs nfv+lv) with balanced classes: 25 svPPA, 25 nflv (50 total).

Reports metrics appropriate for (formerly) imbalanced evaluation:
- balanced_accuracy, macro-F1, class-specific F1 (svPPA F1, nflv F1)
- PR-AUC (positive class = svPPA)
- ROC-AUC, confusion matrix
- threshold note (default 0.5; tuning can trade recall vs precision)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_ppa_with_l2(features_csv: Path, layer2_fixed_csv: Path) -> pd.DataFrame:
    """Load dissociation features and merge Layer 2 fixed on (subject_id, session)."""
    df = pd.read_csv(features_csv)
    df = df[df["subtype"].isin(["svppa", "nfvppa", "lvppa"])].copy()
    df = df.dropna(subset=["s_sem", "s_ph"])
    l2 = pd.read_csv(layer2_fixed_csv)
    l2_cols = ["subject_id", "session"] + [c for c in l2.columns if c.startswith("l2_") and c != "l2_sem_impairment"]
    df = df.merge(l2[l2_cols], on=["subject_id", "session"], how="left")
    df["y"] = (df["subtype"] == "svppa").astype(int)
    return df


def impute_median(X: np.ndarray) -> np.ndarray:
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        m = np.nanmedian(col)
        X[np.isnan(X[:, j]), j] = m if not np.isnan(m) else 0.0
    return X


def run_cv(X: np.ndarray, y: np.ndarray, clf, cv_folds: int, seed: int):
    X = impute_median(X)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    y_pred = cross_val_predict(clf, X_sc, y, cv=cv)
    y_prob = cross_val_predict(clf, X_sc, y, cv=cv, method="predict_proba")[:, 1]
    return y_pred, y_prob


def main():
    parser = argparse.ArgumentParser(description="svPPA vs nflv with balanced 25+25 subsample")
    parser.add_argument("--features", type=Path, default=Path("outputs/ppa_dissociation_fixed_l2.csv"))
    parser.add_argument("--layer2-fixed", type=Path, default=Path("outputs/layer2_fixed.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/sv_vs_nflf_balanced_results.json"))
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-per-class", type=int, default=25, help="Sample size per class (25 svPPA, 25 nflv)")
    args = parser.parse_args()

    df = load_ppa_with_l2(args.features, args.layer2_fixed)
    rng = np.random.default_rng(args.seed)

    # Subsample to n_per_class each (25 svPPA, 25 nflv)
    n_per = args.n_per_class
    df_sv = df[df["y"] == 1]
    df_nflv = df[df["y"] == 0]
    if len(df_sv) < n_per or len(df_nflv) < n_per:
        print(f"Not enough samples: svPPA={len(df_sv)}, nflv={len(df_nflv)}. Need >= {n_per} each.")
        sys.exit(1)
    idx_sv = rng.choice(df_sv.index, size=n_per, replace=False)
    idx_nflv = rng.choice(df_nflv.index, size=n_per, replace=False)
    df_bal = pd.concat([df.loc[idx_sv], df.loc[idx_nflv]], axis=0).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    y = df_bal["y"].values
    n_sv, n_nflv = int((y == 1).sum()), int((y == 0).sum())
    print(f"Balanced: n_svPPA={n_sv}, n_nflv={n_nflv}, total={len(df_bal)} (seed={args.seed})")

    # Feature sets (same as real script)
    base = ["s_ph", "s_sem"]
    base_graph = ["s_ph", "s_sem", "s_sem_graph"]
    base_delta = ["s_ph", "s_sem", "delta"]
    base_delta_graph = ["s_ph", "s_sem", "s_sem_graph", "delta"]
    acoustic = ["phoneme_error_rate", "silence_ratio"]
    l2_semantic = ["l2_anomia", "l2_empty_speech", "l2_semantic_paraphasias"]
    l2_gram = ["l2_agrammatism", "l2_reduced_speech_rate"]
    l2_all = [c for c in df_bal.columns if c.startswith("l2_") and c != "l2_sem_impairment"]

    def available(cols):
        return [c for c in cols if c in df_bal.columns]

    feature_sets = [
        ("S_ph_S_sem", available(base)),
        ("S_ph_S_sem_graph", available(base_graph)),
        ("S_ph_S_sem_Delta", available(base_delta)),
        ("S_ph_S_sem_graph_Delta", available(base_delta_graph)),
        ("+Acoustic", available(base_delta + acoustic)),
        ("+Acoustic_graph", available(base_delta_graph + acoustic)),
        ("+L2_semantic", available(base_delta + l2_semantic)),
        ("+L2_gram", available(base_delta + l2_gram)),
        ("+L2_all", available(base_delta + l2_all)),
        ("Full_clinical", available(base_delta + acoustic + l2_semantic + l2_gram)),
        ("Full_clinical_graph", available(base_delta_graph + acoustic + l2_semantic + l2_gram)),
        ("Full_all_L2", available(base_delta + acoustic + l2_all)),
        ("Full_all_L2_graph", available(base_delta_graph + acoustic + l2_all)),
    ]
    feature_sets = [(n, c) for n, c in feature_sets if len(c) >= 2]

    results = []
    for name, feat_cols in feature_sets:
        X = df_bal[feat_cols].values
        for clf_name, clf in [
            ("LR", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=args.seed)),
            ("RF", RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=args.seed)),
        ]:
            y_pred, y_prob = run_cv(X, y, clf, args.cv_folds, args.seed)
            # Metrics
            acc = accuracy_score(y, y_pred)
            bal_acc = balanced_accuracy_score(y, y_pred)
            macro_f1 = f1_score(y, y_pred, average="macro", zero_division=0)
            f1_svppa = f1_score(y, y_pred, pos_label=1, zero_division=0)
            f1_nflv = f1_score(y, y_pred, pos_label=0, zero_division=0)
            prec_svppa = precision_score(y, y_pred, pos_label=1, zero_division=0)
            rec_svppa = recall_score(y, y_pred, pos_label=1, zero_division=0)
            try:
                roc_auc = roc_auc_score(y, y_prob)
            except Exception:
                roc_auc = float("nan")
            try:
                pr_auc = average_precision_score(y, y_prob, pos_label=1)
            except Exception:
                pr_auc = float("nan")
            cm = confusion_matrix(y, y_pred)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

            r = {
                "model": f"{name}_{clf_name}",
                "accuracy": float(acc),
                "balanced_accuracy": float(bal_acc),
                "macro_f1": float(macro_f1),
                "f1_svppa": float(f1_svppa),
                "f1_nflv": float(f1_nflv),
                "precision_svppa": float(prec_svppa),
                "recall_svppa": float(rec_svppa),
                "pr_auc": float(pr_auc),
                "roc_auc": float(roc_auc),
                "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            }
            results.append(r)
            print(f"  {r['model']}: BalAcc={bal_acc:.3f} macro_F1={macro_f1:.3f} F1_svPPA={f1_svppa:.3f} PR-AUC={pr_auc:.3f}")

    # Threshold note
    threshold_note = (
        "Default decision threshold = 0.5. Lowering the threshold (e.g. 0.3) increases recall for svPPA "
        "(fewer missed cases) at the cost of precision (more nflv predicted as svPPA). Raising the threshold "
        "(e.g. 0.6) increases precision and reduces false positives, with lower recall."
    )

    out = {
        "n_per_class": n_per,
        "n_svppa": n_sv,
        "n_nflv": n_nflv,
        "seed": args.seed,
        "threshold_note": threshold_note,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {args.output}")
    print(f"Threshold: {threshold_note[:80]}...")


if __name__ == "__main__":
    main()
