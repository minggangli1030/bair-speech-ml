#!/usr/bin/env python3
"""
Three-way PPA classification: svPPA vs nfvPPA vs lvPPA (multiclass).

Uses same features and pipeline as binary Task 2. Reports:
- Accuracy, macro-F1, weighted F1
- Per-class F1 (svPPA, nfvPPA, lvPPA)
- Confusion matrix (rows = true, cols = predicted; order sv, nfv, lv)
- ROC-AUC macro (one-vs-rest)
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
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Class order: 0=svPPA, 1=nfvPPA, 2=lvPPA (for consistent confusion matrix)
SUBTYPE_TO_LABEL = {"svppa": 0, "nfvppa": 1, "lvppa": 2}
LABEL_NAMES = ["svPPA", "nfvPPA", "lvPPA"]


def load_ppa_three_way(features_csv: Path, layer2_fixed_csv: Path) -> pd.DataFrame:
    """Load PPA data with three subtypes; merge L2. y in {0,1,2} = sv, nfv, lv."""
    df = pd.read_csv(features_csv)
    df = df[df["subtype"].isin(["svppa", "nfvppa", "lvppa"])].copy()
    df = df.dropna(subset=["s_sem", "s_ph"])
    l2 = pd.read_csv(layer2_fixed_csv)
    l2_cols = ["subject_id", "session"] + [c for c in l2.columns if c.startswith("l2_") and c != "l2_sem_impairment"]
    df = df.merge(l2[l2_cols], on=["subject_id", "session"], how="left")
    df["y"] = df["subtype"].map(SUBTYPE_TO_LABEL)
    if df["y"].isna().any():
        df = df.dropna(subset=["y"])
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
    y_prob = cross_val_predict(clf, X_sc, y, cv=cv, method="predict_proba")
    return y_pred, y_prob


def main():
    parser = argparse.ArgumentParser(description="Three-way PPA: sv vs nfv vs lv")
    parser.add_argument("--features", type=Path, default=Path("outputs/ppa_dissociation_fixed_l2.csv"))
    parser.add_argument("--layer2-fixed", type=Path, default=Path("outputs/layer2_fixed.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/sv_nfv_lv_multiclass_results.json"))
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_ppa_three_way(args.features, args.layer2_fixed)
    y = df["y"].values.astype(int)
    n_sv = int((y == 0).sum())
    n_nfv = int((y == 1).sum())
    n_lv = int((y == 2).sum())
    print(f"Three-way: n_svPPA={n_sv}, n_nfvPPA={n_nfv}, n_lvPPA={n_lv}, total={len(df)}")

    base = ["s_ph", "s_sem"]
    base_graph = ["s_ph", "s_sem", "s_sem_graph"]
    base_delta = ["s_ph", "s_sem", "delta"]
    base_delta_graph = ["s_ph", "s_sem", "s_sem_graph", "delta"]
    acoustic = ["phoneme_error_rate", "silence_ratio"]
    l2_semantic = ["l2_anomia", "l2_empty_speech", "l2_semantic_paraphasias"]
    l2_gram = ["l2_agrammatism", "l2_reduced_speech_rate"]
    l2_all = [c for c in df.columns if c.startswith("l2_") and c != "l2_sem_impairment"]

    def available(cols):
        return [c for c in cols if c in df.columns]

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
        X = df[feat_cols].values
        for clf_name, clf in [
            ("LR", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=args.seed)),
            ("RF", RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=args.seed)),
        ]:
            y_pred, y_prob = run_cv(X, y, clf, args.cv_folds, args.seed)
            acc = accuracy_score(y, y_pred)
            macro_f1 = f1_score(y, y_pred, average="macro", zero_division=0)
            weighted_f1 = f1_score(y, y_pred, average="weighted", zero_division=0)
            f1_per_class = f1_score(y, y_pred, average=None, zero_division=0, labels=[0, 1, 2])
            f1_sv = float(f1_per_class[0])
            f1_nfv = float(f1_per_class[1])
            f1_lv = float(f1_per_class[2])
            try:
                roc_auc = roc_auc_score(y, y_prob, multi_class="ovr", average="macro")
            except Exception:
                roc_auc = float("nan")
            cm = confusion_matrix(y, y_pred, labels=[0, 1, 2])
            cm_list = cm.tolist()

            r = {
                "model": f"{name}_{clf_name}",
                "accuracy": float(acc),
                "macro_f1": float(macro_f1),
                "weighted_f1": float(weighted_f1),
                "f1_svppa": float(f1_sv),
                "f1_nfvppa": float(f1_nfv),
                "f1_lvppa": float(f1_lv),
                "roc_auc_macro": float(roc_auc),
                "confusion_matrix": cm_list,
                "class_order": LABEL_NAMES,
            }
            results.append(r)
            print(f"  {r['model']}: Acc={acc:.3f} macro_F1={macro_f1:.3f} F1_sv={f1_sv:.3f} F1_nfv={f1_nfv:.3f} F1_lv={f1_lv:.3f}")

    out = {
        "n_svppa": n_sv,
        "n_nfvppa": n_nfv,
        "n_lvppa": n_lv,
        "class_order": LABEL_NAMES,
        "seed": args.seed,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
