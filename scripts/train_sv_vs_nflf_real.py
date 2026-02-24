#!/usr/bin/env python3
"""
svPPA vs (nfv+lv) on REAL data only: maximize significance with clinical features.

- Merges Layer 2 fixed (l2_anomia, l2_empty_speech, etc.) into features
- Tries multiple feature sets and LR + RandomForest
- Reports Mann-Whitney p on predicted P(svPPA) and on best univariate/composite
- No simulated data
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features.dissociation import cliffs_delta, interpret_cliffs_delta, mann_whitney_u


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


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    clf,
    cv_folds: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return oof predictions and probabilities (class 1)."""
    X = impute_median(X)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X_sc, y, cv=cv)
    y_prob = cross_val_predict(clf, X_sc, y, cv=cv, method="predict_proba")[:, 1]
    return y_pred, y_prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("outputs/ppa_dissociation_fixed_l2.csv"))
    parser.add_argument("--layer2-fixed", type=Path, default=Path("outputs/layer2_fixed.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/sv_vs_nflf_real_results.json"))
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    df = load_ppa_with_l2(args.features, args.layer2_fixed)
    y = df["y"].values
    n_sv = (y == 1).sum()
    n_nflf = (y == 0).sum()
    print(f"Real PPA only: n_svPPA={n_sv}, n_nflv={n_nflf}, total={len(df)}")

    # Feature sets (clinically grounded). s_sem = composite; s_sem_graph = spatiosemantic scene graph.
    base = ["s_ph", "s_sem"]
    base_graph = ["s_ph", "s_sem", "s_sem_graph"]  # explicit scene graph
    base_delta = ["s_ph", "s_sem", "delta"]
    base_delta_graph = ["s_ph", "s_sem", "s_sem_graph", "delta"]  # composite + scene graph + delta
    acoustic = ["phoneme_error_rate", "silence_ratio"]
    l2_semantic = ["l2_anomia", "l2_empty_speech", "l2_semantic_paraphasias"]
    l2_gram = ["l2_agrammatism", "l2_reduced_speech_rate"]
    l2_all = [c for c in df.columns if c.startswith("l2_") and c != "l2_sem_impairment"]

    def available(cols):
        return [c for c in cols if c in df.columns]

    feature_sets = [
        ("S_ph_S_sem", available(base)),
        ("S_ph_S_sem_graph", available(base_graph)),  # spatiosemantic scene graph explicit
        ("S_ph_S_sem_Delta", available(base_delta)),
        ("S_ph_S_sem_graph_Delta", available(base_delta_graph)),  # composite + scene graph + delta
        ("+Acoustic", available(base_delta + acoustic)),
        ("+Acoustic_graph", available(base_delta_graph + acoustic)),  # includes scene graph
        ("+L2_semantic", available(base_delta + l2_semantic)),
        ("+L2_gram", available(base_delta + l2_gram)),
        ("+L2_all", available(base_delta + l2_all)),
        ("Full_clinical", available(base_delta + acoustic + l2_semantic + l2_gram)),
        ("Full_clinical_graph", available(base_delta_graph + acoustic + l2_semantic + l2_gram)),  # scene graph in
        ("Full_all_L2", available(base_delta + acoustic + l2_all)),
        ("Full_all_L2_graph", available(base_delta_graph + acoustic + l2_all)),  # scene graph in
    ]
    feature_sets = [(n, c) for n, c in feature_sets if len(c) >= 2]

    results = []
    best_p = 1.0
    best_result = None

    for name, feat_cols in feature_sets:
        X = df[feat_cols].values

        for clf_name, clf in [
            ("LR", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
            ("RF", RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42)),
        ]:
            y_pred, y_prob = run_cv(X, y, clf, args.cv_folds)
            acc = accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred, zero_division=0)
            try:
                auc = roc_auc_score(y, y_prob)
            except Exception:
                auc = float("nan")

            # Mann-Whitney on predicted P(svPPA): real sv vs real nflv
            prob_sv = y_prob[y == 1]
            prob_nflv = y_prob[y == 0]
            u, p = mann_whitney_u(prob_sv, prob_nflv)
            d = cliffs_delta(prob_sv, prob_nflv)

            r = {
                "model": f"{name}_{clf_name}",
                "features": feat_cols,
                "n_features": len(feat_cols),
                "accuracy": float(acc),
                "f1": float(f1),
                "roc_auc": float(auc),
                "mann_whitney_u": float(u),
                "p_value": float(p),
                "cliffs_delta": float(d),
                "effect_size": interpret_cliffs_delta(d),
            }
            results.append(r)
            sig = "***" if p < 0.05 else ""
            print(f"  {r['model']}: Acc={acc:.3f} F1={f1:.3f} AUC={auc:.3f}  p={p:.4f} δ={d:.3f} {sig}")

            if p < best_p:
                best_p = p
                best_result = r

    # Univariate: best single feature for group difference
    print("\n--- Univariate Mann-Whitney (real sv vs nflv) ---")
    cand = ["s_sem", "s_ph", "delta", "s_sem_graph", "s_sem_layer2_adequacy_fixed"] + available(l2_semantic + l2_gram)
    cand = [c for c in cand if c in df.columns]
    for col in cand:
        v_sv = df.loc[y == 1, col].dropna().values
        v_nflv = df.loc[y == 0, col].dropna().values
        if len(v_sv) >= 2 and len(v_nflv) >= 2:
            u, p = mann_whitney_u(v_sv, v_nflv)
            d = cliffs_delta(v_sv, v_nflv)
            sig = " ***" if p < 0.05 else ""
            print(f"  {col}: p={p:.4f} δ={d:.3f}{sig}")
            if p < best_p:
                best_p = p
                best_result = {"model": f"univariate_{col}", "p_value": float(p), "cliffs_delta": float(d)}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "n_svppa": int(n_sv),
        "n_nflv": int(n_nflf),
        "results": results,
        "best_p_value": float(best_p),
        "best_result": best_result,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {args.output}")
    print(f"Best p-value: {best_p:.4f}" + (" (significant)" if best_p < 0.05 else " (not significant)"))


if __name__ == "__main__":
    main()
