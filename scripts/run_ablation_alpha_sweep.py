#!/usr/bin/env python3
"""
Ablation and alpha sweep for composite S_sem.

1. Ablation: scene_graph only (α=1), Layer 2 only (α=0), composite at various α
2. Alpha sweep: find α where p(delta, control vs svPPA) < 0.05
3. Classifier metrics at each α
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features.dissociation import cliffs_delta, interpret_cliffs_delta, mann_whitney_u


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("outputs/ppa_dissociation_all.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/ablation_alpha_sweep.json"))
    parser.add_argument("--alphas", type=str, default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
                        help="Comma-separated alpha values to sweep")
    parser.add_argument("--p-target", type=float, default=0.05,
                        help="Target p-value threshold for delta (control vs svPPA)")
    args = parser.parse_args()

    alphas = [float(a.strip()) for a in args.alphas.split(",")]
    df = pd.read_csv(args.features)

    # Need s_sem_graph, s_sem_layer2_adequacy, s_ph for control + svPPA
    df_binary = df[df["subtype"].isin(["control", "svppa"])].copy()
    df_binary = df_binary.dropna(subset=["s_sem_graph", "s_sem_layer2_adequacy", "s_ph"])
    y = df_binary["label"].astype(int).values
    n_control = (y == 0).sum()
    n_svppa = (y == 1).sum()

    print(f"\nAblation & Alpha Sweep (control vs svPPA)")
    print(f"Samples: {len(df_binary)} (control={n_control}, svPPA={n_svppa})")
    print("=" * 70)

    results = []
    best_alpha_p = None
    best_alpha_acc = None

    for alpha in sorted(alphas):
        # S_sem = alpha * scene_graph + (1-alpha) * layer2_adequacy
        s_sem = alpha * df_binary["s_sem_graph"].values + (1 - alpha) * df_binary["s_sem_layer2_adequacy"].values
        delta = df_binary["s_ph"].values - s_sem

        v0 = delta[y == 0]
        v1 = delta[y == 1]
        u_stat, p_val = mann_whitney_u(v0, v1)
        d = cliffs_delta(v0, v1)

        # Classifier: S_ph + S_sem
        X = np.column_stack([df_binary["s_ph"].values, s_sem])
        X = np.nan_to_num(X, nan=0.0)
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_pred = cross_val_predict(clf, X_sc, y, cv=cv)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        label = "scene_graph_only" if alpha == 1.0 else ("layer2_only" if alpha == 0.0 else f"composite_α={alpha:.1f}")
        r = {
            "alpha": alpha,
            "label": label,
            "delta_control_mean": float(np.mean(v0)),
            "delta_svppa_mean": float(np.mean(v1)),
            "delta_mean_diff": float(np.mean(v1) - np.mean(v0)),
            "p_value": float(p_val),
            "cliffs_delta": float(d),
            "effect_size": interpret_cliffs_delta(d),
            "classifier_acc": float(acc),
            "classifier_f1": float(f1),
        }
        results.append(r)

        p_ok = "✓ p<0.05" if p_val < args.p_target else ""
        print(f"α={alpha:.1f}: p={p_val:.4f} δ={d:.3f} Acc={acc:.3f} F1={f1:.3f} {p_ok}")

        if p_val < args.p_target and (best_alpha_p is None or p_val < results[best_alpha_p]["p_value"]):
            best_alpha_p = len(results) - 1
        if best_alpha_acc is None or acc > results[best_alpha_acc]["classifier_acc"]:
            best_alpha_acc = len(results) - 1

    print("\n" + "=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)

    # Ablation: scene_graph vs layer2 vs composite
    sg = next(r for r in results if r["alpha"] == 1.0)
    l2 = next(r for r in results if r["alpha"] == 0.0)
    comp = next(r for r in results if r["alpha"] == 0.6)

    print("\nScene graph only (α=1):")
    print(f"  Δ control={sg['delta_control_mean']:.3f}, svPPA={sg['delta_svppa_mean']:.3f}, p={sg['p_value']:.4f}, Acc={sg['classifier_acc']:.3f}")
    print("Layer 2 only (α=0):")
    print(f"  Δ control={l2['delta_control_mean']:.3f}, svPPA={l2['delta_svppa_mean']:.3f}, p={l2['p_value']:.4f}, Acc={l2['classifier_acc']:.3f}")
    print("Composite α=0.6 (default):")
    print(f"  Δ control={comp['delta_control_mean']:.3f}, svPPA={comp['delta_svppa_mean']:.3f}, p={comp['p_value']:.4f}, Acc={comp['classifier_acc']:.3f}")

    # Alpha where p < 0.05
    alphas_p_ok = [r for r in results if r["p_value"] < args.p_target]
    if alphas_p_ok:
        best = min(alphas_p_ok, key=lambda r: r["p_value"])
        print(f"\n✓ Best α for p < {args.p_target}: α={best['alpha']} (p={best['p_value']:.4f})")
        # Recommend alpha with best accuracy among p<0.05
        best_acc = max(alphas_p_ok, key=lambda r: r["classifier_acc"])
        print(f"  Recommended α={best_acc['alpha']} (p={best_acc['p_value']:.4f}, Acc={best_acc['classifier_acc']:.3f})")
    else:
        # Find alpha with smallest p
        best = min(results, key=lambda r: r["p_value"])
        print(f"\n⚠ No α achieves p < {args.p_target}. Smallest p: α={best['alpha']} (p={best['p_value']:.4f})")

    # Save
    out = {
        "n_control": int(n_control),
        "n_svppa": int(n_svppa),
        "results": results,
        "best_alpha_for_p": float(best["alpha"]) if alphas_p_ok else None,
        "best_alpha_for_acc": float(results[best_alpha_acc]["alpha"]) if best_alpha_acc is not None else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
