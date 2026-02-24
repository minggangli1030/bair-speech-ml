#!/usr/bin/env python3
"""
Ablation: alpha sweep for composite S_sem (scene graph vs Layer 2 weight).

S_sem = alpha * s_sem_graph + (1-alpha) * s_sem_layer2_adequacy.
Sweeps alpha; for each alpha reports p-value for delta (control vs svPPA) and classifier Acc/F1.
Useful to show: scene-graph-only vs Layer2-only vs composite, and which alpha gives p < 0.05.
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
    parser = argparse.ArgumentParser(description="Ablation: alpha sweep for composite S_sem")
    parser.add_argument("--features", type=Path, default=Path("outputs/ppa_dissociation_all.csv"))
    parser.add_argument("--layer2-fixed", type=Path, default=None,
                        help="Optional: CSV with s_sem_layer2_adequacy_fixed (from fix_layer2_population_norm.py)")
    parser.add_argument("--output", type=Path, default=Path("outputs/ablation_alpha_sweep.json"))
    parser.add_argument("--alphas", type=str, default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
                        help="Comma-separated alpha values")
    parser.add_argument("--p-target", type=float, default=0.05)
    args = parser.parse_args()

    alphas = [float(a.strip()) for a in args.alphas.split(",")]
    df = pd.read_csv(args.features)

    if args.layer2_fixed and args.layer2_fixed.exists():
        l2 = pd.read_csv(args.layer2_fixed)
        df = df.merge(
            l2[["subject_id", "session", "s_sem_layer2_adequacy_fixed"]],
            on=["subject_id", "session"],
            how="left",
        )
        if "s_sem_layer2_adequacy_fixed" in df.columns:
            df["s_sem_layer2_adequacy"] = df["s_sem_layer2_adequacy_fixed"].fillna(df["s_sem_layer2_adequacy"])
            print("Using population-normed Layer 2 (s_sem_layer2_adequacy_fixed)")

    df_binary = df[df["subtype"].isin(["control", "svppa"])].copy()
    df_binary = df_binary.dropna(subset=["s_sem_graph", "s_sem_layer2_adequacy", "s_ph"])
    y = df_binary["label"].astype(int).values
    n_control = (y == 0).sum()
    n_svppa = (y == 1).sum()

    print(f"\nAblation & Alpha Sweep (control vs svPPA)")
    print(f"Samples: {len(df_binary)} (control={n_control}, svPPA={n_svppa})")
    print("=" * 70)

    results = []
    for alpha in sorted(alphas):
        s_sem = alpha * df_binary["s_sem_graph"].values + (1 - alpha) * df_binary["s_sem_layer2_adequacy"].values
        delta = df_binary["s_ph"].values - s_sem

        v0 = delta[y == 0]
        v1 = delta[y == 1]
        u_stat, p_val = mann_whitney_u(v0, v1)
        d = cliffs_delta(v0, v1)

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
            "p_value": float(p_val),
            "cliffs_delta": float(d),
            "effect_size": interpret_cliffs_delta(d),
            "classifier_acc": float(acc),
            "classifier_f1": float(f1),
        }
        results.append(r)
        p_ok = "✓ p<0.05" if p_val < args.p_target else ""
        print(f"α={alpha:.1f}: p={p_val:.4f} δ={d:.3f} Acc={acc:.3f} F1={f1:.3f} {p_ok}")

    alphas_p_ok = [r for r in results if r["p_value"] < args.p_target]
    if alphas_p_ok:
        best = min(alphas_p_ok, key=lambda r: r["p_value"])
        print(f"\n✓ Best α for p < {args.p_target}: α={best['alpha']} (p={best['p_value']:.4f})")
        best_acc = max(alphas_p_ok, key=lambda r: r["classifier_acc"])
        print(f"  Recommended α={best_acc['alpha']} (p={best_acc['p_value']:.4f}, Acc={best_acc['classifier_acc']:.3f})")
    else:
        best = min(results, key=lambda r: r["p_value"])
        print(f"\n⚠ No α achieves p < {args.p_target}. Smallest p: α={best['alpha']} (p={best['p_value']:.4f})")

    out = {
        "n_control": int(n_control),
        "n_svppa": int(n_svppa),
        "results": results,
        "best_alpha_for_p": float(best["alpha"]) if alphas_p_ok else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
