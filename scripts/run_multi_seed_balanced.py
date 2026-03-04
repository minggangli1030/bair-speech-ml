#!/usr/bin/env python3
"""
Run Task 2 (svPPA vs nflv) with balanced 25+25 subsample over multiple seeds.
Aggregates: balanced_accuracy, macro-F1, F1_svPPA, F1_nflv, PR-AUC, ROC-AUC (mean ± std),
and confusion matrix averaged over runs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--n-per-class", type=int, default=25)
    parser.add_argument("--features", type=Path, default=Path("outputs/ppa_dissociation_fixed_l2.csv"))
    parser.add_argument("--layer2-fixed", type=Path, default=Path("outputs/layer2_fixed.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/sv_vs_nflf_balanced_multi_seed.json"))
    args = parser.parse_args()

    if not args.features.exists():
        print("Feature file not found. Run pipeline first.")
        sys.exit(1)

    out_dir = args.output.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    single_out = out_dir / "sv_vs_nflf_balanced_results.json"

    runs = []
    for seed in range(args.n_runs):
        subprocess.run([
            sys.executable, "scripts/train_sv_vs_nflf_balanced.py",
            "--features", str(args.features),
            "--layer2-fixed", str(args.layer2_fixed),
            "--output", str(single_out),
            "--seed", str(seed),
            "--n-per-class", str(args.n_per_class),
        ], cwd=REPO_ROOT, check=True)
        with open(single_out) as f:
            runs.append(json.load(f))

    # Aggregate by model name
    model_names = [r["model"] for r in runs[0]["results"]]
    metric_keys = [
        "accuracy", "balanced_accuracy", "macro_f1", "f1_svppa", "f1_nflv",
        "precision_svppa", "recall_svppa", "pr_auc", "roc_auc",
    ]
    aggregated = {}
    for i, name in enumerate(model_names):
        series = [run["results"][i] for run in runs]
        agg = {}
        for k in metric_keys:
            vals = [s.get(k) for s in series if s.get(k) is not None and isinstance(s.get(k), (int, float))]
            if vals:
                arr = np.array(vals, dtype=float)
                agg[k] = {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0}
            else:
                agg[k] = {"mean": None, "std": None}
        # Average confusion matrix over runs (element-wise mean)
        cms = [np.array(s["confusion_matrix"]) for s in series if "confusion_matrix" in s]
        if cms:
            mean_cm = np.mean(cms, axis=0)
            agg["confusion_matrix_mean"] = mean_cm.tolist()
            agg["confusion_matrix_mean_int"] = [[round(mean_cm[0, 0], 1), round(mean_cm[0, 1], 1)], [round(mean_cm[1, 0], 1), round(mean_cm[1, 1], 1)]]
        aggregated[name] = agg

    report = {
        "n_runs": args.n_runs,
        "n_per_class": args.n_per_class,
        "seeds": list(range(args.n_runs)),
        "task": "svPPA vs nflv (balanced 25+25)",
        "threshold_note": runs[0].get("threshold_note", "Default threshold 0.5. Tuning trades recall vs precision."),
        "models_aggregated": aggregated,
        "metrics_reported": [
            "balanced_accuracy", "macro_f1", "f1_svppa", "f1_nflv",
            "precision_svppa", "recall_svppa", "pr_auc", "roc_auc",
            "confusion_matrix_mean (TN, FP; FN, TP)",
        ],
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== Balanced multi-seed complete ===\nResults: {args.output}")
    print("\nBest models by balanced_accuracy (mean ± std):")
    by_bal = [(name, agg["balanced_accuracy"]["mean"]) for name, agg in aggregated.items() if agg["balanced_accuracy"]["mean"] is not None]
    by_bal.sort(key=lambda x: -x[1])
    for name, _ in by_bal[:6]:
        a = aggregated[name]
        bal = a["balanced_accuracy"]
        mf1 = a["macro_f1"]
        f1s = a["f1_svppa"]
        pr = a["pr_auc"]
        cm = a.get("confusion_matrix_mean_int", [])
        print(f"  {name}: BalAcc={bal['mean']:.3f}±{bal['std']:.3f}  macro_F1={mf1['mean']:.3f}  F1_svPPA={f1s['mean']:.3f}  PR-AUC={pr['mean']:.3f}")
        if cm:
            print(f"    CM (mean): TN={cm[0][0]} FP={cm[0][1]}; FN={cm[1][0]} TP={cm[1][1]}")


if __name__ == "__main__":
    main()
