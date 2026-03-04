#!/usr/bin/env python3
"""
Run three-way (sv vs nfv vs lv) multiclass over multiple seeds.
Aggregates: accuracy, macro-F1, weighted-F1, per-class F1, ROC-AUC, confusion matrix (mean).
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
    parser.add_argument("--features", type=Path, default=Path("outputs/ppa_dissociation_fixed_l2.csv"))
    parser.add_argument("--layer2-fixed", type=Path, default=Path("outputs/layer2_fixed.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/sv_nfv_lv_multiclass_multi_seed.json"))
    args = parser.parse_args()

    if not args.features.exists():
        print("Feature file not found. Run pipeline first.")
        sys.exit(1)

    out_dir = args.output.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    single_out = out_dir / "sv_nfv_lv_multiclass_results.json"

    runs = []
    for seed in range(args.n_runs):
        subprocess.run([
            sys.executable, "scripts/train_sv_nfv_lv_multiclass.py",
            "--features", str(args.features),
            "--layer2-fixed", str(args.layer2_fixed),
            "--output", str(single_out),
            "--seed", str(seed),
        ], cwd=REPO_ROOT, check=True)
        with open(single_out) as f:
            runs.append(json.load(f))

    model_names = [r["model"] for r in runs[0]["results"]]
    metric_keys = [
        "accuracy", "macro_f1", "weighted_f1",
        "f1_svppa", "f1_nfvppa", "f1_lvppa",
        "roc_auc_macro",
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
        cms = [np.array(s["confusion_matrix"]) for s in series if "confusion_matrix" in s]
        if cms:
            mean_cm = np.mean(cms, axis=0)
            agg["confusion_matrix_mean"] = mean_cm.tolist()
            agg["confusion_matrix_mean_rounded"] = [[round(x, 1) for x in row] for row in mean_cm.tolist()]
        aggregated[name] = agg

    report = {
        "n_runs": args.n_runs,
        "task": "sv vs nfv vs lv (3-way multiclass)",
        "class_order": runs[0].get("class_order", ["svPPA", "nfvPPA", "lvPPA"]),
        "n_svppa": runs[0].get("n_svppa"),
        "n_nfvppa": runs[0].get("n_nfvppa"),
        "n_lvppa": runs[0].get("n_lvppa"),
        "models_aggregated": aggregated,
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== Three-way multiclass multi-seed complete ===\nResults: {args.output}")
    print("\nBest models by macro-F1 (mean ± std):")
    by_mf1 = [(name, agg["macro_f1"]["mean"]) for name, agg in aggregated.items() if agg["macro_f1"]["mean"] is not None]
    by_mf1.sort(key=lambda x: -x[1])
    for name, _ in by_mf1[:6]:
        a = aggregated[name]
        mf1 = a["macro_f1"]
        acc = a["accuracy"]
        f1s = a["f1_svppa"]
        f1n = a["f1_nfvppa"]
        f1l = a["f1_lvppa"]
        print(f"  {name}: Acc={acc['mean']:.3f}±{acc['std']:.3f}  macro_F1={mf1['mean']:.3f}±{mf1['std']:.3f}  F1_sv={f1s['mean']:.3f} F1_nfv={f1n['mean']:.3f} F1_lv={f1l['mean']:.3f}")
        if "confusion_matrix_mean_rounded" in a:
            print(f"    CM (mean): {a['confusion_matrix_mean_rounded']}")


if __name__ == "__main__":
    main()
