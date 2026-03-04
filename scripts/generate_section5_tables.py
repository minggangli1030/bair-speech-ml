#!/usr/bin/env python3
"""
Generate Section 5 results tables (mean ± std) from outputs/multi_seed_results.json.

Run after: python scripts/run_multi_seed_evaluation.py --n-runs 10
Output: prints markdown tables and a short summary for pasting into docs/SECTION5_RESULTS.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(x: dict, key: str) -> str:
    m, s = x.get(key, {}).get("mean"), x.get(key, {}).get("std")
    if m is None:
        return "—"
    if s is not None and s > 0:
        return f"{m:.3f} ± {s:.3f}"
    return f"{m:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/multi_seed_results.json"))
    args = parser.parse_args()

    if not args.input.exists():
        print("Run first: python scripts/run_multi_seed_evaluation.py --n-runs 10")
        return

    with open(args.input) as f:
        data = json.load(f)

    n_runs = data.get("n_runs", 0)
    print(f"# Section 5 – Filled tables (from {n_runs} runs)\n")

    # Experiment 1: Control vs svPPA
    exp1 = data.get("experiment_1_control_vs_svppa", {})
    agg1 = exp1.get("models_aggregated", {})
    if agg1:
        print("## 5.2 Control vs svPPA (mean ± std)\n")
        print("| Model | Accuracy | F1 | ROC-AUC |")
        print("|-------|----------|-----|---------|")
        for name, metrics in agg1.items():
            acc = fmt(metrics, "accuracy")
            f1 = fmt(metrics, "f1")
            auc = fmt(metrics, "roc_auc")
            print(f"| {name} | {acc} | {f1} | {auc} |")
        print()

    # Experiment 2: svPPA vs nflv (sample of models)
    exp2 = data.get("experiment_2_svppa_vs_nflv", {})
    agg2 = exp2.get("models_aggregated", {})
    best_p = exp2.get("best_p_value", {})
    if agg2:
        print("## 5.3 svPPA vs nflv (mean ± std)\n")
        print("| Model | Accuracy | F1 | ROC-AUC |")
        print("|-------|----------|-----|---------|")
        for name, metrics in list(agg2.items())[:12]:
            acc = fmt(metrics, "accuracy")
            f1 = fmt(metrics, "f1")
            auc = fmt(metrics, "roc_auc")
            print(f"| {name} | {acc} | {f1} | {auc} |")
        if len(agg2) > 12:
            print(f"| … ({len(agg2) - 12} more) | … | … | … |")
        if best_p and "mean" in best_p:
            m, s = best_p.get("mean"), best_p.get("std")
            print(f"\nBest p-value (Mann–Whitney): {m:.4f} ± {s:.4f}" if s is not None else f"\nBest p-value: {m:.4f}")
        print()

    print("Copy the tables above into docs/SECTION5_RESULTS.md to replace the placeholders.")


if __name__ == "__main__":
    main()
