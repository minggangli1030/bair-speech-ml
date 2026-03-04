#!/usr/bin/env python3
"""
Run training/evaluation with multiple random seeds and report mean ± std (uncertainty).

Runs:
1. Control vs svPPA (train_dissociation_classifier --task control_vs_svppa)
2. svPPA vs nflv (train_sv_vs_nflf_real)

Each experiment is repeated n_runs times with seeds 0, 1, ..., n_runs-1.
Outputs: outputs/multi_seed_results.json (per-run and aggregated mean ± std)
and optionally updates a results summary for Section 5.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_cmd(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def aggregate_metrics(runs: list[dict], metric_keys: list[str]) -> dict:
    """For a list of result dicts (same structure each run), compute mean ± std per key."""
    out = {}
    for k in metric_keys:
        vals = [r.get(k) for r in runs if r.get(k) is not None and isinstance(r.get(k), (int, float))]
        if not vals:
            out[k] = {"mean": None, "std": None}
            continue
        arr = np.array(vals, dtype=float)
        out[k] = {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0}
    return out


def main():
    parser = argparse.ArgumentParser(description="Multi-seed evaluation for uncertainty (mean ± std)")
    parser.add_argument("--n-runs", type=int, default=10, help="Number of runs with different seeds")
    parser.add_argument("--features", type=Path, default=Path("outputs/ppa_dissociation_fixed_l2.csv"))
    parser.add_argument("--layer2-fixed", type=Path, default=Path("outputs/layer2_fixed.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/multi_seed_results.json"))
    args = parser.parse_args()

    if not args.features.exists():
        print("Feature file not found. Run the full pipeline first (e.g. scripts/run_paper_pipeline.sh).")
        sys.exit(1)

    out_dir = args.output.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Experiment 1: Control vs svPPA ---
    control_out = out_dir / "classifier_control_vs_svppa.json"
    runs_control: list[list[dict]] = []
    for seed in range(args.n_runs):
        run_cmd([
            sys.executable, "scripts/train_dissociation_classifier.py",
            "--features", str(args.features),
            "--task", "control_vs_svppa",
            "--output", str(control_out),
            "--seed", str(seed),
        ], REPO_ROOT)
        with open(control_out) as f:
            runs_control.append(json.load(f))

    # Aggregate by model name (each run is a list of model dicts)
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    agg_control = {}
    for i, model in enumerate(runs_control[0]):
        name = model.get("model", f"model_{i}")
        runs_for_model = []
        for run in runs_control:
            if i < len(run):
                runs_for_model.append(run[i])
        agg_control[name] = aggregate_metrics(runs_for_model, metric_keys)

    # --- Experiment 2: svPPA vs nflv ---
    sv_nflv_out = out_dir / "sv_vs_nflf_real_results.json"
    runs_sv: list[dict] = []
    for seed in range(args.n_runs):
        run_cmd([
            sys.executable, "scripts/train_sv_vs_nflf_real.py",
            "--features", str(args.features),
            "--layer2-fixed", str(args.layer2_fixed),
            "--output", str(sv_nflv_out),
            "--seed", str(seed),
        ], REPO_ROOT)
        with open(sv_nflv_out) as f:
            runs_sv.append(json.load(f))

    # Aggregate results list by model name
    if runs_sv and "results" in runs_sv[0]:
        all_models = {r["model"]: [] for r in runs_sv[0]["results"]}
        for run in runs_sv:
            for r in run["results"]:
                all_models.setdefault(r["model"], []).append(r)
        agg_sv = {}
        for name, list_of_r in all_models.items():
            agg_sv[name] = aggregate_metrics(list_of_r, ["accuracy", "f1", "roc_auc", "p_value", "cliffs_delta"])
    else:
        agg_sv = {}

    # Best p-value across runs (mean ± std)
    p_vals = [r.get("best_p_value") for r in runs_sv if r.get("best_p_value") is not None]
    best_p_agg = {"mean": float(np.mean(p_vals)), "std": float(np.std(p_vals, ddof=1))} if p_vals else {}

    report = {
        "n_runs": args.n_runs,
        "seeds": list(range(args.n_runs)),
        "experiment_1_control_vs_svppa": {
            "claim": "Dissociation (S_ph, S_sem, Δ) discriminates control from svPPA.",
            "models_aggregated": agg_control,
            "runs_per_model": len(runs_control),
        },
        "experiment_2_svppa_vs_nflv": {
            "claim": "Among PPA subtypes, dissociation and clinical features discriminate svPPA from nfv+lv.",
            "models_aggregated": agg_sv,
            "best_p_value": best_p_agg,
            "runs_per_model": len(runs_sv),
        },
        "uncertainty_reporting": "All metrics reported as mean ± std over n_runs (different random seeds for CV split and classifier).",
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Multi-seed evaluation complete ===")
    print(f"Runs: {args.n_runs}. Results: {args.output}")
    print("\nControl vs svPPA (mean ± std):")
    for name, agg in agg_control.items():
        acc = agg.get("accuracy", {})
        f1 = agg.get("f1", {})
        auc = agg.get("roc_auc", {})
        if acc.get("mean") is not None:
            print(f"  {name}: Acc = {acc['mean']:.3f} ± {acc['std']:.3f}, F1 = {f1['mean']:.3f} ± {f1['std']:.3f}, AUC = {auc['mean']:.3f} ± {auc['std']:.3f}")
    print("\nsvPPA vs nflv (mean ± std, sample):")
    for name in list(agg_sv.keys())[:5]:
        agg = agg_sv[name]
        acc = agg.get("accuracy", {})
        if acc.get("mean") is not None:
            print(f"  {name}: Acc = {acc['mean']:.3f} ± {acc['std']:.3f}")
    if agg_sv and len(agg_sv) > 5:
        print(f"  ... and {len(agg_sv) - 5} more models.")


if __name__ == "__main__":
    main()
