#!/usr/bin/env python3
"""
Ablation study: compare model variants (S_sem only, S_ph+S_sem, +delta, +extras, Layer2 only, full).

Runs the same models as train_dissociation_classifier and saves a summary JSON for the paper.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_dissociation_classifier import load_features, train_and_evaluate


def main():
    parser = argparse.ArgumentParser(description="Ablation study: model variants")
    parser.add_argument("--features", type=Path, default=Path("outputs/ppa_dissociation_fixed_l2.csv"))
    parser.add_argument("--layer2-features", type=Path, default=None)
    parser.add_argument("--task", type=str, default="control_vs_svppa", choices=["control_vs_svppa", "sv_vs_nflf"])
    parser.add_argument("--output", type=Path, default=Path("outputs/ablation_study.json"))
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    df = load_features(args.features, args.layer2_features)
    df = df[df["label"].notna() & df["s_sem"].notna()]

    if args.task == "sv_vs_nflf":
        df = df[df["subtype"].isin(["svppa", "nfvppa", "lvppa"])].copy()
        df["label"] = (df["subtype"] == "svppa").astype(int)

    y = df["label"].astype(int).values
    df_with_delta = df[df["delta"].notna()]
    if len(df_with_delta) < 10 or df_with_delta["label"].nunique() < 2:
        print("Need more samples with delta for ablation.")
        sys.exit(1)

    results = []
    # 1. S_sem only
    X1 = df[["s_sem"]].values
    r1 = train_and_evaluate(X1, y, ["s_sem"], "S_sem_only", args.cv_folds)
    results.append({"model": r1["model"], "accuracy": r1["accuracy"], "f1": r1["f1"], "roc_auc": r1["roc_auc"]})

    # 2. S_ph + S_sem
    X2 = df_with_delta[["s_ph", "s_sem"]].values
    y2 = df_with_delta["label"].astype(int).values
    r2 = train_and_evaluate(X2, y2, ["s_ph", "s_sem"], "S_ph_S_sem", args.cv_folds)
    results.append({"model": r2["model"], "accuracy": r2["accuracy"], "f1": r2["f1"], "roc_auc": r2["roc_auc"]})

    # 3. S_ph + S_sem + delta
    X3 = df_with_delta[["s_ph", "s_sem", "delta"]].values
    r3 = train_and_evaluate(X3, y2, ["s_ph", "s_sem", "delta"], "S_ph_S_sem_Delta", args.cv_folds)
    results.append({"model": r3["model"], "accuracy": r3["accuracy"], "f1": r3["f1"], "roc_auc": r3["roc_auc"]})

    # 4. + extras
    extra = [c for c in ["phoneme_error_rate", "silence_ratio"] if c in df_with_delta.columns]
    if extra:
        feat4 = ["s_ph", "s_sem", "delta"] + extra
        X4 = df_with_delta[feat4].values
        r4 = train_and_evaluate(X4, y2, feat4, "Dissociation_Plus", args.cv_folds)
        results.append({"model": r4["model"], "accuracy": r4["accuracy"], "f1": r4["f1"], "roc_auc": r4["roc_auc"]})

    # 5. Layer2 only (if present)
    l2_cols = [c for c in df.columns if c.startswith("reduced_") or c in [
        "agrammatism", "paragrammatism", "anomia", "empty_speech",
        "semantic_paraphasias", "phonemic_paraphasias_neologisms",
        "self_correction", "overall_communication_impairment"
    ]]
    if not l2_cols:
        l2_cols = [c for c in df.columns if c.startswith("l2_") and c != "l2_sem_impairment"]
    if l2_cols:
        X5 = df[l2_cols].values
        r5 = train_and_evaluate(X5, y, l2_cols, "Layer2_Only", args.cv_folds)
        results.append({"model": r5["model"], "accuracy": r5["accuracy"], "f1": r5["f1"], "roc_auc": r5["roc_auc"]})
        if len(df_with_delta) >= 10:
            feat6 = ["s_ph", "s_sem", "delta"] + l2_cols
            X6 = df_with_delta[feat6].values
            r6 = train_and_evaluate(X6, y2, feat6, "Layer2_Plus_Dissociation", args.cv_folds)
            results.append({"model": r6["model"], "accuracy": r6["accuracy"], "f1": r6["f1"], "roc_auc": r6["roc_auc"]})

    print("\nAblation study (%s)" % args.task)
    print("-" * 50)
    for r in results:
        print(f"  {r['model']}: Acc={r['accuracy']:.3f} F1={r['f1']:.3f} AUC={r['roc_auc']:.3f}")

    out = {"task": args.task, "n_samples": len(df), "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
