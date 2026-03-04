#!/usr/bin/env python3
"""
Compare results BEFORE (original pipeline) vs AFTER (manual diarization data pipeline).

Reads:
  - Before: outputs/classifier_results.json or classifier_control_vs_svppa, outputs/sv_vs_nflf_real_results.json,
            outputs/ppa_stats.json, outputs/ppa_dissociation_full.csv
  - After:  outputs/classifier_control_vs_svppa_manual.json, outputs/sv_vs_nflf_real_results_manual.json,
            outputs/ppa_stats_manual.json, outputs/ppa_dissociation_manual.csv

Prints a side-by-side comparison of accuracy, F1, ROC-AUC, and key stats.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    repo = Path(__file__).resolve().parent.parent
    out = repo / "outputs"

    # Before (original)
    before_control = load_json(out / "classifier_control_vs_svppa.json") or load_json(out / "classifier_results.json")
    before_sv_nflf = load_json(out / "sv_vs_nflf_real_results.json")
    before_stats = load_json(out / "ppa_stats.json")

    # After (manual diarization)
    after_control = load_json(out / "classifier_control_vs_svppa_manual.json")
    after_sv_nflf = load_json(out / "sv_vs_nflf_real_results_manual.json")
    after_stats = load_json(out / "ppa_stats_manual.json")

    def acc_f1_auc_control(data):
        if not data or not isinstance(data, list):
            return None, None, None
        # Prefer S_ph_S_sem or S_ph_S_sem_Delta model
        for m in data:
            if m.get("model") in ("S_ph_S_sem", "S_ph_S_sem_Delta"):
                return m.get("accuracy"), m.get("f1"), m.get("roc_auc")
        if data:
            m = data[0]
            return m.get("accuracy"), m.get("f1"), m.get("roc_auc")
        return None, None, None

    def best_sv_nflf(data):
        if not data or "best_result" not in data:
            return None
        r = data["best_result"]
        return {
            "model": r.get("model"),
            "accuracy": r.get("accuracy"),
            "f1": r.get("f1"),
            "roc_auc": r.get("roc_auc"),
            "p_value": r.get("p_value"),
            "effect_size": r.get("effect_size"),
        }

    print()
    print("=" * 72)
    print("  BEFORE vs AFTER: Manual Diarization Data Pipeline Comparison")
    print("=" * 72)
    print()
    print("  BEFORE = original pipeline (data/ manifest, data/ppa_pipeline_outputs)")
    print("  AFTER  = pipeline on Manual Diarization data (data/Manual Diarization)")
    print()

    # Control vs svPPA classifier
    print("-" * 72)
    print("  1. Control vs svPPA classifier (binary)")
    print("-" * 72)
    b_acc, b_f1, b_auc = acc_f1_auc_control(before_control)
    a_acc, a_f1, a_auc = acc_f1_auc_control(after_control)
    if b_acc is not None or a_acc is not None:
        print(f"  {'Metric':<20} {'BEFORE':>14} {'AFTER':>14} {'Change':>14}")
        print(f"  {'Accuracy':<20} {b_acc if b_acc is not None else '—':>14} {a_acc if a_acc is not None else '—':>14} "
              f"{(a_acc - b_acc) if (a_acc is not None and b_acc is not None) else '—':>14}")
        print(f"  {'F1':<20} {b_f1 if b_f1 is not None else '—':>14} {a_f1 if a_f1 is not None else '—':>14} "
              f"{(a_f1 - b_f1) if (a_f1 is not None and b_f1 is not None) else '—':>14}")
        print(f"  {'ROC-AUC':<20} {b_auc if b_auc is not None else '—':>14} {a_auc if a_auc is not None else '—':>14} "
              f"{(a_auc - b_auc) if (a_auc is not None and b_auc is not None) else '—':>14}")
    else:
        print("  No control vs svPPA results found (before or after).")
    print()

    # svPPA vs nfv+lv
    print("-" * 72)
    print("  2. svPPA vs (nfv+lv) PPA — best model")
    print("-" * 72)
    b_sv = best_sv_nflf(before_sv_nflf)
    a_sv = best_sv_nflf(after_sv_nflf)
    if b_sv or a_sv:
        print(f"  {'Metric':<20} {'BEFORE':>14} {'AFTER':>14} {'Change':>14}")
        for key in ("accuracy", "f1", "roc_auc"):
            bv = b_sv.get(key) if b_sv else None
            av = a_sv.get(key) if a_sv else None
            ch = (av - bv) if (av is not None and bv is not None) else None
            print(f"  {key:<20} {bv if bv is not None else '—':>14} {av if av is not None else '—':>14} {ch if ch is not None else '—':>14}")
        if b_sv:
            print(f"  {'Best model (before)':<20} {str(b_sv.get('model',''))}")
        if a_sv:
            print(f"  {'Best model (after)':<20} {str(a_sv.get('model',''))}")
        if b_sv and a_sv:
            print(f"  {'p_value (before)':<20} {b_sv.get('p_value')}")
            print(f"  {'p_value (after)':<20} {a_sv.get('p_value')}")
    else:
        print("  No sv vs nflf results found.")
    print()

    # Sample counts
    print("-" * 72)
    print("  3. Sample counts")
    print("-" * 72)
    import pandas as pd
    for name, p in [("BEFORE", out / "ppa_dissociation_full.csv"), ("BEFORE (all)", out / "ppa_dissociation_all.csv"),
                    ("AFTER", out / "ppa_dissociation_manual.csv")]:
        if p.exists():
            df = pd.read_csv(p)
            n = len(df)
            n_sylwav = df["has_sylwav"].sum() if "has_sylwav" in df.columns else "—"
            n_trans = df["has_transcript"].sum() if "has_transcript" in df.columns else "—"
            print(f"  {name}: n={n}, with_sylwav={n_sylwav}, with_transcript={n_trans}")
    print()
    print("=" * 72)
    print()
    print("Note: If you ran without HF_TOKEN, AFTER used copied pipeline outputs (same")
    print("features as BEFORE). Set HF_TOKEN and re-run run_manual_diarization_pipeline.sh")
    print("to re-diarize and re-ASR the Manual Diarization audio for a true comparison.")


if __name__ == "__main__":
    main()
