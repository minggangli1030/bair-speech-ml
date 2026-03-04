#!/usr/bin/env python3
"""
Plot longitudinal score change over time (years from first visit).

Reads outputs/longitudinal_plot_data.csv and produces line plots per subject:
- X-axis: Years from first visit
- Y-axis: S_ph, S_sem, Δ, phoneme_error_rate, silence_ratio, transcript_len

Usage:
  python scripts/plot_longitudinal.py
  python scripts/plot_longitudinal.py --subject sub-5468
  python scripts/plot_longitudinal.py --subject sub-5468 --output outputs/longitudinal_plots/sub-5468.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def main():
    parser = argparse.ArgumentParser(description="Plot longitudinal scores over years from first visit")
    parser.add_argument("--data", type=Path, default=Path("outputs/longitudinal_plot_data.csv"))
    parser.add_argument("--subject", type=str, default=None, help="Plot one subject (e.g. sub-5468); default all")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/longitudinal_plots"))
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    if not args.data.exists():
        print("Plot data not found. Run: python scripts/longitudinal_analysis.py")
        return
    if not HAS_MPL:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    df = pd.read_csv(args.data)
    subjects = [args.subject] if args.subject else df["subject_id"].unique().tolist()
    subjects = [s for s in subjects if s in df["subject_id"].values]
    if not subjects:
        print("No matching subjects.")
        return

    score_cols = [c for c in ["s_ph", "s_sem", "delta", "phoneme_error_rate", "silence_ratio", "transcript_len"]
                  if c in df.columns]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for sid in subjects:
        sub = df[df["subject_id"] == sid].sort_values("years_from_first_visit")
        x = sub["years_from_first_visit"].values
        n_visits = int(sub["n_visits"].iloc[0])
        time_span = float(sub["time_span_years"].iloc[0])

        # One figure: 2 rows — (1) adequacy scores, (2) error rates / transcript length
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        fig.suptitle(f"Longitudinal scores — {sid} ({n_visits} visits, {time_span:.2f} years)", fontsize=12)

        # Panel 1: S_ph, S_sem, delta (adequacy; higher = better except delta can be negative)
        ax1 = axes[0]
        if "s_ph" in score_cols:
            ax1.plot(x, sub["s_ph"], "o-", color="C0", label="S_ph", linewidth=2, markersize=8)
        if "s_sem" in score_cols:
            ax1.plot(x, sub["s_sem"], "s-", color="C1", label="S_sem", linewidth=2, markersize=8)
        if "delta" in score_cols:
            ax1.plot(x, sub["delta"], "^-", color="C2", label="Δ (S_ph − S_sem)", linewidth=2, markersize=8)
        ax1.set_ylabel("Score")
        ax1.set_title("Adequacy scores (S_ph, S_sem, Δ)")
        ax1.legend(loc="best", fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(left=-0.05)

        # Panel 2: phoneme_error_rate, silence_ratio (rates); transcript_len (count, different scale)
        ax2 = axes[1]
        if "phoneme_error_rate" in score_cols:
            ax2.plot(x, sub["phoneme_error_rate"], "o-", color="C3", label="Phoneme error rate", linewidth=2, markersize=8)
        if "silence_ratio" in score_cols:
            ax2.plot(x, sub["silence_ratio"], "s-", color="C4", label="Silence ratio", linewidth=2, markersize=8)
        ax2.set_xlabel("Years from first visit")
        ax2.set_ylabel("Rate [0–1]")
        ax2.set_title("Error and silence rates")
        ax2.legend(loc="best", fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(left=-0.05)
        if "transcript_len" in score_cols:
            ax2_twin = ax2.twinx()
            ax2_twin.plot(x, sub["transcript_len"], "d-", color="C5", label="Transcript length", linewidth=2, markersize=8)
            ax2_twin.set_ylabel("Transcript length (words)")
            ax2_twin.legend(loc="upper right", fontsize=9)

        plt.tight_layout()
        out_path = args.output_dir / f"{sid}.{args.format}"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
