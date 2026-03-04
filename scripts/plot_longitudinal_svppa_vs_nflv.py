#!/usr/bin/env python3
"""
Compare longitudinal change: svPPA vs lv+nfv (nfvPPA + lvPPA).

Builds longitudinal data for both groups (subjects with 2+ visits), aggregates
at 3 time points (0, 1, 2 years from first visit). Y-axis: % change from initial
visit (per subject) so change over time is more obvious. Plots mean ± full range.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


TIME_BIN_EDGES = [0.0, 0.5, 1.5, 100.0]
X_POSITIONS = [0.0, 1.0, 2.0]
FIG_WIDTH = 10.0
FIG_HEIGHT = 4.5
ALPHA_SHADE = 0.25


def parse_session_to_days(session: str) -> float | None:
    m = re.match(r"ses-(\d{8})", str(session).strip())
    if not m:
        return None
    y, mo, d = int(m.group(1)[:4]), int(m.group(1)[4:6]), int(m.group(1)[6:8])
    return y * 365 + mo * 31 + d


def build_longitudinal_df(df: pd.DataFrame, subtype_filter: list[str]) -> pd.DataFrame | None:
    """Filter by subtype, keep subjects with 2+ visits, add years_from_first_visit."""
    sub = df[df["subtype"].astype(str).str.strip().str.lower().isin(subtype_filter)].copy()
    if len(sub) == 0:
        return None
    sub["_days"] = sub["session"].apply(parse_session_to_days)
    sub = sub.dropna(subset=["_days"])
    counts = sub.groupby("subject_id").size()
    repeated = counts[counts > 1].index.tolist()
    if not repeated:
        return None
    sub = sub[sub["subject_id"].isin(repeated)].sort_values(["subject_id", "_days"])
    rows = []
    for sid in repeated:
        grp = sub[sub["subject_id"] == sid]
        days = grp["_days"].values
        first_day = days[0]
        for _, row in grp.iterrows():
            years = (row["_days"] - first_day) / 365.0
            out = {"subject_id": sid, "years_from_first_visit": years}
            for c in ["s_ph", "s_sem", "delta"]:
                if c in row.index and pd.notna(row.get(c)):
                    out[c] = float(row[c])
            if "s_ph" in out and "s_sem" in out:
                out["total_score"] = (out["s_ph"] + out["s_sem"]) / 2
            rows.append(out)
    return pd.DataFrame(rows)


def aggregate_three_bins(df: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(x, mean, y_min, y_max) for 3 time bins."""
    y = df["years_from_first_visit"].values.astype(float)
    v = df[col].values.astype(float)
    valid = np.isfinite(v) & np.isfinite(y)
    y, v = y[valid], v[valid]
    if len(y) == 0:
        return np.array(X_POSITIONS), np.array([np.nan] * 3), np.array([np.nan] * 3), np.array([np.nan] * 3)
    means, mins, maxs = [], [], []
    for i in range(3):
        lo, hi = TIME_BIN_EDGES[i], TIME_BIN_EDGES[i + 1]
        mask = (y >= lo) & (y < hi)
        if mask.sum() > 0:
            means.append(np.mean(v[mask]))
            mins.append(np.min(v[mask]))
            maxs.append(np.max(v[mask]))
        else:
            means.append(np.nan)
            mins.append(np.nan)
            maxs.append(np.nan)
    return np.array(X_POSITIONS), np.array(means), np.array(mins), np.array(maxs)


def main():
    parser = argparse.ArgumentParser(description="Longitudinal comparison: svPPA vs lv+nfv")
    parser.add_argument("--data", type=Path, default=Path("outputs/ppa_dissociation_full.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/figures/longitudinal_svppa_vs_nflv.png"))
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    if not HAS_MPL:
        print("matplotlib required.")
        return
    if not args.data.exists():
        print(f"Data not found: {args.data}")
        return

    df = pd.read_csv(args.data)
    if "subtype" not in df.columns:
        print("No subtype column.")
        return

    # Column names may be s_ph/s_sem
    if "S_ph" not in df.columns and "s_ph" in df.columns:
        df["s_ph"] = df["s_ph"]
    if "S_sem" not in df.columns and "s_sem" in df.columns:
        df["s_sem"] = df["s_sem"]

    df_sv = build_longitudinal_df(df, ["svppa"])
    df_nflv = build_longitudinal_df(df, ["nfvppa", "lvppa"])

    if df_sv is None or len(df_sv) == 0:
        print("No svPPA longitudinal data (need 2+ visits per subject).")
        return
    if df_nflv is None or len(df_nflv) == 0:
        print("No lv+nfv longitudinal data (need 2+ visits per subject).")
        return

    n_sv = df_sv["subject_id"].nunique()
    n_nflv = df_nflv["subject_id"].nunique()
    print(f"svPPA: {n_sv} subjects, {len(df_sv)} visits")
    print(f"lv+nfv: {n_nflv} subjects, {len(df_nflv)} visits")

    outcome_cols = ["s_ph", "s_sem", "total_score"]
    if "total_score" not in df_sv.columns:
        df_sv["total_score"] = (df_sv["s_ph"] + df_sv["s_sem"]) / 2
    if "total_score" not in df_nflv.columns:
        df_nflv["total_score"] = (df_nflv["s_ph"] + df_nflv["s_sem"]) / 2

    # % change from initial visit (per subject baseline); baseline must be aligned by subject_id
    def add_pct_change(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        for col in outcome_cols:
            if col not in d.columns:
                continue
            first_idx = d.groupby("subject_id")["years_from_first_visit"].idxmin()
            baseline_series = d.loc[first_idx, col].copy()
            baseline_series.index = first_idx.index  # index = subject_id
            baseline = baseline_series.reindex(d["subject_id"]).values.astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                pct = np.where(baseline > 0, (d[col].values.astype(float) - baseline) / baseline * 100, 0.0)
            d[f"pct_{col}"] = np.where(np.isfinite(pct), pct, 0.0)
        return d

    df_sv = add_pct_change(df_sv)
    df_nflv = add_pct_change(df_nflv)
    pct_cols = [f"pct_{c}" for c in outcome_cols]

    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, FIG_HEIGHT), sharex=True)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, color="lightgray", alpha=0.7)
        ax.set_xlim(0, 2)
        ax.set_xticks([0, 0.5, 1, 1.5, 2])
        ax.set_xlabel("Years from first visit", fontsize=11)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.6)

    titles = ["S_ph (phonological)", "S_sem (semantic)", "Total score"]
    for k, col in enumerate(pct_cols):
        ax = axes[k]
        ax.set_ylabel("% change from initial visit", fontsize=11)
        ax.set_title(titles[k], fontsize=12)

        # svPPA (mean line only, no shaded region)
        x, mu, rlo, rhi = aggregate_three_bins(df_sv, col)
        valid = np.isfinite(mu)
        if valid.any():
            xv, muv = x[valid], mu[valid]
            ax.plot(xv, muv, "o-", color="C0", linewidth=2.5, markersize=8, label=f"svPPA (n={n_sv})")

        # lv+nfv (mean line only)
        x, mu, rlo, rhi = aggregate_three_bins(df_nflv, col)
        valid = np.isfinite(mu)
        if valid.any():
            xv, muv = x[valid], mu[valid]
            ax.plot(xv, muv, "s-", color="C1", linewidth=2.5, markersize=8, label=f"lv+nfv (n={n_nflv})")

        ax.legend(loc="best", fontsize=9)

    # Zoom in: consistent y limits from mean lines only, tight margin
    y_min, y_max = np.inf, -np.inf
    for col in pct_cols:
        for d in (df_sv, df_nflv):
            if col not in d.columns:
                continue
            _, mu, _, _ = aggregate_three_bins(d, col)
            valid = np.isfinite(mu)
            if valid.any():
                y_min = min(y_min, np.min(mu[valid]))
                y_max = max(y_max, np.max(mu[valid]))
    if np.isfinite(y_min) and np.isfinite(y_max):
        margin = (y_max - y_min) * 0.25 or 2.0  # tight zoom on the mean trend
        for ax in axes:
            ax.set_ylim(y_min - margin, y_max + margin)

    fig.suptitle(
        "Longitudinal change: svPPA vs lv+nfv (% change from initial visit)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = args.output if args.output.suffix.lower() == f".{args.format}" else args.output.with_suffix(f".{args.format}")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
