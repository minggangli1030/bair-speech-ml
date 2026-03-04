#!/usr/bin/env python3
"""
4.6 Longitudinal Analysis (Conditional).

Identifies svPPA subjects with repeated sessions (same subject_id, multiple sessions
e.g. sub-5468 across ses-20070313, ses-20080403, ses-20090403). If present, compares
scores (S_ph, S_sem, Δ) over time and reports consistency. If no longitudinal data,
exits cleanly (section omitted).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def parse_session_to_days(session: str) -> float | None:
    """Parse ses-YYYYMMDD to days since epoch (for ordering and slope)."""
    m = re.match(r"ses-(\d{8})", str(session).strip())
    if not m:
        return None
    y, mo, d = int(m.group(1)[:4]), int(m.group(1)[4:6]), int(m.group(1)[6:8])
    # Simple ordinal days (approximate)
    return y * 365 + mo * 31 + d


def session_to_date_str(session: str) -> str | None:
    """Parse ses-YYYYMMDD to YYYY-MM-DD."""
    m = re.match(r"ses-(\d{4})(\d{2})(\d{2})", str(session).strip())
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"


def main():
    parser = argparse.ArgumentParser(description="Longitudinal analysis for svPPA (repeated speakers)")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("outputs/ppa_dissociation_full.csv"),
        help="CSV with subject_id, session, subtype, s_ph, s_sem, delta",
    )
    parser.add_argument("--output-json", type=Path, default=Path("outputs/longitudinal_results.json"))
    parser.add_argument("--output-md", type=Path, default=Path("outputs/longitudinal_results.md"))
    parser.add_argument("--output-plot-csv", type=Path, default=Path("outputs/longitudinal_plot_data.csv"),
                        help="Plot-ready CSV: years_from_first_visit and all scores per visit")
    args = parser.parse_args()

    if not args.features.exists():
        print("Longitudinal analysis: feature file not found; section omitted.")
        return

    df = pd.read_csv(args.features)
    if "subtype" not in df.columns:
        print("Longitudinal analysis: no subtype column; section omitted.")
        return

    svppa = df[df["subtype"].astype(str).str.strip().str.lower() == "svppa"].copy()
    if len(svppa) == 0:
        print("Longitudinal analysis: no svPPA samples; section omitted.")
        return

    # Repeated speakers: same subject_id, multiple rows
    session_counts = svppa.groupby("subject_id").size()
    repeated_ids = session_counts[session_counts > 1].index.tolist()
    if not repeated_ids:
        print("Longitudinal analysis: no svPPA subjects with multiple sessions; section omitted.")
        return

    # Parse session time
    svppa["_days"] = svppa["session"].apply(parse_session_to_days)
    longitudinal = svppa[svppa["subject_id"].isin(repeated_ids)].copy()
    longitudinal = longitudinal.dropna(subset=["_days"]).sort_values(["subject_id", "_days"])

    # Build plot-ready table: years_from_first_visit, visit_number, time_span_years, all numeric scores
    plot_rows = []
    for sid in repeated_ids:
        sub = longitudinal[longitudinal["subject_id"] == sid].sort_values("_days").copy()
        days = sub["_days"].values
        first_day = days[0]
        last_day = days[-1]
        time_span_years = (last_day - first_day) / 365.0
        n_visits = len(sub)
        for i, (_, row) in enumerate(sub.iterrows()):
            years_from_first = (row["_days"] - first_day) / 365.0
            session_date = session_to_date_str(row["session"])
            out = {
                "subject_id": sid,
                "session": row["session"],
                "session_date": session_date or "",
                "years_from_first_visit": round(years_from_first, 4),
                "visit_number": i + 1,
                "n_visits": n_visits,
                "time_span_years": round(time_span_years, 4),
            }
            for col in longitudinal.columns:
                if col in ("subject_id", "session", "subtype", "label", "_days") or col.startswith("has_"):
                    continue
                try:
                    out[col] = round(float(row[col]), 6) if pd.notna(row[col]) else ""
                except (TypeError, ValueError):
                    pass
            plot_rows.append(out)
    plot_df = pd.DataFrame(plot_rows)
    args.output_plot_csv.parent.mkdir(parents=True, exist_ok=True)
    plot_df.to_csv(args.output_plot_csv, index=False)

    score_cols = [c for c in ["s_ph", "s_sem", "delta"] if c in longitudinal.columns]
    if not score_cols:
        print("Longitudinal analysis: no score columns (s_ph, s_sem, delta); section omitted.")
        return

    # Per-subject stats: first/last, slope (simple linear regression in time), within-subject mean/std
    results = {
        "available": True,
        "n_subjects_longitudinal": len(repeated_ids),
        "subject_ids": repeated_ids,
        "n_sessions_total": int(longitudinal.groupby("subject_id").size().sum()),
        "per_subject": {},
        "summary": {},
    }

    for sid in repeated_ids:
        sub = longitudinal[longitudinal["subject_id"] == sid].sort_values("_days")
        times = sub["_days"].values
        n_sessions = len(sub)
        first_session = sub["session"].iloc[0]
        last_session = sub["session"].iloc[-1]
        entry = {
            "n_sessions": n_sessions,
            "first_session": first_session,
            "last_session": last_session,
            "scores": {},
        }
        for col in score_cols:
            vals = sub[col].astype(float)
            first_val = float(vals.iloc[0])
            last_val = float(vals.iloc[-1])
            change = last_val - first_val
            slope = float((vals.iloc[-1] - vals.iloc[0]) / (times[-1] - times[0])) if (times[-1] - times[0]) != 0 else 0.0
            entry["scores"][col] = {
                "first": first_val,
                "last": last_val,
                "change": change,
                "slope_per_day": slope,
                "mean": float(vals.mean()),
                "std": float(vals.std()) if len(vals) > 1 else 0.0,
            }
        results["per_subject"][sid] = entry

    # Summary: across subjects, mean change and consistency (mean within-subject std)
    for col in score_cols:
        changes = [results["per_subject"][s]["scores"][col]["change"] for s in repeated_ids]
        within_std = [results["per_subject"][s]["scores"][col]["std"] for s in repeated_ids]
        results["summary"][col] = {
            "mean_change_first_to_last": float(pd.Series(changes).mean()),
            "sd_change": float(pd.Series(changes).std()) if len(changes) > 1 else 0.0,
            "mean_within_subject_std": float(pd.Series(within_std).mean()),
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Markdown report
    md_lines = [
        "# Longitudinal Analysis (svPPA repeated speakers)",
        "",
        f"**Condition:** Only run when svPPA has repeated speakers (same `subject_id`, multiple sessions).",
        "",
        f"**Result:** Data available. {results['n_subjects_longitudinal']} svPPA subjects with multiple sessions "
        f"({results['n_sessions_total']} sessions total).",
        "",
        "## Subject IDs with repeated sessions",
        "",
        ", ".join(repeated_ids),
        "",
        "## Score change (first → last session)",
        "",
        "| Subject | Sessions | First session | Last session | Δ (first→last) S_ph | S_sem | delta |",
        "|---------|----------|---------------|--------------|---------------------|-------|-------|",
    ]
    for sid in repeated_ids:
        e = results["per_subject"][sid]
        s_ph_c = e["scores"].get("s_ph", {}).get("change", 0)
        s_sem_c = e["scores"].get("s_sem", {}).get("change", 0)
        d_c = e["scores"].get("delta", {}).get("change", 0)
        md_lines.append(
            f"| {sid} | {e['n_sessions']} | {e['first_session']} | {e['last_session']} | "
            f"{s_ph_c:.4f} | {s_sem_c:.4f} | {d_c:.4f} |"
        )
    md_lines.extend([
        "",
        "## Summary (across longitudinal subjects)",
        "",
    ])
    for col in score_cols:
        s = results["summary"][col]
        md_lines.append(f"- **{col}**: mean change (first→last) = {s['mean_change_first_to_last']:.4f}, "
                        f"SD(change) = {s['sd_change']:.4f}, mean within-subject SD = {s['mean_within_subject_std']:.4f}")
    md_lines.append("")
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # Console summary
    print("=== 4.6 Longitudinal Analysis (Conditional) ===")
    print(f"svPPA repeated speakers: {results['n_subjects_longitudinal']} subjects, {results['n_sessions_total']} sessions.")
    print("Subject IDs:", ", ".join(repeated_ids))
    print("\nScore change (first → last session) summary:")
    for col in score_cols:
        s = results["summary"][col]
        print(f"  {col}: mean change = {s['mean_change_first_to_last']:.4f}, mean within-subject SD = {s['mean_within_subject_std']:.4f}")
    print(f"\nResults saved to {args.output_json}, {args.output_md}, and {args.output_plot_csv}")


if __name__ == "__main__":
    main()
