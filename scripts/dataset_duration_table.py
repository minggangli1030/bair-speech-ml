#!/usr/bin/env python3
"""
Compute Table 1: Dataset composition and duration summary for WAB Picnic recordings.
Reads data/manifest_ppa_full.csv and measures each audio file under data/.
Output: prints a markdown table and optionally writes to CSV.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import soundfile as sf
except ImportError:
    sf = None


def get_duration_sec(audio_path: Path) -> float:
    if sf is None:
        raise RuntimeError("Install soundfile: pip install soundfile")
    try:
        info = sf.info(str(audio_path))
        return info.duration
    except Exception:
        data, sr = sf.read(str(audio_path))
        return len(data) / sr


def main():
    parser = argparse.ArgumentParser(description="Dataset duration summary (Table 1)")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest_ppa_full.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=None, help="Optional: write table to CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    rows_data = []
    for _, row in df.iterrows():
        audio_path = args.data_dir / row["audio_path"]
        if not audio_path.exists():
            continue
        dur = get_duration_sec(audio_path)
        rows_data.append({"subtype": row["subtype"], "label": row["label"], "duration_sec": dur})

    dur_df = pd.DataFrame(rows_data)
    group_specs = [
        ("Healthy controls", 0, "control"),
        ("svPPA", 1, "svppa"),
        ("nfvPPA", 2, "nfvppa"),
        ("lvPPA", 3, "lvppa"),
    ]

    table_rows = []
    for group_name, label, subtype in group_specs:
        sub = dur_df[dur_df["subtype"] == subtype]
        n = len(sub)
        if n == 0:
            continue
        secs = sub["duration_sec"].values
        avg_s = float(np.mean(secs))
        var_s2 = float(np.var(secs, ddof=0))
        total_h = float(secs.sum() / 3600)
        table_rows.append({
            "Group": group_name,
            "Recordings": n,
            "Label": label,
            "Avg len (s)": round(avg_s, 2),
            "Var (s²)": round(var_s2, 2),
            "Total len (h)": round(total_h, 2),
            "Subtype n": n,
        })

    all_sec = dur_df["duration_sec"].values
    table_rows.append({
        "Group": "Total",
        "Recordings": len(dur_df),
        "Label": "",
        "Avg len (s)": round(float(np.mean(all_sec)), 2),
        "Var (s²)": round(float(np.var(all_sec, ddof=0)), 2),
        "Total len (h)": round(float(all_sec.sum() / 3600), 2),
        "Subtype n": len(dur_df),
    })

    # Print markdown table
    print("Table 1: Dataset composition and duration summary for WAB Picnic recordings.\n")
    print("| Group | Recordings | Label | Avg len (s) | Var (s²) | Total len (h) | Subtype n |")
    print("|-------|------------|-------|--------------|----------|---------------|-----------|")
    for r in table_rows:
        label_str = r["Label"] if r["Label"] != "" else ""
        group = f"**{r['Group']}**" if r["Group"] == "Total" else r["Group"]
        rec = f"**{r['Recordings']}**" if r["Group"] == "Total" else r["Recordings"]
        subn = f"**{r['Subtype n']}**" if r["Group"] == "Total" else r["Subtype n"]
        avg = f"**{r['Avg len (s)']}**" if r["Group"] == "Total" else r["Avg len (s)"]
        var = f"**{r['Var (s²)']}**" if r["Group"] == "Total" else r["Var (s²)"]
        tot = f"**{r['Total len (h)']}**" if r["Group"] == "Total" else r["Total len (h)"]
        print(f"| {group} | {rec} | {label_str} | {avg} | {var} | {tot} | {subn} |")

    if args.output:
        out_df = pd.DataFrame(table_rows)
        out_df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
