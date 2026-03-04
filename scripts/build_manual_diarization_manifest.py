#!/usr/bin/env python3
"""
Build a PPA manifest from the Manual Diarization folder.

Scans data/Manual Diarization for .wav files, parses subject_id and session from
filenames (sub-XXX_ses-YYYYMMDD_...), and infers label/subtype from folder name.
Output: data/manifest_manual_diarization.csv (same columns as manifest_ppa_full.csv).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

def main():
    repo_root = Path(__file__).resolve().parent.parent
    manual_dir = repo_root / "data" / "Manual Diarization"
    if not manual_dir.exists():
        print(f"Error: {manual_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    # Folder name -> (subtype, label). Match manifest_ppa_full: 0=control, 1=svppa, 2=lvppa, 3=nfvppa
    FOLDER_TO_SUBTYPE = {
        "picnics-controls-enhanced copy": ("control", 0),
        "picnics-svppa-enhanced copy": ("svppa", 1),
        "picnics-lvppa-enhanced copy": ("lvppa", 2),
        "picnics-nfvppa-enhanced copy": ("nfvppa", 3),
    }

    # sub-10683_ses-20181019_task-picnic_lang-en_final.wav
    pattern = re.compile(r"sub-(\d+)_ses-(\d+)_task-picnic_lang-en_final\.wav", re.I)

    rows = []
    for subdir in sorted(manual_dir.iterdir()):
        if not subdir.is_dir():
            continue
        folder_name = subdir.name
        if folder_name not in FOLDER_TO_SUBTYPE:
            continue
        subtype, label = FOLDER_TO_SUBTYPE[folder_name]
        for wav in subdir.glob("*.wav"):
            m = pattern.match(wav.name)
            if not m:
                continue
            subject_id = f"sub-{m.group(1)}"
            session = f"ses-{m.group(2)}"
            # audio_path relative to data/Manual Diarization (so data_dir can be Manual Diarization)
            rel = f"{folder_name}/{wav.name}"
            rows.append({
                "subject_id": subject_id,
                "session": session,
                "audio_path": rel,
                "pipeline_output": "",  # filled by pipeline
                "label": label,
                "subtype": subtype,
                "task": "picnic",
            })

    df = pd.DataFrame(rows)
    out_path = repo_root / "data" / "manifest_manual_diarization.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print("By subtype:", df["subtype"].value_counts().to_dict())


if __name__ == "__main__":
    main()
