#!/usr/bin/env python3
"""
Diagnose and fix Layer 2 adequacy computation.

The bug: _zscore_vector was z-scoring each sample against ITSELF (its own 20 L1 features),
not against the population. This collapses all Layer 2 scores to ~0.65 with tiny variance.

Fix: compute population mean/std from the full dataset, pass into compute_layer2_features.
"""
import re
import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.layer1 import compute_layer1_features, LAYER1_NAMES
from src.features.layer2 import compute_layer2_features, LAYER2_NAMES
from src.features.phoneme_adequacy import parse_pipeline_output
from src.features.dissociation import mann_whitney_u, cliffs_delta

LAYER2_SEMANTIC_IMPAIRMENT = ["anomia", "empty_speech", "semantic_paraphasias"]

logging.basicConfig(level=logging.WARNING)


def _parse_timestamp(s: str) -> float:
    """Parse 'MM:SS.mmm' to seconds."""
    parts = str(s).strip().split(":")
    if len(parts) != 2:
        return 0.0
    try:
        mins = float(parts[0])
        secs = float(parts[1])
        return mins * 60 + secs
    except ValueError:
        return 0.0


def build_transcript_dict(parsed: dict) -> dict:
    """Build transcript dict with word-level timestamps from pipeline segments."""
    segments = parsed.get("segments", [])
    words_list = []
    full_text_parts = []

    for seg in segments:
        if "Participant" not in seg.get("speaker", ""):
            continue
        try:
            start_sec = _parse_timestamp(seg.get("start", "0:0.0"))
            end_sec = _parse_timestamp(seg.get("end", "0:0.0"))
        except Exception:
            start_sec, end_sec = 0.0, 0.0
        transcript = (seg.get("transcript") or "").strip()
        if transcript.lower().startswith("s ") and len(transcript) > 2:
            transcript = transcript[2:].strip()
        if not transcript:
            continue
        tok = re.findall(r"[a-zA-Z]+", transcript)
        if not tok:
            continue
        full_text_parts.extend(tok)
        dur = max(end_sec - start_sec, 0.001)
        step = dur / len(tok)
        for i, w in enumerate(tok):
            ws = start_sec + i * step
            we = start_sec + (i + 1) * step
            words_list.append({"word": w, "start": ws, "end": we})

    text = " ".join(full_text_parts)
    return {"text": text, "words": words_list}


def load_pipeline_outputs(pipeline_dir: Path, manifest_path: Path) -> pd.DataFrame:
    """Load all available pipeline outputs and extract Layer 1 features."""
    df = pd.read_csv(manifest_path)
    records = []

    for _, row in df.iterrows():
        audio_rel = row["audio_path"]
        stem = Path(audio_rel).stem
        cand = pipeline_dir / f"{stem}_feature_extraction.txt"
        if not cand.exists():
            continue

        parsed = parse_pipeline_output(cand)
        if not parsed or not parsed.get("transcript"):
            continue

        transcript_dict = build_transcript_dict(parsed)
        if not transcript_dict["text"]:
            continue

        try:
            l1 = compute_layer1_features(
                transcript_dict,
                phonetic_data=None,
                expected_entities=None,
                use_semantic_distance=False,
            )
            records.append({
                "subject_id": row["subject_id"],
                "session": row.get("session", ""),
                "subtype": row.get("subtype", ""),
                "label": row.get("label"),
                **l1,
            })
        except Exception as e:
            print(f"  Warning: L1 failed for {row['subject_id']}: {e}")

    return pd.DataFrame(records)


def compute_population_normed_layer2(l1_df: pd.DataFrame):
    """
    Compute Layer 2 with population-level z-scoring.
    Uses the POPULATION mean/std across all samples, not within-sample.
    """
    l1_cols = [c for c in LAYER1_NAMES if c in l1_df.columns]
    X = l1_df[l1_cols].values.astype(float)
    with np.errstate(invalid="ignore"):
        pop_mean = np.nanmean(X, axis=0)
        pop_std = np.nanstd(X, axis=0)
    pop_std[pop_std == 0] = 1.0

    print(f"Population normalization computed from {len(l1_df)} samples")
    print(f"L1 features used: {len(l1_cols)}/{len(LAYER1_NAMES)}")

    results = []
    for i, row in l1_df.iterrows():
        l1_dict = {name: row.get(name, np.nan) for name in LAYER1_NAMES}

        raw_z, scores = compute_layer2_features(
            l1_dict,
            population_mean=pop_mean,
            population_std=pop_std,
        )

        sem_vals = [
            scores[n] for n in LAYER2_SEMANTIC_IMPAIRMENT
            if n in scores and not np.isnan(scores.get(n, np.nan))
        ]
        sem_impairment = float(np.mean(sem_vals)) if sem_vals else np.nan
        sem_adequacy = 1.0 - sem_impairment if not np.isnan(sem_impairment) else np.nan

        results.append({
            "subject_id": row["subject_id"],
            "session": row.get("session", ""),
            "subtype": row.get("subtype", ""),
            "label": row.get("label"),
            **{f"l2_{k}": v for k, v in scores.items()},
            "s_sem_layer2_adequacy_fixed": sem_adequacy,
            "l2_sem_impairment": sem_impairment,
        })

    return pd.DataFrame(results), pop_mean, pop_std


def validate_fix(results_df: pd.DataFrame) -> None:
    """Validate fixed Layer 2 shows discriminative signal across subtypes."""
    col = "s_sem_layer2_adequacy_fixed"
    print("\n" + "=" * 60)
    print("LAYER 2 ADEQUACY — FIXED (population-normed)")
    print("=" * 60)

    print(f"\nOverall range: {results_df[col].min():.3f} – {results_df[col].max():.3f}")
    print(f"Overall std: {results_df[col].std():.3f}")
    q75, q25 = results_df[col].quantile(0.75), results_df[col].quantile(0.25)
    print(f"IQR: {q75 - q25:.3f}")

    print("\nBy subtype:")
    for s in ["control", "svppa", "lvppa", "nfvppa"]:
        sub = results_df[results_df["subtype"] == s][col].dropna()
        if len(sub) > 0:
            print(f"  {s:8s}: mean={sub.mean():.3f}  std={sub.std():.3f}  n={len(sub)}")

    print("\nPairwise Mann-Whitney on fixed Layer 2 adequacy:")
    subtypes = ["control", "svppa", "lvppa", "nfvppa"]
    for i in range(len(subtypes)):
        for j in range(i + 1, len(subtypes)):
            s1, s2 = subtypes[i], subtypes[j]
            v1 = results_df[results_df["subtype"] == s1][col].dropna().values
            v2 = results_df[results_df["subtype"] == s2][col].dropna().values
            if len(v1) >= 3 and len(v2) >= 3:
                u, p = mann_whitney_u(v1, v2)
                cd = cliffs_delta(v1, v2)
                print(f"  {s1} vs {s2}: p={p:.4f}  cliffs_d={cd:.3f}")

    print("\nIndividual L2 construct means by subtype (control vs svPPA):")
    l2_cols = [c for c in results_df.columns if c.startswith("l2_") and c != "l2_sem_impairment"]
    for l2col in l2_cols:
        construct = l2col.replace("l2_", "")
        vals = {}
        for s in ["control", "svppa"]:
            sub = results_df[results_df["subtype"] == s][l2col].dropna()
            if len(sub) > 0:
                vals[s] = sub.mean()
        if vals:
            ctrl = vals.get("control", float("nan"))
            svppa = vals.get("svppa", float("nan"))
            diff = svppa - ctrl if not (np.isnan(ctrl) or np.isnan(svppa)) else 0
            direction = "svPPA higher" if diff > 0.05 else ("control higher" if diff < -0.05 else "~equal")
            print(f"  {construct:35s}: ctrl={ctrl:.3f}  svppa={svppa:.3f}  ({direction})")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest_ppa_full.csv"))
    parser.add_argument("--pipeline-dir", type=Path, default=Path("data/ppa_pipeline_outputs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/layer2_fixed.csv"))
    parser.add_argument("--pop-stats-output", type=Path, default=Path("outputs/population_l1_stats.json"))
    args = parser.parse_args()

    print("Step 1: Loading pipeline outputs and extracting Layer 1 features...")
    l1_df = load_pipeline_outputs(args.pipeline_dir, args.manifest)
    print(f"  Got L1 features for {len(l1_df)} samples")

    if len(l1_df) == 0:
        print("ERROR: No samples loaded. Check --pipeline-dir path.")
        sys.exit(1)

    print("\nStep 2: Computing population-normed Layer 2...")
    results_df, pop_mean, pop_std = compute_population_normed_layer2(l1_df)

    print("\nStep 3: Validating fix...")
    validate_fix(results_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output, index=False)
    print(f"\nSaved fixed Layer 2 features to {args.output}")

    pop_stats = {
        "feature_names": LAYER1_NAMES,
        "population_mean": pop_mean.tolist(),
        "population_std": pop_std.tolist(),
        "n_samples": len(l1_df),
    }
    with open(args.pop_stats_output, "w") as f:
        json.dump(pop_stats, f, indent=2)
    print(f"Saved population L1 stats to {args.pop_stats_output}")

    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Merge layer2_fixed.csv into ppa_dissociation_all.csv (s_sem_layer2_adequacy_fixed)")
    print("2. Re-run run_full_analysis.py with population-normed Layer 2")
    print("3. Re-run ablation alpha sweep and classifier with corrected Layer 2")
    print("If the fix works, Layer 2 should show:")
    print("  - svPPA: higher anomia, empty_speech, semantic_paraphasias than controls")
    print("  - nfvPPA: higher reduced_speech_rate, agrammatism")
    print("  - Controls: lowest impairment across all constructs")


if __name__ == "__main__":
    main()
