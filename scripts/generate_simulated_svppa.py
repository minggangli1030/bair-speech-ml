#!/usr/bin/env python3
"""
Generate simulated svPPA samples by degrading nfvPPA/lvPPA transcripts.

For each nfv/lv pipeline output: degrade transcript -> recompute S_sem (graph + Layer2 with pop norm).
Keep S_ph from the original. Output rows with label=1 (simulated svPPA) for classifier augmentation.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.phoneme_adequacy import parse_pipeline_output
from src.features.graph_semantic import compute_s_sem
from src.features.layer1 import compute_layer1_features, LAYER1_NAMES
from src.features.layer2 import compute_layer2_features, LAYER2_NAMES
from src.features.simulate_svppa import degrade_transcript, transcript_dict_from_text

# Use ALL 10 Layer 2 constructs for adequacy
LAYER2_ALL_CONSTRUCTS = [
    "reduced_speech_rate",
    "reduced_length_complexity",
    "agrammatism",
    "paragrammatism",
    "anomia",
    "empty_speech",
    "semantic_paraphasias",
    "phonemic_paraphasias_neologisms",
    "self_correction",
    "overall_communication_impairment",
]


def _parse_timestamp(s: str) -> float:
    parts = str(s).strip().split(":")
    if len(parts) != 2:
        return 0.0
    try:
        return float(parts[0]) * 60 + float(parts[1])
    except ValueError:
        return 0.0


def build_transcript_dict(parsed: dict) -> dict:
    """Build transcript dict from pipeline parsed output."""
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate simulated svPPA from nfv/lv transcripts")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest_ppa_full.csv"))
    parser.add_argument("--pipeline-dir", type=Path, default=Path("data/ppa_pipeline_outputs"))
    parser.add_argument("--features", type=Path, default=Path("outputs/ppa_dissociation_fixed_l2.csv"),
                        help="CSV with s_ph, s_sem (for merging S_ph)")
    parser.add_argument("--pop-stats", type=Path, default=Path("outputs/population_l1_stats.json"))
    parser.add_argument("--gold-graph", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--replace-ratio", type=float, default=0.7)
    parser.add_argument("--output", type=Path, default=Path("outputs/simulated_svppa.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load population stats for Layer 2
    if not args.pop_stats.exists():
        print(f"ERROR: {args.pop_stats} not found. Run fix_layer2_population_norm.py first.")
        sys.exit(1)
    with open(args.pop_stats) as f:
        pop_stats = json.load(f)
    pop_mean = np.array(pop_stats["population_mean"])
    pop_std = np.array(pop_stats["population_std"])
    if len(pop_std) < len(LAYER1_NAMES):
        pop_std = np.resize(pop_std, len(LAYER1_NAMES))
        pop_std[pop_std == 0] = 1.0

    # Load features for S_ph lookup
    feat_df = pd.read_csv(args.features)
    feat_df = feat_df.set_index(["subject_id", "session"])

    manifest = pd.read_csv(args.manifest)
    # Only nfvPPA and lvPPA
    nflf = manifest[manifest["subtype"].isin(["nfvppa", "lvppa"])]

    gold_path = args.gold_graph
    if gold_path is None:
        gold_path = Path(__file__).resolve().parent.parent / "data" / "graph" / "picnic_gold_star.json"
    if not gold_path.exists():
        gold_path = None

    records = []
    for _, row in nflf.iterrows():
        subject_id = row["subject_id"]
        session = row.get("session", "")
        stem = Path(row["audio_path"]).stem
        cand = args.pipeline_dir / f"{stem}_feature_extraction.txt"
        if not cand.exists():
            continue
        parsed = parse_pipeline_output(cand)
        if not parsed or not parsed.get("transcript"):
            continue
        orig_dict = build_transcript_dict(parsed)
        if not orig_dict["text"]:
            continue

        # Degrade transcript (simulate svPPA)
        degraded_text = degrade_transcript(
            orig_dict["text"],
            replace_ratio=args.replace_ratio,
            insert_filler_ratio=0.12,
            seed=args.seed + hash((subject_id, session)) % 10000,
        )
        deg_dict = transcript_dict_from_text(degraded_text)
        if not deg_dict["text"]:
            continue

        try:
            s_sem_graph, _ = compute_s_sem(deg_dict, gold_graph_path=gold_path)
        except Exception:
            s_sem_graph = float("nan")

        try:
            l1 = compute_layer1_features(
                deg_dict,
                phonetic_data=None,
                expected_entities=None,
                use_semantic_distance=False,
            )
            _, l2_scores = compute_layer2_features(
                l1,
                population_mean=pop_mean,
                population_std=pop_std,
            )
            vals = [l2_scores.get(n, np.nan) for n in LAYER2_ALL_CONSTRUCTS if n in l2_scores]
            layer2_adequacy = float(1.0 - np.nanmean(vals)) if vals else float("nan")
        except Exception:
            layer2_adequacy = float("nan")

        s_sem = args.alpha * (s_sem_graph if not np.isnan(s_sem_graph) else 0) + (1 - args.alpha) * (layer2_adequacy if not np.isnan(layer2_adequacy) else 0)
        if np.isnan(s_sem_graph) and np.isnan(layer2_adequacy):
            s_sem = float("nan")

        # S_ph from original (same subject/session)
        key = (subject_id, session)
        if key not in feat_df.index:
            continue
        s_ph = feat_df.loc[key, "s_ph"]
        if pd.isna(s_ph):
            continue
        delta = float(s_ph) - s_sem if not np.isnan(s_sem) else float("nan")

        records.append({
            "subject_id": subject_id,
            "session": session,
            "subtype": "simulated_svppa",
            "label": 1,
            "simulated_svppa": True,
            "s_ph": float(s_ph),
            "s_sem": float(s_sem) if not np.isnan(s_sem) else None,
            "s_sem_graph": float(s_sem_graph) if not np.isnan(s_sem_graph) else None,
            "s_sem_layer2_adequacy": layer2_adequacy if not np.isnan(layer2_adequacy) else None,
            "delta": delta,
            "phoneme_error_rate": feat_df.loc[key, "phoneme_error_rate"] if "phoneme_error_rate" in feat_df.columns else None,
            "silence_ratio": feat_df.loc[key, "silence_ratio"] if "silence_ratio" in feat_df.columns else None,
        })

    out_df = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Generated {len(out_df)} simulated svPPA samples -> {args.output}")
    if len(out_df) > 0:
        print(f"  s_sem range: {out_df['s_sem'].min():.3f} - {out_df['s_sem'].max():.3f}, mean={out_df['s_sem'].mean():.3f}")
        print(f"  s_ph mean={out_df['s_ph'].mean():.3f}")


if __name__ == "__main__":
    main()
