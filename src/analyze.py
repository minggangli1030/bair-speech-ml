"""
Phonetic-Semantic Dissociation Analysis for PPA.

This module implements the full analysis pipeline:
1. Parse pipeline outputs (ASR + SylWav phonemes)
2. Compute S_ph (phonological adequacy)
3. Compute S_sem (semantic adequacy from scene graph)
4. Compute Δ = S_ph - S_sem (dissociation index)
5. Run statistical validation (Mann-Whitney U, Cliff's δ)
6. Generate classification features

Key differentiation from Peters et al. 2025:
- Explicit dissociation index grounded in dual-stream theory
- Fully automated (no manual diarization)
- Interpretable per-subject profiles
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from .features.phoneme_adequacy import compute_s_ph_from_file, parse_pipeline_output, compute_s_ph
from .features.graph_semantic import compute_s_sem
from .features.dissociation import (
    compute_dissociation,
    DissociationResult,
    validate_dissociation_hypothesis,
    compute_group_statistics,
)

logger = logging.getLogger(__name__)


def extract_dissociation_features(
    manifest_path: Path,
    data_dir: Path,
    gold_graph_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Extract phonetic-semantic dissociation features from manifest.
    
    For each sample:
    - Parse pipeline output (if available) for ASR transcript and SylWav phonemes
    - Compute S_ph from phoneme sequences
    - Compute S_sem from transcript using scene graph
    - Compute Δ = S_ph - S_sem
    
    Returns DataFrame with dissociation features per sample.
    """
    df = pd.read_csv(manifest_path)
    
    results = []
    
    for _, row in df.iterrows():
        subject_id = row["subject_id"]
        session = row.get("session", "")
        label = row.get("label", None)
        subtype = row.get("subtype", "")
        
        pipeline_output = row.get("pipeline_output", "")
        
        s_ph = float("nan")
        s_sem = float("nan")
        s_ph_details = {}
        s_sem_details = {}
        transcript = ""
        
        # Try to get data from pipeline output
        if pd.notna(pipeline_output) and pipeline_output:
            pipeline_path = data_dir / pipeline_output
            if pipeline_path.exists():
                parsed = parse_pipeline_output(pipeline_path)
                if parsed:
                    transcript = parsed.get("transcript", "")
                    phonemes = parsed.get("phonemes", "")
                    
                    if transcript and phonemes:
                        s_ph, s_ph_details = compute_s_ph(transcript, phonemes)
        
        # Compute S_sem if we have a transcript
        if transcript:
            # Create transcript dict for compute_s_sem
            transcript_dict = {
                "text": transcript,
                "words": [{"word": w, "start": i*0.5, "end": i*0.5+0.3} 
                         for i, w in enumerate(transcript.split())],
            }
            s_sem, s_sem_details = compute_s_sem(transcript_dict, gold_graph_path=gold_graph_path)
        
        # Compute dissociation
        dissoc = compute_dissociation(s_ph, s_sem, s_ph_details, s_sem_details)
        
        result = {
            "subject_id": subject_id,
            "session": session,
            "label": label,
            "subtype": subtype,
            "transcript": transcript[:500] if transcript else "",
            **dissoc.to_dict(),
        }
        results.append(result)
        
        logger.info(
            "Processed %s %s: S_ph=%.3f, S_sem=%.3f, Δ=%.3f",
            subject_id, session,
            s_ph if not np.isnan(s_ph) else -1,
            s_sem if not np.isnan(s_sem) else -1,
            dissoc.delta if not np.isnan(dissoc.delta) else -1,
        )
    
    out_df = pd.DataFrame(results)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_path, index=False)
        logger.info("Wrote dissociation features to %s", output_path)
    
    return out_df


def run_statistical_validation(
    features_df: pd.DataFrame,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """
    Run statistical validation of dissociation hypothesis.
    
    Tests whether Δ separates svPPA from controls.
    """
    # Filter to samples with valid dissociation scores
    valid = features_df[features_df["delta"].notna()]
    
    if len(valid) == 0:
        logger.warning("No valid dissociation scores to analyze")
        return {"error": "no_valid_scores"}
    
    # Run hypothesis validation
    label_map = {0: "control", 1: "svppa", 2: "lvppa", 3: "nfvppa"}
    
    results = validate_dissociation_hypothesis(
        valid,
        s_ph_column="s_ph",
        s_sem_column="s_sem",
        delta_column="delta",
        label_column="label",
        label_map=label_map,
    )
    
    # Add feature statistics for all key features
    for feat in ["s_ph", "s_sem", "delta", "phoneme_error_rate", "silence_ratio"]:
        if feat in valid.columns and feat not in results.get("feature_statistics", {}):
            results.setdefault("feature_statistics", {})[feat] = compute_group_statistics(
                valid, feat, "label"
            )
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Wrote statistical validation to %s", output_path)
    
    return results


def generate_classification_features(
    features_df: pd.DataFrame,
    include_layer1: bool = False,
    include_layer2: bool = True,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Generate feature matrix for classification.
    
    Core features (always included):
    - s_ph: phonological adequacy
    - s_sem: semantic adequacy  
    - delta: dissociation index
    
    Additional features from s_ph computation:
    - phoneme_error_rate
    - silence_ratio
    - phonemes_per_word
    
    Returns (X, feature_names, y).
    """
    core_features = ["s_ph", "s_sem", "delta", "phoneme_error_rate", "silence_ratio", "phonemes_per_word"]
    
    available = [f for f in core_features if f in features_df.columns]
    
    X = features_df[available].to_numpy(dtype=float)
    y = features_df["label"].to_numpy(dtype=float)
    
    # Impute NaN with column median
    for j in range(X.shape[1]):
        col = X[:, j]
        median = np.nanmedian(col)
        X[np.isnan(X[:, j]), j] = median if not np.isnan(median) else 0.0
    
    return X, available, y


def print_summary(features_df: pd.DataFrame) -> None:
    """Print summary statistics."""
    print("\n" + "="*60)
    print("PHONETIC-SEMANTIC DISSOCIATION ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\nTotal samples: {len(features_df)}")
    
    print("\nSamples by subtype:")
    for subtype in features_df["subtype"].unique():
        n = len(features_df[features_df["subtype"] == subtype])
        valid_delta = features_df[(features_df["subtype"] == subtype) & features_df["delta"].notna()]
        print(f"  {subtype}: {n} samples ({len(valid_delta)} with Δ)")
    
    # Statistics for samples with valid scores
    valid = features_df[features_df["delta"].notna()]
    
    if len(valid) > 0:
        print(f"\nSamples with valid Δ: {len(valid)}")
        print("\nFeature statistics:")
        for feat in ["s_ph", "s_sem", "delta"]:
            if feat in valid.columns:
                vals = valid[feat].dropna()
                print(f"  {feat}: mean={vals.mean():.3f}, std={vals.std():.3f}, "
                      f"min={vals.min():.3f}, max={vals.max():.3f}")
        
        print("\nBy subtype (mean ± std):")
        for subtype in valid["subtype"].unique():
            sub_df = valid[valid["subtype"] == subtype]
            print(f"\n  {subtype}:")
            for feat in ["s_ph", "s_sem", "delta"]:
                if feat in sub_df.columns:
                    vals = sub_df[feat].dropna()
                    if len(vals) > 0:
                        print(f"    {feat}: {vals.mean():.3f} ± {vals.std():.3f}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    parser = argparse.ArgumentParser(description="PPA Dissociation Analysis")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest_ppa_full.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("/data/sarikapasumarthy/data/PPA/enhanced"))
    parser.add_argument("--gold-graph", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/dissociation_features.csv"))
    parser.add_argument("--stats-output", type=Path, default=Path("outputs/dissociation_stats.json"))
    args = parser.parse_args()
    
    # Extract features
    df = extract_dissociation_features(
        args.manifest,
        args.data_dir,
        gold_graph_path=args.gold_graph,
        output_path=args.output,
    )
    
    # Print summary
    print_summary(df)
    
    # Run statistical validation
    stats = run_statistical_validation(df, output_path=args.stats_output)
    
    if "hypothesis_test" in stats:
        print("\nHypothesis Test Results:")
        ht = stats["hypothesis_test"]
        print(f"  Comparison: {ht['comparison']}")
        print(f"  svPPA Δ mean: {ht['svppa_mean']:.3f}")
        print(f"  Control Δ mean: {ht['control_mean']:.3f}")
        print(f"  Mann-Whitney U: {ht['mann_whitney_u']:.2f}")
        print(f"  p-value: {ht['p_value']:.4f}")
        print(f"  Cliff's δ: {ht['cliffs_delta']:.3f} ({ht['effect_size']})")
        print(f"  Hypothesis supported: {ht['hypothesis_supported']}")
