#!/usr/bin/env python3
"""
Generate a heatmap showing proportion of participants naming each Content Unit (CU)
by diagnostic group, similar to the Cookie Theft CIU analysis.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.graph_semantic import compute_s_sem
from src.features.phoneme_adequacy import parse_pipeline_output


def build_transcript_dict(parsed: dict) -> dict:
    """Build transcript dict from parsed pipeline output."""
    segments = parsed.get("segments", [])
    words_list = []
    full_text_parts = []

    for seg in segments:
        if "Participant" not in seg.get("speaker", ""):
            continue
        transcript = (seg.get("transcript") or "").strip()
        if transcript.lower().startswith("s ") and len(transcript) > 2:
            transcript = transcript[2:]
        full_text_parts.append(transcript)
        words = transcript.split()
        for w in words:
            words_list.append({"word": w, "start": 0.0, "end": 0.1})

    text = " ".join(full_text_parts)
    return {"text": text, "words": words_list}


def main():
    parser = argparse.ArgumentParser(description="Generate CU heatmap")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest_ppa_full.csv"))
    parser.add_argument("--pipeline-dir", type=Path, default=Path("data/ppa_pipeline_outputs"))
    parser.add_argument("--gold-graph", type=Path, default=Path("data/graph/picnic_gold_star.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/cu_heatmap.png"))
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    gold = json.loads(args.gold_graph.read_text())
    
    # Get entity labels and CU info
    entity_labels = [n["label"] for n in gold["nodes"]]
    cu_info = gold["content_units"]
    
    # Build CU descriptions
    cu_descriptions = []
    for cu in cu_info:
        if cu["kind"] == "edge":
            # Find node labels
            subj_label = next((n["label"] for n in gold["nodes"] if n["id"] == cu["subj"]), cu["subj"])
            obj_label = next((n["label"] for n in gold["nodes"] if n["id"] == cu["obj"]), cu["obj"])
            desc = f"{subj_label} {cu['rel']} {obj_label}"
        else:
            node_label = next((n["label"] for n in gold["nodes"] if n["id"] == cu["node"]), cu["node"])
            desc = f"{node_label} {cu['value']}"
        cu_descriptions.append((cu["id"], desc))
    
    # Collect data per sample
    results = []
    
    for _, row in manifest.iterrows():
        audio_path = Path(row["audio_path"])
        stem = audio_path.stem
        pipeline_file = args.pipeline_dir / f"{stem}_feature_extraction.txt"
        
        if not pipeline_file.exists():
            continue
            
        try:
            parsed = parse_pipeline_output(pipeline_file)
            transcript_dict = build_transcript_dict(parsed)
            
            if not transcript_dict["text"].strip():
                continue
                
            s_sem, details = compute_s_sem(transcript_dict, args.gold_graph)
            
            # Get matched entities and CUs
            matched_entities = set(details.get("matched_nodes", []))
            matched_cus = set(details.get("matched_cus", []))
            
            # Record for this sample
            sample_data = {
                "subject_id": row["subject_id"],
                "subtype": row["subtype"],
            }
            
            # Entity mentions
            for ent in entity_labels:
                sample_data[f"ent_{ent}"] = 1 if ent in matched_entities else 0
            
            # CU mentions
            for cu_id, _ in cu_descriptions:
                sample_data[f"cu_{cu_id}"] = 1 if cu_id in matched_cus else 0
            
            results.append(sample_data)
            
        except Exception as e:
            print(f"Error processing {stem}: {e}")
            continue
    
    df = pd.DataFrame(results)
    print(f"Processed {len(df)} samples")
    print(df["subtype"].value_counts())
    
    # Calculate proportions by group
    groups = ["control", "svppa", "nfvppa", "lvppa"]
    group_labels = ["HC", "svPPA", "nfvPPA", "lvPPA"]
    
    # Entity proportions
    ent_cols = [c for c in df.columns if c.startswith("ent_")]
    ent_labels = [c.replace("ent_", "") for c in ent_cols]
    
    ent_matrix = []
    for g in groups:
        group_df = df[df["subtype"] == g]
        if len(group_df) == 0:
            ent_matrix.append([0] * len(ent_cols))
        else:
            props = group_df[ent_cols].mean().values
            ent_matrix.append(props)
    
    ent_matrix = np.array(ent_matrix).T  # entities x groups
    
    # CU proportions
    cu_cols = [c for c in df.columns if c.startswith("cu_")]
    cu_labels = [desc for cu_id, desc in cu_descriptions]
    
    cu_matrix = []
    for g in groups:
        group_df = df[df["subtype"] == g]
        if len(group_df) == 0:
            cu_matrix.append([0] * len(cu_cols))
        else:
            props = group_df[cu_cols].mean().values
            cu_matrix.append(props)
    
    cu_matrix = np.array(cu_matrix).T  # CUs x groups
    
    # Create figure with two heatmaps
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 14), 
                                     gridspec_kw={'height_ratios': [len(ent_labels), len(cu_labels)]})
    
    # Entity heatmap
    sns.heatmap(ent_matrix, ax=ax1, cmap="RdBu_r", vmin=0, vmax=1,
                xticklabels=group_labels, yticklabels=ent_labels,
                cbar_kws={'label': 'Proportion'})
    ax1.set_title("Proportion of participants mentioning each entity", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Diagnostic Group")
    ax1.set_ylabel("Entity")
    
    # CU heatmap
    sns.heatmap(cu_matrix, ax=ax2, cmap="RdBu_r", vmin=0, vmax=1,
                xticklabels=group_labels, yticklabels=cu_labels,
                cbar_kws={'label': 'Proportion'})
    ax2.set_title("Proportion of participants producing each Content Unit", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Diagnostic Group")
    ax2.set_ylabel("Content Unit")
    
    plt.tight_layout()
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to {args.output}")
    
    # Also save a simplified version with just CUs
    fig2, ax = plt.subplots(figsize=(6, 8))
    sns.heatmap(cu_matrix, ax=ax, cmap="RdBu_r", vmin=0, vmax=1,
                xticklabels=group_labels, yticklabels=cu_labels,
                annot=True, fmt='.2f', cbar_kws={'label': 'Proportion'})
    ax.set_title("Content Unit Coverage by Diagnostic Group", fontsize=12, fontweight='bold')
    ax.set_xlabel("Diagnostic Group")
    ax.set_ylabel("Content Unit")
    plt.tight_layout()
    
    cu_output = args.output.parent / "cu_heatmap_simple.png"
    plt.savefig(cu_output, dpi=300, bbox_inches='tight')
    print(f"Saved simplified heatmap to {cu_output}")
    
    # Print summary statistics
    print("\n--- Entity mention rates by group ---")
    for i, ent in enumerate(ent_labels):
        row = ent_matrix[i]
        print(f"{ent:20s}: HC={row[0]:.2f}  sv={row[1]:.2f}  nfv={row[2]:.2f}  lv={row[3]:.2f}")
    
    print("\n--- CU coverage rates by group ---")
    for i, cu in enumerate(cu_labels):
        row = cu_matrix[i]
        print(f"{cu:30s}: HC={row[0]:.2f}  sv={row[1]:.2f}  nfv={row[2]:.2f}  lv={row[3]:.2f}")


if __name__ == "__main__":
    main()
