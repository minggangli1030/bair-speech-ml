#!/usr/bin/env python3
"""
Generate publication-quality heatmaps for the Interspeech paper.
Shows proportion of participants naming each entity and CU by diagnostic group.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.graph_semantic import compute_s_sem
from src.features.phoneme_adequacy import parse_pipeline_output


def build_transcript_dict(parsed: dict) -> dict:
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


def categorize_entities(entity_labels):
    """Categorize entities into semantic categories."""
    animate = ["family", "man", "woman", "boy", "girl", "dog"]
    inanimate = ["kite", "kite string", "blanket", "picnic basket", "food", "drink", "book", "ball"]
    exterior = ["tree", "grass", "sky", "lake", "boat", "house", "path"]
    
    categories = {}
    for ent in entity_labels:
        if ent in animate:
            categories[ent] = "Animate"
        elif ent in inanimate:
            categories[ent] = "Inanimate"
        elif ent in exterior:
            categories[ent] = "Exterior"
        else:
            categories[ent] = "Other"
    return categories


def main():
    parser = argparse.ArgumentParser(description="Generate publication-quality CU heatmap")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest_ppa_full.csv"))
    parser.add_argument("--pipeline-dir", type=Path, default=Path("data/ppa_pipeline_outputs"))
    parser.add_argument("--gold-graph", type=Path, default=Path("data/graph/picnic_gold_star.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/cu_heatmap_paper.png"))
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    gold = json.loads(args.gold_graph.read_text())
    
    entity_labels = [n["label"] for n in gold["nodes"]]
    cu_info = gold["content_units"]
    
    # Build CU descriptions
    cu_descriptions = []
    for cu in cu_info:
        if cu["kind"] == "edge":
            subj_label = next((n["label"] for n in gold["nodes"] if n["id"] == cu["subj"]), cu["subj"])
            obj_label = next((n["label"] for n in gold["nodes"] if n["id"] == cu["obj"]), cu["obj"])
            desc = f"{subj_label} {cu['rel']} {obj_label}"
        else:
            node_label = next((n["label"] for n in gold["nodes"] if n["id"] == cu["node"]), cu["node"])
            desc = f"{node_label} {cu['value']}"
        cu_descriptions.append((cu["id"], desc))
    
    # Categorize entities
    entity_categories = categorize_entities(entity_labels)
    
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
            
            matched_entities = set(details.get("matched_nodes", []))
            matched_cus = set(details.get("matched_cus", []))
            
            sample_data = {
                "subject_id": row["subject_id"],
                "subtype": row["subtype"],
            }
            
            for ent in entity_labels:
                sample_data[f"ent_{ent}"] = 1 if ent in matched_entities else 0
            
            for cu_id, _ in cu_descriptions:
                sample_data[f"cu_{cu_id}"] = 1 if cu_id in matched_cus else 0
            
            results.append(sample_data)
            
        except Exception as e:
            print(f"Error processing {stem}: {e}")
            continue
    
    df = pd.DataFrame(results)
    print(f"Processed {len(df)} samples")
    print(df["subtype"].value_counts())
    
    groups = ["control", "svppa", "nfvppa", "lvppa"]
    group_labels = ["HC", "svPPA", "nfvPPA", "lvPPA"]
    
    # Select key entities for cleaner visualization
    key_entities = ["family", "man", "woman", "boy", "girl", "dog", "kite", 
                    "blanket", "picnic basket", "book", "tree", "lake", "boat", "house"]
    
    ent_cols = [f"ent_{e}" for e in key_entities]
    ent_labels = key_entities
    
    ent_matrix = []
    for g in groups:
        group_df = df[df["subtype"] == g]
        if len(group_df) == 0:
            ent_matrix.append([0] * len(ent_cols))
        else:
            props = [group_df[col].mean() for col in ent_cols]
            ent_matrix.append(props)
    
    ent_matrix = np.array(ent_matrix).T
    
    # Key actions/CUs
    key_cus = ["kite flying", "family picnicking", "dog running", "boat on lake"]
    cu_cols = [c for c in df.columns if c.startswith("cu_")]
    
    # Filter to CUs that have some variance
    cu_matrix = []
    cu_labels_filtered = []
    for cu_id, desc in cu_descriptions:
        col = f"cu_{cu_id}"
        if col in df.columns:
            props = []
            for g in groups:
                group_df = df[df["subtype"] == g]
                if len(group_df) == 0:
                    props.append(0)
                else:
                    props.append(group_df[col].mean())
            if max(props) > 0.05:  # Only include if at least one group > 5%
                cu_matrix.append(props)
                cu_labels_filtered.append(desc.title())
    
    cu_matrix = np.array(cu_matrix)
    
    # Create figure - single panel with entities
    fig, ax = plt.subplots(figsize=(6, 10))
    
    # Format entity labels
    formatted_ent_labels = [f"{i+1} - {e.title()}" for i, e in enumerate(ent_labels)]
    
    # Main heatmap
    sns.heatmap(ent_matrix, ax=ax, cmap="RdBu_r", vmin=0, vmax=1,
                xticklabels=group_labels, yticklabels=formatted_ent_labels,
                cbar_kws={'label': ''}, linewidths=0.5, linecolor='white')
    
    ax.set_title("Proportion of participants naming entity", fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel("Diagnostic Group", fontsize=10)
    ax.set_ylabel("Entity", fontsize=10)
    ax.tick_params(axis='both', labelsize=9)
    
    # Add colorbar label on the side
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=9)
    
    plt.tight_layout()
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved entity heatmap to {args.output}")
    
    # Create action/CU heatmap separately
    if len(cu_matrix) > 0:
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        
        formatted_cu_labels = [f"{i+1} - {c}" for i, c in enumerate(cu_labels_filtered)]
        
        sns.heatmap(cu_matrix, ax=ax2, cmap="RdBu_r", vmin=0, vmax=1,
                    xticklabels=group_labels, yticklabels=formatted_cu_labels,
                    annot=True, fmt='.2f', annot_kws={'size': 8},
                    cbar_kws={'label': ''}, linewidths=0.5, linecolor='white')
        
        ax2.set_title("Proportion of participants producing action CU", fontsize=11, fontweight='bold', pad=10)
        ax2.set_xlabel("Diagnostic Group", fontsize=10)
        ax2.set_ylabel("Content Unit (Action)", fontsize=10)
        ax2.tick_params(axis='both', labelsize=9)
        
        plt.tight_layout()
        
        cu_output = args.output.parent / "cu_actions_heatmap_paper.png"
        plt.savefig(cu_output, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved CU action heatmap to {cu_output}")
    
    # Create combined figure matching reference style
    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(6, 12), 
                                       gridspec_kw={'height_ratios': [14, 4]})
    
    # Top: Entity heatmap
    sns.heatmap(ent_matrix, ax=ax3a, cmap="RdBu_r", vmin=0, vmax=1,
                xticklabels=group_labels, yticklabels=formatted_ent_labels,
                cbar_kws={'label': ''}, linewidths=0.5, linecolor='white')
    ax3a.set_title("Proportion of participants naming entity", fontsize=11, fontweight='bold', pad=10)
    ax3a.set_xlabel("")
    ax3a.set_ylabel("Entity", fontsize=10)
    ax3a.tick_params(axis='both', labelsize=9)
    
    # Bottom: Action CU heatmap (summarized by category)
    # Group entities by category
    category_order = ["Animate", "Inanimate", "Exterior"]
    category_labels = ["Animate", "Inanimate", "Exterior"]
    
    cat_matrix = []
    for cat in category_order:
        cat_entities = [e for e in key_entities if entity_categories.get(e) == cat]
        if not cat_entities:
            continue
        cat_cols = [f"ent_{e}" for e in cat_entities]
        props = []
        for g in groups:
            group_df = df[df["subtype"] == g]
            if len(group_df) == 0:
                props.append(0)
            else:
                props.append(group_df[[c for c in cat_cols if c in group_df.columns]].mean().mean())
        cat_matrix.append(props)
    
    # Add action CUs row
    action_cus = ["CU_A01", "CU_A02", "CU_A03"]  # kite flying, dog running, family picnicking
    action_props = []
    for g in groups:
        group_df = df[df["subtype"] == g]
        if len(group_df) == 0:
            action_props.append(0)
        else:
            action_cols = [f"cu_{cu}" for cu in action_cus if f"cu_{cu}" in group_df.columns]
            if action_cols:
                action_props.append(group_df[action_cols].mean().mean())
            else:
                action_props.append(0)
    cat_matrix.append(action_props)
    
    cat_matrix = np.array(cat_matrix)
    cat_labels = category_labels + ["Actions"]
    
    sns.heatmap(cat_matrix, ax=ax3b, cmap="RdBu_r", vmin=0, vmax=1,
                xticklabels=group_labels, yticklabels=cat_labels,
                annot=True, fmt='.2f', annot_kws={'size': 9},
                cbar_kws={'label': ''}, linewidths=0.5, linecolor='white')
    ax3b.set_title("Mean proportion per category", fontsize=11, fontweight='bold', pad=10)
    ax3b.set_xlabel("Diagnostic Group", fontsize=10)
    ax3b.set_ylabel("Category", fontsize=10)
    ax3b.tick_params(axis='both', labelsize=9)
    
    plt.tight_layout()
    
    combined_output = args.output.parent / "cu_combined_heatmap_paper.png"
    plt.savefig(combined_output, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved combined heatmap to {combined_output}")
    
    # Print summary statistics for the paper
    print("\n" + "="*60)
    print("SUMMARY STATISTICS FOR PAPER")
    print("="*60)
    
    print("\n--- Entity mention rates by group (key entities) ---")
    for i, ent in enumerate(ent_labels):
        row = ent_matrix[i]
        # Mark significant differences (HC vs svPPA)
        diff = row[0] - row[1]  # HC - svPPA
        marker = "**" if diff > 0.3 else ("*" if diff > 0.15 else "")
        print(f"{ent:18s}: HC={row[0]:.2f}  sv={row[1]:.2f}  nfv={row[2]:.2f}  lv={row[3]:.2f}  {marker}")
    
    print("\n--- Category mean rates ---")
    for i, cat in enumerate(cat_labels):
        row = cat_matrix[i]
        diff = row[0] - row[1]
        marker = "**" if diff > 0.3 else ("*" if diff > 0.15 else "")
        print(f"{cat:12s}: HC={row[0]:.2f}  sv={row[1]:.2f}  nfv={row[2]:.2f}  lv={row[3]:.2f}  {marker}")


if __name__ == "__main__":
    main()
