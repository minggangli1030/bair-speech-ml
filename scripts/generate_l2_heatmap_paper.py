#!/usr/bin/env python3
"""
Generate publication-quality heatmap showing Layer 2 clinical construct scores
by diagnostic group. This demonstrates the interpretability of the pipeline.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def main():
    parser = argparse.ArgumentParser(description="Generate L2 clinical construct heatmap")
    parser.add_argument("--features", type=Path, default=Path("outputs/ppa_dissociation_full.csv"))
    parser.add_argument("--l2-features", type=Path, default=Path("outputs/layer2_fixed.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/l2_heatmap_paper.png"))
    args = parser.parse_args()

    df_main = pd.read_csv(args.features)
    df_l2 = pd.read_csv(args.l2_features)
    
    # Merge on subject_id and session
    df = df_main.merge(df_l2, on=["subject_id", "session", "subtype", "label"], how="inner")
    
    print(f"Loaded {len(df)} samples (after merging)")
    print(df["subtype"].value_counts())
    
    # Layer 2 constructs (clinical features)
    l2_constructs = [
        ("l2_anomia", "Anomia"),
        ("l2_empty_speech", "Empty Speech"),
        ("l2_semantic_paraphasias", "Semantic Paraphasias"),
        ("l2_agrammatism", "Agrammatism"),
        ("l2_paragrammatism", "Paragrammatism"),
        ("l2_reduced_speech_rate", "Reduced Speech Rate"),
        ("l2_reduced_length_complexity", "Reduced Length/Complexity"),
        ("l2_phonemic_paraphasias_neologisms", "Phonemic Paraphasias"),
        ("l2_self_correction", "Self-Correction"),
        ("l2_overall_communication_impairment", "Overall Impairment"),
    ]
    
    # Core model features
    core_features = [
        ("s_ph", "S_ph (Phonological)"),
        ("s_sem", "S_sem (Semantic)"),
        ("delta", "Δ (Dissociation)"),
        ("s_sem_graph", "S_sem Graph"),
    ]
    
    groups = ["control", "svppa", "nfvppa", "lvppa"]
    group_labels = ["HC", "svPPA", "nfvPPA", "lvPPA"]
    
    # Build L2 matrix
    l2_cols = [c[0] for c in l2_constructs]
    l2_labels = [c[1] for c in l2_constructs]
    
    l2_matrix = []
    for g in groups:
        group_df = df[df["subtype"] == g]
        means = [group_df[col].mean() for col in l2_cols]
        l2_matrix.append(means)
    
    l2_matrix = np.array(l2_matrix).T  # constructs x groups
    
    # Build core features matrix
    core_cols = [c[0] for c in core_features if c[0] in df.columns]
    core_labels = [c[1] for c in core_features if c[0] in df.columns]
    
    core_matrix = []
    for g in groups:
        group_df = df[df["subtype"] == g]
        means = [group_df[col].mean() for col in core_cols]
        core_matrix.append(means)
    
    core_matrix = np.array(core_matrix).T
    
    # Statistical significance markers
    def get_significance(col, groups_df):
        """Calculate significance between svPPA and HC."""
        hc = groups_df[groups_df["subtype"] == "control"][col].dropna()
        sv = groups_df[groups_df["subtype"] == "svppa"][col].dropna()
        if len(hc) > 0 and len(sv) > 0:
            _, p = stats.mannwhitneyu(hc, sv, alternative='two-sided')
            if p < 0.001:
                return "***"
            elif p < 0.01:
                return "**"
            elif p < 0.05:
                return "*"
        return ""
    
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 10), 
                                    gridspec_kw={'height_ratios': [10, 4]})
    
    # Panel 1: Layer 2 constructs
    # Add significance markers to labels
    l2_labels_sig = []
    for i, col in enumerate(l2_cols):
        sig = get_significance(col, df)
        l2_labels_sig.append(f"{l2_labels[i]}{sig}")
    
    sns.heatmap(l2_matrix, ax=ax1, cmap="RdBu_r", vmin=0, vmax=1,
                xticklabels=group_labels, yticklabels=l2_labels_sig,
                annot=True, fmt='.2f', annot_kws={'size': 9},
                cbar_kws={'label': 'Score [0-1]'}, 
                linewidths=0.5, linecolor='white')
    ax1.set_title("Layer 2 Clinical Construct Scores by Diagnostic Group", 
                  fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax1.tick_params(axis='both', labelsize=10)
    
    # Panel 2: Core adequacy scores
    core_labels_sig = []
    for i, col in enumerate(core_cols):
        sig = get_significance(col, df)
        core_labels_sig.append(f"{core_labels[i]}{sig}")
    
    sns.heatmap(core_matrix, ax=ax2, cmap="RdBu_r", vmin=0, vmax=1,
                xticklabels=group_labels, yticklabels=core_labels_sig,
                annot=True, fmt='.2f', annot_kws={'size': 10},
                cbar_kws={'label': 'Score [0-1]'}, 
                linewidths=0.5, linecolor='white')
    ax2.set_title("Core Adequacy Scores by Diagnostic Group", 
                  fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Diagnostic Group", fontsize=11)
    ax2.set_ylabel("")
    ax2.tick_params(axis='both', labelsize=10)
    
    plt.tight_layout()
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved combined heatmap to {args.output}")
    
    # Also create L2-only version
    fig2, ax = plt.subplots(figsize=(6, 8))
    
    sns.heatmap(l2_matrix, ax=ax, cmap="RdBu_r", vmin=0, vmax=1,
                xticklabels=group_labels, yticklabels=l2_labels_sig,
                annot=True, fmt='.2f', annot_kws={'size': 9},
                cbar_kws={'label': 'Score [0-1]'}, 
                linewidths=0.5, linecolor='white')
    ax.set_title("Layer 2 Clinical Construct Scores", 
                 fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Diagnostic Group", fontsize=11)
    ax.set_ylabel("")
    ax.tick_params(axis='both', labelsize=10)
    
    plt.tight_layout()
    
    l2_output = args.output.parent / "l2_only_heatmap_paper.png"
    plt.savefig(l2_output, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved L2-only heatmap to {l2_output}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("LAYER 2 CONSTRUCT SCORES BY GROUP")
    print("="*70)
    print(f"{'Construct':<35} {'HC':>8} {'svPPA':>8} {'nfvPPA':>8} {'lvPPA':>8}")
    print("-"*70)
    for i, label in enumerate(l2_labels):
        row = l2_matrix[i]
        sig = get_significance(l2_cols[i], df)
        print(f"{label:<35} {row[0]:>8.3f} {row[1]:>8.3f} {row[2]:>8.3f} {row[3]:>8.3f} {sig}")
    
    print("\n" + "="*70)
    print("CORE ADEQUACY SCORES BY GROUP")
    print("="*70)
    print(f"{'Feature':<25} {'HC':>8} {'svPPA':>8} {'nfvPPA':>8} {'lvPPA':>8}")
    print("-"*70)
    for i, label in enumerate(core_labels):
        row = core_matrix[i]
        sig = get_significance(core_cols[i], df)
        print(f"{label:<25} {row[0]:>8.3f} {row[1]:>8.3f} {row[2]:>8.3f} {row[3]:>8.3f} {sig}")
    
    print("\n* p<0.05, ** p<0.01, *** p<0.001 (HC vs svPPA, Mann-Whitney U)")
    
    # Highlight key clinical patterns
    print("\n" + "="*70)
    print("KEY CLINICAL PATTERNS (supporting paper claims)")
    print("="*70)
    
    # svPPA patterns
    sv_anomia = df[df["subtype"] == "svppa"]["l2_anomia"].mean()
    sv_empty = df[df["subtype"] == "svppa"]["l2_empty_speech"].mean()
    sv_semantic = df[df["subtype"] == "svppa"]["l2_semantic_paraphasias"].mean()
    print(f"\nsvPPA semantic deficits:")
    print(f"  Anomia: {sv_anomia:.1%}")
    print(f"  Empty Speech: {sv_empty:.1%}")
    print(f"  Semantic Paraphasias: {sv_semantic:.1%}")
    
    # nfvPPA patterns
    nfv_agram = df[df["subtype"] == "nfvppa"]["l2_agrammatism"].mean()
    nfv_rate = df[df["subtype"] == "nfvppa"]["l2_reduced_speech_rate"].mean()
    print(f"\nnfvPPA motor/grammar deficits:")
    print(f"  Agrammatism: {nfv_agram:.1%}")
    print(f"  Reduced Speech Rate: {nfv_rate:.1%}")


if __name__ == "__main__":
    main()
