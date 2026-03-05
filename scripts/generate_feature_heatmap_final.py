#!/usr/bin/env python3
"""
Generate final publication-quality heatmap for Interspeech paper.
Shows the model input features by diagnostic group.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def main():
    parser = argparse.ArgumentParser(description="Generate final feature heatmap")
    parser.add_argument("--features", type=Path, default=Path("outputs/ppa_dissociation_full.csv"))
    parser.add_argument("--l2-features", type=Path, default=Path("outputs/layer2_fixed.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/feature_heatmap_final.png"))
    args = parser.parse_args()

    df_main = pd.read_csv(args.features)
    df_l2 = pd.read_csv(args.l2_features)
    df = df_main.merge(df_l2, on=["subject_id", "session", "subtype", "label"], how="inner")
    
    print(f"Loaded {len(df)} samples")
    print(df["subtype"].value_counts())
    
    groups = ["control", "svppa", "nfvppa", "lvppa"]
    group_labels = ["HC", "svPPA", "nfvPPA", "lvPPA"]
    
    # Select key features for the paper
    # Adequacy scores (higher = better)
    adequacy_features = [
        ("s_ph", "S_ph (Phonological)"),
        ("s_sem", "S_sem (Semantic)"),
        ("s_sem_graph", "S_sem^graph (Scene Graph)"),
    ]
    
    # Impairment scores (higher = more impairment) - select most clinically relevant
    impairment_features = [
        ("l2_anomia", "Anomia"),
        ("l2_empty_speech", "Empty Speech"),
        ("l2_semantic_paraphasias", "Semantic Paraphasias"),
        ("l2_agrammatism", "Agrammatism"),
        ("l2_reduced_speech_rate", "Reduced Speech Rate"),
        ("l2_phonemic_paraphasias_neologisms", "Phonemic Paraphasias"),
    ]
    
    def get_significance_markers(col, df):
        """Get significance markers for svPPA vs HC and nfvPPA vs HC."""
        hc = df[df["subtype"] == "control"][col].dropna()
        sv = df[df["subtype"] == "svppa"][col].dropna()
        nfv = df[df["subtype"] == "nfvppa"][col].dropna()
        
        markers = []
        # HC vs svPPA
        if len(hc) > 0 and len(sv) > 0:
            _, p = stats.mannwhitneyu(hc, sv, alternative='two-sided')
            if p < 0.001:
                markers.append("***")
            elif p < 0.01:
                markers.append("**")
            elif p < 0.05:
                markers.append("*")
            else:
                markers.append("")
        else:
            markers.append("")
        return markers[0]
    
    # Build adequacy matrix (flip so higher = better shown as red)
    adequacy_cols = [f[0] for f in adequacy_features]
    adequacy_labels = [f[1] for f in adequacy_features]
    
    adequacy_matrix = []
    for g in groups:
        group_df = df[df["subtype"] == g]
        means = [group_df[col].mean() for col in adequacy_cols]
        adequacy_matrix.append(means)
    adequacy_matrix = np.array(adequacy_matrix).T
    
    # Build impairment matrix
    impairment_cols = [f[0] for f in impairment_features]
    impairment_labels = [f[1] for f in impairment_features]
    
    impairment_matrix = []
    for g in groups:
        group_df = df[df["subtype"] == g]
        means = [group_df[col].mean() for col in impairment_cols]
        impairment_matrix.append(means)
    impairment_matrix = np.array(impairment_matrix).T
    
    # Add significance markers to labels
    adequacy_labels_sig = []
    for col, label in zip(adequacy_cols, adequacy_labels):
        sig = get_significance_markers(col, df)
        adequacy_labels_sig.append(f"{label}{sig}")
    
    impairment_labels_sig = []
    for col, label in zip(impairment_cols, impairment_labels):
        sig = get_significance_markers(col, df)
        impairment_labels_sig.append(f"{label}{sig}")
    
    # Create figure: wide, with generous spacing to avoid overlap
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(18, 7),
        gridspec_kw={"width_ratios": [1, 1.3], "wspace": 0.5},
    )
    
    # Panel 1: Adequacy scores
    sns.heatmap(
        adequacy_matrix,
        ax=ax1,
        cmap="RdBu_r",
        vmin=0,
        vmax=0.5,
        xticklabels=group_labels,
        yticklabels=adequacy_labels_sig,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 14, "weight": "bold"},
        cbar_kws={"label": "Adequacy Score", "shrink": 0.7},
        linewidths=1.0,
        linecolor="white",
        square=False,
    )
    ax1.set_title("Adequacy Scores (higher = better)", fontsize=15, fontweight="bold", pad=10)
    ax1.set_xlabel("Diagnostic Group", fontsize=14)
    ax1.tick_params(axis="y", labelsize=14, which="major", pad=8)
    for label in ax1.get_yticklabels():
        label.set_fontweight("bold")
    ax1.tick_params(axis="x", labelsize=14, rotation=0)
    
    # Panel 2: Clinical construct scores
    sns.heatmap(
        impairment_matrix,
        ax=ax2,
        cmap="RdBu_r",
        vmin=0.2,
        vmax=0.7,
        xticklabels=group_labels,
        yticklabels=impairment_labels_sig,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 14},
        cbar_kws={"label": "Construct Score", "shrink": 0.7},
        linewidths=1.0,
        linecolor="white",
        square=False,
    )
    ax2.set_title("Clinical Construct Scores (Layer 2)", fontsize=15, fontweight="bold", pad=10)
    ax2.set_xlabel("Diagnostic Group", fontsize=14)
    ax2.tick_params(axis="y", labelsize=14, which="major", pad=8)
    for label in ax2.get_yticklabels():
        label.set_fontweight("bold")
    ax2.tick_params(axis="x", labelsize=14, rotation=0)
    
    # Explicit margins to prevent overlap and clipping
    plt.subplots_adjust(left=0.12, right=0.92, bottom=0.14, top=0.92, wspace=0.45)
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    print(f"\nSaved heatmap to {args.output}")
    
    # Print summary for paper
    print("\n" + "="*70)
    print("KEY STATISTICS FOR PAPER TEXT")
    print("="*70)
    
    print("\n--- Adequacy Scores ---")
    for i, (col, label) in enumerate(zip(adequacy_cols, adequacy_labels)):
        row = adequacy_matrix[i]
        sig = get_significance_markers(col, df)
        hc_sv_diff = row[0] - row[1]
        print(f"{label}: HC={row[0]:.3f}, svPPA={row[1]:.3f} (diff={hc_sv_diff:+.3f}) {sig}")
    
    print("\n--- Clinical Construct Scores (HC vs svPPA) ---")
    for i, (col, label) in enumerate(zip(impairment_cols, impairment_labels)):
        row = impairment_matrix[i]
        sig = get_significance_markers(col, df)
        sv_hc_diff = row[1] - row[0]
        print(f"{label}: HC={row[0]:.3f}, svPPA={row[1]:.3f} (diff={sv_hc_diff:+.3f}) {sig}")
    
    print("\n* p<0.05, ** p<0.01, *** p<0.001 (HC vs svPPA)")


if __name__ == "__main__":
    main()
