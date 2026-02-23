"""
Phonetic-Semantic Dissociation Index for PPA subtype classification.

Key innovation vs. Peters et al. 2025:
- Explicit dissociation index Δ = S_ph - S_sem that operationalizes
  the clinically validated double dissociation between phonological
  and semantic processing across PPA variants.

Expected patterns:
- svPPA: Δ >> 0 (high S_ph, low S_sem) — phonology preserved, semantics impaired
- nfvPPA: Δ << 0 (low S_ph, high S_sem) — phonology impaired, semantics preserved
- lvPPA: Δ ≈ 0 or mixed — both systems affected
- Controls: Δ ≈ 0 (both high) — no dissociation

References:
- Henry et al. 2014: Phonological processing in PPA
- Hickok & Poeppel 2007: Dual-stream model of speech processing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DissociationResult:
    """Result of phonetic-semantic dissociation computation."""
    s_ph: float  # Phonological adequacy [0, 1]
    s_sem: float  # Semantic adequacy [0, 1]
    delta: float  # Dissociation index = S_ph - S_sem
    
    # Additional features
    phoneme_error_rate: float = 0.0
    silence_ratio: float = 0.0
    phonemes_per_word: float = 0.0
    
    # Metadata
    s_ph_details: dict[str, Any] = field(default_factory=dict)
    s_sem_details: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to flat dictionary for DataFrame."""
        return {
            "s_ph": self.s_ph,
            "s_sem": self.s_sem,
            "delta": self.delta,
            "phoneme_error_rate": self.phoneme_error_rate,
            "silence_ratio": self.silence_ratio,
            "phonemes_per_word": self.phonemes_per_word,
        }


def compute_dissociation(
    s_ph: float,
    s_sem: float,
    s_ph_details: dict[str, Any] | None = None,
    s_sem_details: dict[str, Any] | None = None,
) -> DissociationResult:
    """
    Compute phonetic-semantic dissociation index.
    
    Δ = S_ph - S_sem
    
    Interpretation:
    - Δ > 0: Phonology better than semantics (svPPA pattern)
    - Δ < 0: Semantics better than phonology (nfvPPA pattern)
    - Δ ≈ 0: No dissociation (controls or lvPPA)
    """
    s_ph_details = s_ph_details or {}
    s_sem_details = s_sem_details or {}
    
    # Handle NaN
    if np.isnan(s_ph) or np.isnan(s_sem):
        delta = float("nan")
    else:
        delta = s_ph - s_sem
    
    return DissociationResult(
        s_ph=s_ph,
        s_sem=s_sem,
        delta=delta,
        phoneme_error_rate=s_ph_details.get("phoneme_error_rate", 0.0),
        silence_ratio=s_ph_details.get("silence_ratio", 0.0),
        phonemes_per_word=s_ph_details.get("phonemes_per_word", 0.0),
        s_ph_details=s_ph_details,
        s_sem_details=s_sem_details,
    )


# -------------------------
# Statistical Validation
# -------------------------


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Cliff's delta effect size.
    
    δ = (# concordant pairs - # discordant pairs) / (n1 * n2)
    
    Interpretation (Romano et al. 2006):
    - |δ| < 0.147: negligible
    - 0.147 ≤ |δ| < 0.33: small
    - 0.33 ≤ |δ| < 0.474: medium
    - |δ| ≥ 0.474: large
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    
    # Remove NaN
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    
    n1, n2 = len(x), len(y)
    
    # Count concordant and discordant pairs
    concordant = 0
    discordant = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                concordant += 1
            elif xi < yj:
                discordant += 1
    
    delta = (concordant - discordant) / (n1 * n2)
    return delta


def interpret_cliffs_delta(delta: float) -> str:
    """Interpret Cliff's delta magnitude."""
    if np.isnan(delta):
        return "undefined"
    d = abs(delta)
    if d < 0.147:
        return "negligible"
    elif d < 0.33:
        return "small"
    elif d < 0.474:
        return "medium"
    else:
        return "large"


def mann_whitney_u(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Compute Mann-Whitney U test statistic and p-value.
    
    Returns (U statistic, p-value).
    """
    try:
        from scipy.stats import mannwhitneyu
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        if len(x) < 2 or len(y) < 2:
            return float("nan"), float("nan")
        stat, p = mannwhitneyu(x, y, alternative="two-sided")
        return float(stat), float(p)
    except ImportError:
        logger.warning("scipy not available for Mann-Whitney U test")
        return float("nan"), float("nan")


def compute_group_statistics(
    features_df,
    feature_name: str,
    group_column: str = "label",
) -> dict[str, Any]:
    """
    Compute pairwise statistical comparisons for a feature across groups.
    
    Returns dict with:
    - group_means: {group: mean}
    - group_stds: {group: std}
    - pairwise_comparisons: [{groups, mann_whitney_u, p_value, cliffs_delta, effect_size}]
    """
    import pandas as pd
    
    groups = features_df[group_column].dropna().unique()
    
    # Group statistics
    group_means = {}
    group_stds = {}
    group_data = {}
    
    for g in groups:
        mask = features_df[group_column] == g
        vals = features_df.loc[mask, feature_name].dropna().values
        group_data[g] = vals
        group_means[g] = float(np.mean(vals)) if len(vals) > 0 else float("nan")
        group_stds[g] = float(np.std(vals)) if len(vals) > 0 else float("nan")
    
    # Pairwise comparisons
    comparisons = []
    group_list = list(groups)
    for i in range(len(group_list)):
        for j in range(i + 1, len(group_list)):
            g1, g2 = group_list[i], group_list[j]
            x, y = group_data[g1], group_data[g2]
            
            u_stat, p_val = mann_whitney_u(x, y)
            delta = cliffs_delta(x, y)
            
            comparisons.append({
                "group_1": g1,
                "group_2": g2,
                "n_1": len(x),
                "n_2": len(y),
                "mann_whitney_u": u_stat,
                "p_value": p_val,
                "cliffs_delta": delta,
                "effect_size": interpret_cliffs_delta(delta),
                "significant": p_val < 0.05 if not np.isnan(p_val) else False,
            })
    
    return {
        "feature": feature_name,
        "group_means": group_means,
        "group_stds": group_stds,
        "pairwise_comparisons": comparisons,
    }


def validate_dissociation_hypothesis(
    features_df,
    s_ph_column: str = "s_ph",
    s_sem_column: str = "s_sem",
    delta_column: str = "delta",
    label_column: str = "label",
    label_map: dict[int, str] | None = None,
) -> dict[str, Any]:
    """
    Validate the phonetic-semantic dissociation hypothesis across PPA subtypes.
    
    Expected:
    - svPPA (label=1): Δ >> 0, high S_ph, low S_sem
    - Controls (label=0): Δ ≈ 0, high S_ph, high S_sem
    
    Returns comprehensive statistical report.
    """
    label_map = label_map or {0: "control", 1: "svPPA"}
    
    results = {
        "hypothesis": "svPPA shows Δ >> 0 (phonology preserved, semantics impaired)",
        "features_analyzed": [s_ph_column, s_sem_column, delta_column],
        "n_total": len(features_df),
        "n_by_group": {},
        "feature_statistics": {},
    }
    
    # Count by group
    for label, name in label_map.items():
        n = len(features_df[features_df[label_column] == label])
        results["n_by_group"][name] = n
    
    # Statistics for each feature
    for feat in [s_ph_column, s_sem_column, delta_column]:
        if feat in features_df.columns:
            results["feature_statistics"][feat] = compute_group_statistics(
                features_df, feat, label_column
            )
    
    # Key hypothesis test: Is Δ significantly different between svPPA and controls?
    if delta_column in features_df.columns and 0 in features_df[label_column].values and 1 in features_df[label_column].values:
        control_delta = features_df[features_df[label_column] == 0][delta_column].dropna().values
        svppa_delta = features_df[features_df[label_column] == 1][delta_column].dropna().values
        
        u_stat, p_val = mann_whitney_u(svppa_delta, control_delta)
        delta_effect = cliffs_delta(svppa_delta, control_delta)
        
        results["hypothesis_test"] = {
            "comparison": "svPPA vs control",
            "feature": delta_column,
            "svppa_mean": float(np.mean(svppa_delta)) if len(svppa_delta) > 0 else float("nan"),
            "control_mean": float(np.mean(control_delta)) if len(control_delta) > 0 else float("nan"),
            "mann_whitney_u": u_stat,
            "p_value": p_val,
            "cliffs_delta": delta_effect,
            "effect_size": interpret_cliffs_delta(delta_effect),
            "hypothesis_supported": (
                p_val < 0.05 and delta_effect > 0.33
                if not (np.isnan(p_val) or np.isnan(delta_effect))
                else False
            ),
        }
    
    return results
