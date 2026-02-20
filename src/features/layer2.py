"""Layer 2 constructs (10) inferred from Layer 1 via z-scores and weighted averages."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .layer1 import LAYER1_NAMES

logger = logging.getLogger(__name__)

LAYER2_NAMES = [
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

# Default weights: construct index -> list of (layer1_index, weight)
# Layer 1 indices 0..19 as in LAYER1_NAMES
DEFAULT_LAYER2_WEIGHTS: dict[int, list[tuple[int, float]]] = {
    0: [(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0)],   # Reduced speech rate
    1: [(5, 1.0), (6, 1.0), (7, 1.0), (9, 1.0)],             # Reduced length/complexity
    2: [(7, 1.0), (14, 1.0), (15, 1.0), (16, 1.0)],          # Agrammatism
    3: [(15, 1.0), (16, 1.0), (12, 1.0)],                    # Paragrammatism
    4: [(17, 1.0), (18, 1.0), (2, 1.0), (8, 1.0)],           # Anomia
    5: [(10, 1.0), (11, 1.0), (13, 1.0), (12, 1.0)],         # Empty speech
    6: [(19, 1.0), (12, 1.0), (10, 1.0)],                    # Semantic paraphasias
    7: [(15, 1.0), (18, 1.0)],                                # Phonemic paraphasias
    8: [(18, 1.0), (8, 1.0), (4, 1.0)],                      # Self-correction
    9: [(0, 1.0), (5, 1.0), (10, 1.0), (11, 1.0), (19, 1.0)], # Overall impairment
}


def _layer1_vector(l1_dict: dict[str, float]) -> np.ndarray:
    """Ordered vector of Layer 1 values (length 20). NaN preserved."""
    return np.array([l1_dict.get(name, np.nan) for name in LAYER1_NAMES], dtype=float)


def _zscore_vector(vec: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None) -> np.ndarray:
    """Z-score vector; use provided mean/std if given (e.g. from training)."""
    out = np.full_like(vec, np.nan)
    valid = ~np.isnan(vec)
    if mean is not None and std is not None:
        out[valid] = (vec[valid] - mean[valid]) / np.where(std[valid] != 0, std[valid], 1.0)
        return out
    if np.any(valid):
        m = np.nanmean(vec)
        s = np.nanstd(vec)
        s = s if s and s > 0 else 1.0
        out[valid] = (vec[valid] - m) / s
    return out


def _weighted_score(zvec: np.ndarray, indices_and_weights: list[tuple[int, float]]) -> float:
    """Weighted average of z-scores at given indices. NaN excluded from denominator."""
    vals = []
    wsum = 0.0
    for i, w in indices_and_weights:
        if i < len(zvec) and not np.isnan(zvec[i]):
            vals.append(zvec[i] * w)
            wsum += w
    if not vals or wsum == 0:
        return np.nan
    return sum(vals) / wsum


def _sigmoid_to_01(x: float) -> float:
    """Map real line to (0, 1). Use tanh then scale: (tanh(x)+1)/2."""
    if np.isnan(x):
        return np.nan
    return float((np.tanh(x) + 1.0) / 2.0)


def compute_layer2_features(
    layer1_dict: dict[str, float],
    layer2_weights: dict[int, list[tuple[int, float]]] | None = None,
    population_mean: np.ndarray | None = None,
    population_std: np.ndarray | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Infer Layer 2 scores from Layer 1. Uses z-scores then weighted average per construct,
    then maps to [0, 1] via (tanh(z)+1)/2.

    Returns:
        raw_z: dict of LAYER2_NAMES -> raw weighted z-score
        scores: dict of LAYER2_NAMES -> score in [0, 1]
    """
    weights = layer2_weights or DEFAULT_LAYER2_WEIGHTS
    vec = _layer1_vector(layer1_dict)
    zvec = _zscore_vector(vec, population_mean, population_std)

    raw_z: dict[str, float] = {}
    scores: dict[str, float] = {}
    for idx, name in enumerate(LAYER2_NAMES):
        ind_w = weights.get(idx, [])
        z = _weighted_score(zvec, ind_w)
        raw_z[name] = float(z) if not np.isnan(z) else np.nan
        scores[name] = _sigmoid_to_01(z)
    return raw_z, scores
