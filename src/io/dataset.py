"""Load manifest CSV, iterate subjects, and load audio."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import pandas as pd

logger = logging.getLogger(__name__)


def load_manifest(manifest_path: Path, audio_dir: Path | None = None) -> pd.DataFrame:
    """
    Load manifest CSV with columns: subject_id, audio_path, label (optional), task (optional).
    audio_path can be relative to manifest dir or audio_dir; resolved to absolute.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    base = manifest_path.parent
    adir = audio_dir or base
    df = pd.read_csv(manifest_path)
    required = {"subject_id", "audio_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")

    # Resolve audio paths
    def resolve(p: str) -> Path:
        path = Path(p)
        if not path.is_absolute():
            path = (base / path) if (base / path).exists() else (adir / path)
        return path.resolve()

    df["audio_path_resolved"] = df["audio_path"].astype(str).map(resolve)
    return df


def iter_subjects(
    df: pd.DataFrame,
    label_column: str = "label",
) -> Iterator[tuple[str, pd.DataFrame, str | None]]:
    """
    Iterate over unique subjects. Yields (subject_id, subject_rows, label).
    Label is taken from first row per subject if column exists; None otherwise.
    """
    for subject_id, group in df.groupby("subject_id", sort=False):
        label = None
        if label_column in group.columns:
            vals = group[label_column].dropna().unique()
            label = str(vals[0]) if len(vals) else None
        yield subject_id, group, label


def load_audio_for_subject(
    subject_rows: pd.DataFrame,
    target_sr: int = 16000,
) -> tuple[list[Path], list[float], list[list[float]]]:
    """
    Load audio files for a subject's rows. Returns (paths, sample_rates, waveforms).
    Waveforms are normalized to target_sr and returned as list of (samples as list of float).
    """
    import soundfile as sf
    import numpy as np
    try:
        import resampy
    except ImportError:
        resampy = None

    paths: list[Path] = []
    sample_rates: list[float] = []
    waveforms: list[list[float]] = []

    for _, row in subject_rows.iterrows():
        path = row["audio_path_resolved"]
        if not path.exists():
            logger.warning("Audio file not found: %s", path)
            continue
        try:
            data, sr = sf.read(str(path), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != target_sr and resampy is not None:
                data = resampy.resample(data, sr, target_sr)
                sr = target_sr
            paths.append(Path(path))
            sample_rates.append(float(sr))
            waveforms.append(data.tolist())
        except Exception as e:
            logger.warning("Failed to load %s: %s", path, e)

    return paths, sample_rates, waveforms
