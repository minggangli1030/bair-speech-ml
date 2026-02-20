"""Disk cache for ASR transcripts and phonetic encoder outputs."""

from __future__ import annotations

import json
import hashlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _cache_key(subject_id: str, audio_path: str, suffix: str) -> str:
    """Generate a stable cache key from subject and audio path."""
    raw = f"{subject_id}:{audio_path}:{suffix}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get_transcript_path(cache_dir: Path, subject_id: str, audio_path: str) -> Path:
    """Path where transcript JSON for this subject/audio is stored."""
    key = _cache_key(subject_id, audio_path, "transcript")
    return cache_dir / "transcripts" / f"{subject_id}_{key}.json"


def get_phonetic_path(cache_dir: Path, subject_id: str, audio_path: str) -> Path:
    """Path where phonetic output for this subject/audio is stored."""
    key = _cache_key(subject_id, audio_path, "phonetic")
    return cache_dir / "phonetics" / f"{subject_id}_{key}.json"


def load_cached_transcript(cache_dir: Path, subject_id: str, audio_path: str) -> dict[str, Any] | None:
    """Load cached transcript JSON if it exists. Returns None if missing."""
    path = get_transcript_path(cache_dir, subject_id, audio_path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.debug("Loaded cached transcript: %s", path.name)
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load cache %s: %s", path, e)
        return None


def save_transcript(cache_dir: Path, subject_id: str, audio_path: str, data: dict[str, Any]) -> Path:
    """Save transcript JSON to cache. Creates parent dirs if needed."""
    path = get_transcript_path(cache_dir, subject_id, audio_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.debug("Saved transcript cache: %s", path)
    return path


def load_cached_phonetic(cache_dir: Path, subject_id: str, audio_path: str) -> dict[str, Any] | None:
    """Load cached phonetic output if it exists. Returns None if missing."""
    path = get_phonetic_path(cache_dir, subject_id, audio_path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.debug("Loaded cached phonetic: %s", path.name)
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load cache %s: %s", path, e)
        return None


def save_phonetic(cache_dir: Path, subject_id: str, audio_path: str, data: dict[str, Any]) -> Path:
    """Save phonetic output to cache."""
    path = get_phonetic_path(cache_dir, subject_id, audio_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.debug("Saved phonetic cache: %s", path)
    return path
