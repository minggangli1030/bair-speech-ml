"""ASR interface and Whisper implementation. Pluggable backend."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ASRInterface(ABC):
    """Pluggable ASR backend: transcribe audio and return words with timestamps."""

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path | str,
        language: str | None = None,
    ) -> dict[str, Any]:
        """
        Transcribe audio file. Returns dict with at least:
        - "text": full transcript string
        - "words": list of {"word": str, "start": float, "end": float} (if available)
        - "segments": optional list of segments with start/end/text
        """
        pass


class WhisperASR(ASRInterface):
    """Whisper-based ASR with word-level timestamps when available."""

    def __init__(
        self,
        model_name: str = "base",
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._extra = kwargs

    def _load_model(self) -> Any:
        if self._model is None:
            import whisper
            self._model = whisper.load_model(self.model_name, device=self.device)
            logger.info("Loaded Whisper model: %s on %s", self.model_name, self.device)
        return self._model

    def transcribe(
        self,
        audio_path: Path | str,
        language: str | None = None,
    ) -> dict[str, Any]:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio not found: {path}")
        model = self._load_model()
        result = model.transcribe(
            str(path),
            language=language,
            word_timestamps=True,
            **self._extra,
        )
        # Build words from segments when word_timestamps exist
        words: list[dict[str, Any]] = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []) or []:
                word = w.get("word", "").strip()
                if not word:
                    continue
                words.append({
                    "word": word,
                    "start": float(w.get("start", 0)),
                    "end": float(w.get("end", 0)),
                })
            # If no word-level, use segment as single "word"
            if not seg.get("words") and seg.get("text", "").strip():
                words.append({
                    "word": seg["text"].strip(),
                    "start": float(seg.get("start", 0)),
                    "end": float(seg.get("end", 0)),
                })
        return {
            "text": result.get("text", "").strip(),
            "words": words,
            "segments": result.get("segments", []),
            "language": result.get("language"),
        }


def get_asr(backend: str = "whisper", **kwargs: Any) -> ASRInterface:
    """Factory: return ASR implementation by backend name."""
    backend = backend.lower()
    if backend == "whisper":
        return WhisperASR(**kwargs)
    # TODO: add wav2vec2 when needed
    # if backend == "wav2vec2":
    #     return Wav2Vec2ASR(**kwargs)
    raise ValueError(f"Unknown ASR backend: {backend}")
