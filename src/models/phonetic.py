"""Phonetic encoder interface and stub. Lab-provided model is an external callable."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PhoneticEncoderInterface(ABC):
    """
    Lab-provided phonetic encoding: map audio or text to phone-level representation.
    Treat as external callable with Python API; implement adapter here.
    """

    @abstractmethod
    def encode(
        self,
        audio_path: Path | str | None = None,
        text: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Encode to phoneme-level. Returns dict with at least:
        - "phones": list of phone labels or (start, end, phone)
        - "duration_sec": optional total duration
        Either audio_path or text should be provided.
        """
        pass


class StubPhoneticEncoder(PhoneticEncoderInterface):
    """
    Stub when lab encoder is not available. Returns placeholder structure
    so pipeline runs; articulation rate etc. can use word-level timing only.
    """

    def encode(
        self,
        audio_path: Path | str | None = None,
        text: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # TODO: Integrate lab-provided phonetic encoding model when available.
        return {
            "phones": [],
            "duration_sec": None,
            "stub": True,
        }


def get_phonetic_encoder(
    encoder_path: str | None = None,
    use_stub_if_unavailable: bool = True,
) -> PhoneticEncoderInterface:
    """
    Factory: load lab encoder from encoder_path (e.g. module:ClassName or path to callable).
    If not available and use_stub_if_unavailable, return StubPhoneticEncoder.
    """
    if encoder_path:
        try:
            # Optional: dynamic load, e.g. importlib.import_module(module).ClassName()
            # For now we only expose stub; user can register their encoder.
            # Example: "mylab.phonetic:LabPhoneticEncoder"
            if ":" in encoder_path:
                mod_name, attr = encoder_path.rsplit(":", 1)
                import importlib
                mod = importlib.import_module(mod_name)
                cls = getattr(mod, attr)
                return cls()
            raise ImportError("encoder_path must be module:ClassName")
        except Exception as e:
            logger.warning("Phonetic encoder load failed (%s): %s", encoder_path, e)
            if use_stub_if_unavailable:
                return StubPhoneticEncoder()
            raise
    return StubPhoneticEncoder()
