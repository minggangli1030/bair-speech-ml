"""PPA audio pipeline: diarization (pyannote) + ASR (Whisper) + optional phoneme (SylWav/Chenxu)."""

from .runner import run_pipeline_on_audio, run_pipeline_on_manifest

__all__ = ["run_pipeline_on_audio", "run_pipeline_on_manifest"]
