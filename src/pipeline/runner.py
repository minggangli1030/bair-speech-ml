"""
Diarization + ASR + optional phoneme pipeline for PPA Picnic task.

Ported from code/diarization_asr_pipeline.py. Runs on any audio directory;
output format matches existing *_feature_extraction.txt for compatibility
with phoneme_adequacy.parse_pipeline_output() and dissociation analysis.

Requires: pyannote.audio, whisper, soundfile.
Optional: phoneme decoding via code repo (set CODE_DIR or phoneme_code_dir).
"""

from __future__ import annotations

import os
import re
import shutil
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

# Lazy imports for heavy deps
_pyannote_pipeline = None
_whisper_model = None
_decode_phonemes_fn = None


def _format_time(seconds: float) -> str:
    """Format seconds as MM:SS.mmm."""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    minutes = total_seconds // 60
    secs = total_seconds % 60
    milliseconds = int((td.total_seconds() - total_seconds) * 1000)
    return f"{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def _extract_audio_segment(audio_path: Path, start_time: float, end_time: float, output_path: Path) -> Path:
    """Extract a segment from audio file."""
    audio_data, sample_rate = sf.read(str(audio_path))
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    segment = audio_data[start_sample:end_sample]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), segment, sample_rate)
    return output_path


def _clean_asr_text(text: str) -> str:
    """Remove common ASR artifacts (e.g. leading 's')."""
    text = (text or "").strip()
    while text.lower().startswith("s ") or (
        len(text) > 1 and text[0].lower() == "s" and text[1:2].strip() and (text[1].isupper() or text[1].islower())
    ):
        text = text[1:].lstrip()
    if len(text) > 1 and text[0].lower() == "s" and text[1] != " ":
        text = text[1:].lstrip()
    return text


def _get_diarizer(hf_token: str | None = None):
    """Lazy-load pyannote diarization pipeline."""
    global _pyannote_pipeline
    if _pyannote_pipeline is None:
        from pyannote.audio import Pipeline
        token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            raise RuntimeError("Pyannote requires HF token. Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN.")
        _pyannote_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
    return _pyannote_pipeline


def _get_whisper_model(model_size: str = "base"):
    """Lazy-load Whisper ASR model."""
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model(model_size)
    return _whisper_model


def _transcribe_segment(audio_path: Path, asr_model: Any, verbose: bool = False) -> str:
    """Transcribe segment with Whisper."""
    try:
        import torch
        result = asr_model.transcribe(
            str(audio_path),
            language="en",
            fp16=torch.cuda.is_available(),
            task="transcribe",
            verbose=False,
            condition_on_previous_text=False,
        )
        text = (result.get("text") or "").strip()
        return _clean_asr_text(text)
    except Exception as e:
        if verbose:
            print(f" [ASR Error: {e}]", flush=True)
        return f"[ASR Error: {str(e)}]"


def _get_phoneme_decoder(phoneme_code_dir: Path | str | None) -> Callable[[Path], str] | None:
    """
    Return decode_phonemes(audio_path) callable if code repo is available.
    phoneme_code_dir: path to repo containing chenxu_inference.py and sylber-wavlm-main/.
    """
    global _decode_phonemes_fn
    if _decode_phonemes_fn is not None:
        return _decode_phonemes_fn
    code_dir = phoneme_code_dir or os.environ.get("CODE_DIR")
    if not code_dir:
        return None
    code_path = Path(code_dir)
    if not (code_path / "chenxu_inference.py").exists():
        return None
    import sys
    if str(code_path) not in sys.path:
        sys.path.insert(0, str(code_path))
    try:
        from chenxu_inference import decode_phonemes as _decode
        _decode_phonemes_fn = _decode
        return _decode_phonemes_fn
    except Exception:
        return None


def run_pipeline_on_audio(
    audio_path: Path,
    output_dir: Path,
    *,
    hf_token: str | None = None,
    num_speakers: int = 2,
    min_segment_duration: float = 1.0,
    whisper_model_size: str = "base",
    phoneme_code_dir: Path | str | None = None,
    skip_existing: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run diarization → ASR → (optional) phoneme on one audio file.
    Writes {stem}_feature_extraction.txt and {stem}.rttm to output_dir.
    Returns dict with segments, participant_transcript, participant_phonemes.
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = audio_path.stem
    rttm_path = output_dir / f"{base_name}.rttm"
    output_txt = output_dir / f"{base_name}_feature_extraction.txt"

    if skip_existing and output_txt.exists():
        if verbose:
            print(f"  Skip (exists): {output_txt.name}", flush=True)
        return _parse_existing_output(output_txt)

    # Diarization
    if verbose:
        print(f"  Diarizing...", flush=True)
    pipe = _get_diarizer(hf_token)
    audio_data, sr = sf.read(str(audio_path))
    duration_sec = len(audio_data) / sr
    t0 = time.time()
    out = pipe(str(audio_path), num_speakers=num_speakers)
    diar_elapsed = time.time() - t0
    if verbose:
        print(f"  Diarization done in {diar_elapsed:.1f}s", flush=True)

    ann = out.speaker_diarization
    with open(rttm_path, "w", encoding="utf-8") as f:
        ann.write_rttm(f)

    segments_raw = sorted(list(ann.itertracks(yield_label=True)), key=lambda x: x[0].start)
    segments = [
        (seg, track, label)
        for seg, track, label in segments_raw
        if (seg.end - seg.start) >= min_segment_duration
    ]

    speaker_durations = {}
    for seg, _, label in segments:
        d = seg.end - seg.start
        speaker_durations[label] = speaker_durations.get(label, 0.0) + d
    sorted_speakers = sorted(speaker_durations.items(), key=lambda x: -x[1])
    speaker_mapping = {}
    if len(sorted_speakers) >= 1:
        speaker_mapping[sorted_speakers[0][0]] = "Participant"
    if len(sorted_speakers) >= 2:
        speaker_mapping[sorted_speakers[1][0]] = "Clinician"

    # ASR + phoneme per segment
    decode_phonemes = _get_phoneme_decoder(phoneme_code_dir)
    asr_model = _get_whisper_model(whisper_model_size)
    temp_dir = output_dir / "temp_segments"
    temp_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, (segment, track, label) in enumerate(segments, 1):
        seg_start, seg_end = segment.start, segment.end
        speaker_name = speaker_mapping.get(label, "Unknown")
        temp_audio = temp_dir / f"{base_name}_seg_{idx}.wav"
        _extract_audio_segment(audio_path, seg_start, seg_end, temp_audio)

        transcription = _transcribe_segment(temp_audio, asr_model, verbose=verbose)
        phonemes = "[Phoneme decoding not available]"
        if decode_phonemes:
            try:
                phonemes = decode_phonemes(str(temp_audio))
                if phonemes.startswith("["):
                    phonemes = "[Phoneme Error]"
            except Exception as e:
                phonemes = f"[Phoneme Error: {e}]"

        results.append({
            "segment_num": idx,
            "start_time": seg_start,
            "end_time": seg_end,
            "duration": seg_end - seg_start,
            "speaker": speaker_name,
            "speaker_id": label,
            "transcription": transcription,
            "phonemes": phonemes,
        })
        temp_audio.unlink(missing_ok=True)

    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Build participant transcript and phonemes
    participant_texts = []
    participant_phonemes = []
    for r in results:
        if r["speaker"] == "Participant":
            t = _clean_asr_text(r["transcription"])
            if t and not t.startswith("["):
                participant_texts.append(t)
            if r["phonemes"] and not r["phonemes"].startswith("["):
                participant_phonemes.append(r["phonemes"])

    # Write feature_extraction.txt (same format as code/diarization_asr_pipeline.py)
    lines = [
        "=" * 100,
        "PHASE 1: FEATURE EXTRACTION",
        "=" * 100,
        f"Audio File: {audio_path.name}",
        f"Total Segments: {len(segments)}",
        f"Speakers: Participant and Clinician",
        "=" * 100,
        "",
        "Pipeline: Diarization → ASR (OpenAI Whisper) + Phoneme Model (Chenxu) → Feature Extraction",
        "",
        "=" * 100,
        "COMBINED OUTPUT: Timestamps | ASR Transcription | Phonemic Encoding",
        "=" * 100,
        "",
        f"{'#':<4} {'Time Range':<25} {'Speaker':<12} {'Duration':<10} {'ASR Transcription':<50} {'Phonemic Encoding'}",
        "-" * 100,
    ]
    for r in results:
        time_range = f"{_format_time(r['start_time'])} → {_format_time(r['end_time'])}"
        trans = (r["transcription"] or "")[:50]
        lines.append(
            f"{r['segment_num']:<4} {time_range:<25} {r['speaker']:<12} {r['duration']:<10.2f}s {trans:<50} {r['phonemes']}"
        )
    lines.extend([
        "",
        "=" * 100,
        "DETAILED SEGMENT VIEW",
        "=" * 100,
        "",
    ])
    for r in results:
        lines.append(f"Segment {r['segment_num']}:")
        lines.append(f"  Timestamp: {_format_time(r['start_time'])} - {_format_time(r['end_time'])} ({r['duration']:.2f}s)")
        lines.append(f"  Speaker: {r['speaker']} ({r['speaker_id']})")
        lines.append(f"  ASR Transcription: {r['transcription']}")
        lines.append(f"  Phonemic Encoding: {r['phonemes']}")
        lines.append("")

    output_txt.write_text("\n".join(lines), encoding="utf-8")

    return {
        "segments": results,
        "participant_transcript": " ".join(participant_texts),
        "participant_phonemes": " ".join(participant_phonemes),
        "output_txt": str(output_txt),
        "rttm_path": str(rttm_path),
    }


def _parse_existing_output(txt_path: Path) -> dict[str, Any]:
    """Parse existing feature_extraction.txt to get participant transcript and phonemes."""
    from ..features.phoneme_adequacy import parse_pipeline_output
    parsed = parse_pipeline_output(txt_path)
    return {
        "segments": [],
        "participant_transcript": parsed.get("transcript", ""),
        "participant_phonemes": parsed.get("phonemes", ""),
        "output_txt": str(txt_path),
        "rttm_path": str(txt_path.with_suffix(".rttm")),
    }


def run_pipeline_on_manifest(
    manifest_path: Path,
    data_dir: Path,
    output_dir: Path,
    *,
    audio_column: str = "audio_path",
    hf_token: str | None = None,
    phoneme_code_dir: Path | str | None = None,
    skip_existing: bool = True,
    max_files: int | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """
    Run pipeline on all audio files listed in manifest.
    manifest CSV must have audio_path (relative to data_dir) or audio_column.
    Writes all outputs under output_dir (flat: same structure as code pipeline_outputs).
    """
    import pandas as pd
    df = pd.read_csv(manifest_path)
    if audio_column not in df.columns:
        raise ValueError(f"Manifest missing column: {audio_column}")
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = df[audio_column].dropna().unique().tolist()
    if max_files is not None:
        paths = paths[:max_files]

    results = []
    for i, rel in enumerate(paths):
        audio_path = data_dir / rel
        if not audio_path.exists():
            if verbose:
                print(f"[{i+1}/{len(paths)}] Skip (missing): {rel}", flush=True)
            continue
        if verbose:
            print(f"[{i+1}/{len(paths)}] {audio_path.name}", flush=True)
        try:
            out = run_pipeline_on_audio(
                audio_path,
                output_dir,
                hf_token=hf_token,
                phoneme_code_dir=phoneme_code_dir,
                skip_existing=skip_existing,
                verbose=verbose,
            )
            out["audio_path"] = str(audio_path)
            out["relative_path"] = rel
            results.append(out)
        except Exception as e:
            if verbose:
                print(f"  Error: {e}", flush=True)
            results.append({"audio_path": str(audio_path), "error": str(e)})
    return results
