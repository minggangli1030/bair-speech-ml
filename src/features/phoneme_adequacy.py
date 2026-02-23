"""
Phonological adequacy scoring (S_ph) from SylWav phoneme sequences.

Computes phonological adequacy by comparing:
1. SylWav phoneme sequences (direct acoustic-to-phoneme)
2. Expected phonemes from ASR transcript (via G2P)

The divergence between these measures motor-speech and phonological errors.
svPPA should show HIGH S_ph (phonology preserved), nfvPPA/lvPPA show LOW S_ph.
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# CMU ARPAbet phoneme inventory (39 phonemes + SIL)
ARPABET_PHONEMES = {
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "DX",
    "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K", "L",
    "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH",
    "UH", "UW", "V", "W", "Y", "Z", "ZH", "SIL"
}

# Simple G2P dictionary for common words (extend as needed)
# In production, use a proper G2P like phonemizer or CMUdict
G2P_DICT = {
    "a": ["AH"], "the": ["DH", "AH"], "and": ["AE", "N", "D"], "is": ["IH", "Z"],
    "are": ["AA", "R"], "there": ["DH", "EH", "R"], "here": ["HH", "IH", "R"],
    "man": ["M", "AE", "N"], "woman": ["W", "UH", "M", "AH", "N"],
    "boy": ["B", "OY"], "girl": ["G", "ER", "L"], "dog": ["D", "AO", "G"],
    "kite": ["K", "AY", "T"], "blanket": ["B", "L", "AE", "NG", "K", "IH", "T"],
    "tree": ["T", "R", "IY"], "house": ["HH", "AW", "S"], "lake": ["L", "EY", "K"],
    "water": ["W", "AO", "T", "ER"], "boat": ["B", "OW", "T"],
    "ball": ["B", "AO", "L"], "book": ["B", "UH", "K"], "reading": ["R", "IY", "D", "IH", "NG"],
    "sitting": ["S", "IH", "T", "IH", "NG"], "holding": ["HH", "OW", "L", "D", "IH", "NG"],
    "flying": ["F", "L", "AY", "IH", "NG"], "running": ["R", "AH", "N", "IH", "NG"],
    "picnic": ["P", "IH", "K", "N", "IH", "K"], "family": ["F", "AE", "M", "AH", "L", "IY"],
    "grass": ["G", "R", "AE", "S"], "sky": ["S", "K", "AY"],
    "something": ["S", "AH", "M", "TH", "IH", "NG"], "thing": ["TH", "IH", "NG"],
    "with": ["W", "IH", "TH"], "this": ["DH", "IH", "S"], "that": ["DH", "AE", "T"],
    "it": ["IH", "T"], "in": ["IH", "N"], "on": ["AA", "N"], "of": ["AH", "V"],
    "to": ["T", "UW"], "for": ["F", "AO", "R"], "but": ["B", "AH", "T"],
    "or": ["AO", "R"], "an": ["AE", "N"], "like": ["L", "AY", "K"],
    "i": ["AY"], "you": ["Y", "UW"], "he": ["HH", "IY"], "she": ["SH", "IY"],
    "they": ["DH", "EY"], "we": ["W", "IY"], "my": ["M", "AY"], "his": ["HH", "IH", "Z"],
    "her": ["HH", "ER"], "their": ["DH", "EH", "R"],
}


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words."""
    return re.findall(r"[a-zA-Z]+", text.lower())


def _g2p_word(word: str) -> list[str]:
    """Convert word to phonemes using simple dictionary lookup."""
    word = word.lower().strip()
    if word in G2P_DICT:
        return G2P_DICT[word]
    # Fallback: use character-based approximation
    phonemes = []
    for c in word:
        if c in "aeiou":
            phonemes.append("AH")
        elif c in "bcdfghjklmnpqrstvwxyz":
            mapping = {
                "b": "B", "c": "K", "d": "D", "f": "F", "g": "G",
                "h": "HH", "j": "JH", "k": "K", "l": "L", "m": "M",
                "n": "N", "p": "P", "q": "K", "r": "R", "s": "S",
                "t": "T", "v": "V", "w": "W", "x": "K", "y": "Y", "z": "Z"
            }
            phonemes.append(mapping.get(c, "AH"))
    return phonemes if phonemes else ["AH"]


def _g2p_transcript(transcript: str) -> list[str]:
    """Convert transcript to expected phoneme sequence."""
    words = _tokenize(transcript)
    phonemes = []
    for w in words:
        phonemes.extend(_g2p_word(w))
    return phonemes


def _parse_sylwav_phonemes(phoneme_str: str) -> list[str]:
    """Parse SylWav phoneme string into list of phonemes."""
    tokens = phoneme_str.strip().split()
    # Filter to valid ARPAbet phonemes
    return [p for p in tokens if p in ARPABET_PHONEMES or p.rstrip("0123456789") in ARPABET_PHONEMES]


def _levenshtein_distance(seq1: list[str], seq2: list[str]) -> int:
    """Compute Levenshtein edit distance between two sequences."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


def compute_s_ph(
    transcript: str,
    sylwav_phonemes: str | list[str],
) -> tuple[float, dict[str, Any]]:
    """
    Compute S_ph (phonological adequacy) from transcript and SylWav phonemes.
    
    S_ph measures consistency between:
    - Expected phonemes (G2P of ASR transcript)
    - Observed phonemes (SylWav acoustic phoneme recognizer)
    
    Higher S_ph = phonologically well-formed speech (svPPA pattern)
    Lower S_ph = phonological/motor errors (nfvPPA/lvPPA pattern)
    
    Returns:
        s_ph: float in [0, 1], phonological adequacy
        details: dict with edit distance, phoneme counts, etc.
    """
    if isinstance(sylwav_phonemes, str):
        observed = _parse_sylwav_phonemes(sylwav_phonemes)
    else:
        observed = [p for p in sylwav_phonemes if p in ARPABET_PHONEMES]
    
    expected = _g2p_transcript(transcript)
    
    if not expected or not observed:
        return float("nan"), {"error": "empty_sequence"}
    
    # Remove SIL tokens for alignment (silence is not a phonological error)
    observed_no_sil = [p for p in observed if p != "SIL"]
    
    # Compute edit distance
    edit_dist = _levenshtein_distance(expected, observed_no_sil)
    max_len = max(len(expected), len(observed_no_sil))
    
    # Phoneme error rate (PER)
    per = edit_dist / max_len if max_len > 0 else 0.0
    
    # S_ph = 1 - PER (higher is better)
    s_ph = max(0.0, min(1.0, 1.0 - per))
    
    # Additional features
    silence_ratio = len([p for p in observed if p == "SIL"]) / len(observed) if observed else 0.0
    phoneme_rate = len(observed_no_sil) / max(1, len(_tokenize(transcript)))  # phonemes per word
    
    details = {
        "s_ph": s_ph,
        "phoneme_error_rate": per,
        "edit_distance": edit_dist,
        "expected_phonemes": len(expected),
        "observed_phonemes": len(observed_no_sil),
        "silence_ratio": silence_ratio,
        "phonemes_per_word": phoneme_rate,
    }
    
    return s_ph, details


def parse_pipeline_output(filepath: Path | str) -> dict[str, Any]:
    """
    Parse the pipeline output file to extract ASR transcription and phoneme sequences.
    
    Returns dict with:
        - transcript: full participant transcript
        - phonemes: full phoneme sequence
        - segments: list of {timestamp, speaker, duration, transcript, phonemes}
    """
    filepath = Path(filepath)
    if not filepath.exists():
        logger.warning("Pipeline output not found: %s", filepath)
        return {}
    
    content = filepath.read_text(encoding="utf-8", errors="ignore")
    
    segments = []
    participant_transcript = []
    participant_phonemes = []
    
    # Parse the detailed segment view
    segment_pattern = re.compile(
        r"Segment\s+\d+:\s*\n"
        r"\s*Timestamp:\s*(\d+:\d+\.\d+)\s*-\s*(\d+:\d+\.\d+)\s*\(([^)]+)\)\s*\n"
        r"\s*Speaker:\s*(\w+)[^\n]*\n"
        r"\s*ASR Transcription:\s*([^\n]*)\n"
        r"\s*Phonemic Encoding:\s*([^\n]*)",
        re.MULTILINE
    )
    
    for match in segment_pattern.finditer(content):
        start_time, end_time, duration, speaker, transcript, phonemes = match.groups()
        
        seg = {
            "start": start_time,
            "end": end_time,
            "duration": duration.strip(),
            "speaker": speaker.strip(),
            "transcript": transcript.strip(),
            "phonemes": phonemes.strip(),
        }
        segments.append(seg)
        
        # Only use Participant segments (not Clinician)
        if "Participant" in speaker:
            participant_transcript.append(transcript.strip())
            participant_phonemes.append(phonemes.strip())
    
    return {
        "transcript": " ".join(participant_transcript),
        "phonemes": " ".join(participant_phonemes),
        "segments": segments,
        "n_participant_segments": len(participant_transcript),
    }


def compute_s_ph_from_file(pipeline_output_path: Path | str) -> tuple[float, dict[str, Any]]:
    """
    Compute S_ph from a pipeline output file.
    
    Convenience function that parses the file and computes phonological adequacy.
    """
    parsed = parse_pipeline_output(pipeline_output_path)
    
    if not parsed or not parsed.get("transcript") or not parsed.get("phonemes"):
        return float("nan"), {"error": "parse_failed", "path": str(pipeline_output_path)}
    
    s_ph, details = compute_s_ph(parsed["transcript"], parsed["phonemes"])
    details["n_segments"] = parsed.get("n_participant_segments", 0)
    details["transcript_preview"] = parsed["transcript"][:200] + "..." if len(parsed["transcript"]) > 200 else parsed["transcript"]
    
    return s_ph, details
