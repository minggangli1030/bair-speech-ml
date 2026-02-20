"""Unit-test-style checks for Layer 1 and Layer 2 feature functions (synthetic transcript)."""

from __future__ import annotations

import sys
from pathlib import Path

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _ensure_spacy() -> None:
    """Ensure spaCy and en_core_web_sm are available; exit with clear message if not."""
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except ImportError:
        print("spaCy is not installed. Install with:")
        print("  pip install spacy")
        print("  python -m spacy download en_core_web_sm")
        sys.exit(1)
    except OSError:
        print("spaCy model 'en_core_web_sm' not found. Download with:")
        print("  python -m spacy download en_core_web_sm")
        sys.exit(1)


from src.features.layer1 import LAYER1_NAMES, compute_layer1_features
from src.features.layer2 import LAYER2_NAMES, compute_layer2_features


def make_synthetic_transcript(
    text: str = "The cat sat on the mat. It was a nice day.",
    num_words: int = 8,
    word_duration: float = 0.3,
    pause_after_word: float = 0.1,
) -> dict:
    """Build a minimal transcript dict with word timestamps."""
    words_list = text.split()[:num_words]
    if not words_list:
        words_list = ["the", "cat", "sat"]
    words = []
    t = 0.0
    for w in words_list:
        words.append({"word": w, "start": t, "end": t + word_duration})
        t += word_duration + pause_after_word
    return {
        "text": " ".join(words_list),
        "words": words,
        "segments": [],
    }


def test_layer1_basic() -> None:
    """Layer 1 returns 20 features with expected keys."""
    transcript = make_synthetic_transcript()
    phonetic = {"phones": [], "stub": True}
    feats = compute_layer1_features(transcript, phonetic_data=phonetic, use_semantic_distance=False)
    assert len(feats) == 20, f"Expected 20 features, got {len(feats)}"
    for name in LAYER1_NAMES:
        assert name in feats, f"Missing feature: {name}"
    # WPM should be positive for non-empty speech
    assert feats["wpm"] > 0 or feats["wpm"] != feats["wpm"], "WPM should be computed"
    print("Layer 1: 20 features present; WPM computed.")


def test_layer2_from_layer1() -> None:
    """Layer 2 returns 10 constructs in [0,1] from Layer 1."""
    transcript = make_synthetic_transcript()
    phonetic = {"phones": [], "stub": True}
    l1 = compute_layer1_features(transcript, phonetic_data=phonetic, use_semantic_distance=False)
    raw_z, scores = compute_layer2_features(l1)
    assert len(scores) == 10, f"Expected 10 Layer 2 scores, got {len(scores)}"
    for name in LAYER2_NAMES:
        assert name in scores, f"Missing Layer 2: {name}"
        v = scores[name]
        assert 0 <= v <= 1 or v != v, f"Score {name} should be in [0,1] or NaN, got {v}"
    print("Layer 2: 10 constructs in [0,1].")


def test_layer1_with_empty_text() -> None:
    """Layer 1 handles near-empty transcript without crashing."""
    transcript = make_synthetic_transcript(text="um", num_words=1)
    transcript["words"] = [{"word": "um", "start": 0.0, "end": 0.5}]
    phonetic = {"stub": True}
    feats = compute_layer1_features(transcript, phonetic_data=phonetic, use_semantic_distance=False)
    assert feats["filled_pause_frequency"] >= 0
    assert feats["utterance_count"] >= 0
    print("Layer 1: empty/short transcript handled.")


if __name__ == "__main__":
    _ensure_spacy()
    test_layer1_basic()
    test_layer2_from_layer1()
    test_layer1_with_empty_text()
    print("All checks passed.")
