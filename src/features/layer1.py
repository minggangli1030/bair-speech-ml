"""Layer 1 proxy features (20) for svPPA speech analysis."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .utils import (
    get_nlp,
    open_class_closed_class_ratio,
    content_word_density,
    pronoun_ratio,
    type_token_ratio,
    function_word_omission_heuristic,
    dependency_parse_completeness,
    morphological_error_heuristic,
    get_nouns_and_proper_nouns,
)

logger = logging.getLogger(__name__)

LAYER1_NAMES = [
    "wpm",
    "articulation_rate_phones_per_sec",
    "mean_pause_duration",
    "pause_to_speech_ratio",
    "initial_response_latency",
    "mlu",
    "median_utterance_length",
    "clause_count_per_utterance",
    "abandoned_utterance_rate",
    "utterance_count",
    "open_class_closed_class_ratio",
    "content_word_density",
    "type_token_ratio",
    "pronoun_usage_ratio",
    "function_word_omission_rate",
    "morphological_error_rate",
    "dependency_parse_completeness",
    "word_finding_pause_duration",
    "filled_pause_frequency",
    "semantic_distance_to_expected",
]

# Filled pause types
FILLED_PAUSES = {"um", "uh", "er", "ah", "eh", "hm", "hmm", "uhm", "umhm"}

# Pause threshold for "word-finding" (seconds)
WORD_FINDING_PAUSE_THRESHOLD = 1.0

# Minimum gap to count as pause (seconds)
MIN_PAUSE_GAP = 0.3


def _get_words_with_times(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract list of {word, start, end} from transcript."""
    words = transcript.get("words") or []
    out = []
    for w in words:
        if isinstance(w, dict):
            out.append({
                "word": w.get("word", "").strip(),
                "start": float(w.get("start", 0)),
                "end": float(w.get("end", 0)),
            })
        else:
            out.append({"word": str(w).strip(), "start": 0.0, "end": 0.0})
    return [x for x in out if x["word"]]


def _utterances_from_words(words: list[dict], pause_threshold_sec: float = 0.5) -> list[list[dict]]:
    """Split words into utterances by pause threshold."""
    if not words:
        return []
    utts = []
    current = [words[0]]
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i - 1]["end"]
        if gap >= pause_threshold_sec:
            utts.append(current)
            current = [words[i]]
        else:
            current.append(words[i])
    if current:
        utts.append(current)
    return utts


def _speech_duration_excl_pauses(words: list[dict], min_pause: float = MIN_PAUSE_GAP) -> tuple[float, float, list[float]]:
    """Total duration, speech duration (excl pauses), and list of pause durations."""
    if not words:
        return 0.0, 0.0, []
    t_start = words[0]["start"]
    t_end = words[-1]["end"]
    total = t_end - t_start
    pauses = []
    speech = 0.0
    for w in words:
        speech += w["end"] - w["start"]
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i - 1]["end"]
        if gap >= min_pause:
            pauses.append(gap)
    return total, speech, pauses


def _phone_count_from_phonetic(phonetic_data: dict[str, Any], word_count: int) -> int | None:
    """Phone count from phonetic cache; if stub, return None (caller will use word-based proxy)."""
    if phonetic_data.get("stub") or not phonetic_data.get("phones"):
        return None
    phones = phonetic_data["phones"]
    if isinstance(phones[0], (list, tuple)):
        return len(phones)
    return len(phones)


def compute_layer1_features(
    transcript: dict[str, Any],
    phonetic_data: dict[str, Any] | None = None,
    expected_entities: list[str] | None = None,
    spacy_model: str = "en_core_web_sm",
    use_semantic_distance: bool = True,
) -> dict[str, float]:
    """
    Compute all 20 Layer 1 proxy features. Returns dict with LAYER1_NAMES keys.
    Uses float('nan') for missing/undefined values.
    """
    words = _get_words_with_times(transcript)
    text = transcript.get("text") or " ".join(w["word"] for w in words)
    phonetic_data = phonetic_data or {}

    total_dur, speech_dur, pause_durations = _speech_duration_excl_pauses(words)
    utts = _utterances_from_words(words)

    # --- A. Speech rate & timing ---
    n_words = len(words)
    duration_min = total_dur / 60.0 if total_dur > 0 else 0.0
    wpm = (n_words / duration_min) if duration_min > 0 else float("nan")

    phone_count = _phone_count_from_phonetic(phonetic_data, n_words)
    if phone_count is not None and speech_dur > 0:
        articulation_rate = phone_count / speech_dur
    else:
        # Proxy: syllables ~ words * 1.4, treat as "phone-like" for rate
        articulation_rate = (n_words * 1.4 / speech_dur) if speech_dur > 0 else float("nan")

    mean_pause = float(np.mean(pause_durations)) if pause_durations else 0.0
    pause_to_speech = (sum(pause_durations) / speech_dur) if speech_dur > 0 else (float("nan") if pause_durations else 0.0)

    initial_latency = words[0]["start"] if words else float("nan")

    # --- B. Utterance length & fragmentation ---
    utt_lengths = [len(u) for u in utts]
    mlu = float(np.mean(utt_lengths)) if utt_lengths else float("nan")
    median_utt = float(np.median(utt_lengths)) if utt_lengths else float("nan")

    # Clause count per utterance: use sentence split as proxy (spaCy sents); then clauses ~ subj/verb chunks. Simplified: sents per utt.
    try:
        nlp = get_nlp(spacy_model)
        doc = nlp(text)
        sents = list(doc.sents)
        clause_per_utt = len(sents) / len(utts) if utts and sents else float("nan")
        if utts and not sents:
            clause_per_utt = 1.0  # one "clause" per utt if no sentence boundary
    except Exception:
        clause_per_utt = 1.0 if utts else float("nan")

    # Abandoned utterance: heuristic = very short utterances (e.g. < 3 words) as fraction
    abandoned = sum(1 for u in utt_lengths if u < 3) / len(utt_lengths) if utt_lengths else 0.0
    utterance_count = float(len(utts))

    # --- C. Lexical ---
    try:
        nlp = get_nlp(spacy_model)
        doc = nlp(text)
        open_closed = open_class_closed_class_ratio(doc)
        content_density = content_word_density(doc)
        ttr = type_token_ratio(doc)
        pron_ratio = pronoun_ratio(doc)
        func_omission = function_word_omission_heuristic(doc)
        dep_completeness = dependency_parse_completeness(doc)
        morph_error = morphological_error_heuristic(doc)
        nouns = get_nouns_and_proper_nouns(doc)
    except Exception as e:
        logger.warning("spaCy processing failed: %s", e)
        open_closed = content_density = ttr = pron_ratio = float("nan")
        func_omission = dep_completeness = morph_error = 0.0
        nouns = []

    open_closed = open_closed if open_closed is not None else float("nan")
    content_density = content_density if content_density is not None else float("nan")
    ttr = ttr if ttr is not None else float("nan")
    pron_ratio = pron_ratio if pron_ratio is not None else float("nan")
    dep_completeness = dep_completeness if dep_completeness is not None else float("nan")

    # --- E. Word-finding pause & filled pause ---
    filled_count = sum(1 for w in words if w["word"].lower().strip() in FILLED_PAUSES)
    filled_pause_freq = filled_count / len(words) if words else 0.0

    # Word-finding pause: mean duration of pauses >= WORD_FINDING_PAUSE_THRESHOLD
    wf_pauses = [p for p in pause_durations if p >= WORD_FINDING_PAUSE_THRESHOLD]
    word_finding_pause_dur = float(np.mean(wf_pauses)) if wf_pauses else 0.0

    # --- Semantic distance ---
    semantic_dist = float("nan")
    if use_semantic_distance and (nouns or expected_entities):
        semantic_dist = _semantic_distance_to_expected(nouns, expected_entities or [])

    return {
        LAYER1_NAMES[0]: wpm,
        LAYER1_NAMES[1]: articulation_rate,
        LAYER1_NAMES[2]: mean_pause,
        LAYER1_NAMES[3]: pause_to_speech,
        LAYER1_NAMES[4]: initial_latency,
        LAYER1_NAMES[5]: mlu,
        LAYER1_NAMES[6]: median_utt,
        LAYER1_NAMES[7]: clause_per_utt,
        LAYER1_NAMES[8]: abandoned,
        LAYER1_NAMES[9]: utterance_count,
        LAYER1_NAMES[10]: open_closed,
        LAYER1_NAMES[11]: content_density,
        LAYER1_NAMES[12]: ttr,
        LAYER1_NAMES[13]: pron_ratio,
        LAYER1_NAMES[14]: func_omission,
        LAYER1_NAMES[15]: morph_error,
        LAYER1_NAMES[16]: dep_completeness,
        LAYER1_NAMES[17]: word_finding_pause_dur,
        LAYER1_NAMES[18]: filled_pause_freq,
        LAYER1_NAMES[19]: semantic_dist,
    }


def _semantic_distance_to_expected(nouns: list[str], expected_entities: list[str]) -> float:
    """
    Min cosine distance between transcript nouns and expected entities.
    Returns NaN if sentence-transformers not available or no entities.
    """
    if not expected_entities or not nouns:
        return float("nan")
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        # TODO: stub when sentence-transformers not installed
        return float("nan")
    try:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        noun_embs = model.encode(list(set(nouns)))
        entity_embs = model.encode(list(set(expected_entities)))
        sim = cosine_similarity(noun_embs, entity_embs)
        max_sim = float(np.max(sim))
        return 1.0 - max_sim  # distance
    except Exception as e:
        logger.warning("Semantic distance failed: %s", e)
        return float("nan")
