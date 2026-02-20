"""Tokenization, POS tagging, dependency parsing helpers (spaCy)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Open-class POS (content): NOUN, VERB, ADJ, ADV
OPEN_CLASS_POS = {"NOUN", "VERB", "ADJ", "ADV"}
# Closed-class (function): DET, ADP, PRON, AUX, PART, CCONJ, SCONJ, etc.
CLOSED_CLASS_POS = {"DET", "ADP", "PRON", "AUX", "PART", "CCONJ", "SCONJ", "INTJ", "SYM", "PUNCT", "X", "SPACE"}
# For pronoun ratio we use PRON only
# For function word omission: DET, ADP, AUX

_nlp = None


def get_nlp(model: str = "en_core_web_sm"):
    """Lazy-load spaCy model."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
        except ImportError as e:
            raise ImportError(
                "spaCy is not installed. Install with: pip install spacy && python -m spacy download en_core_web_sm"
            ) from e
        try:
            _nlp = spacy.load(model)
        except OSError:
            logger.warning("spaCy model %s not found. Run: python -m spacy download %s", model, model)
            raise
    return _nlp


def parse_text(text: str, spacy_model: str = "en_core_web_sm") -> Any:
    """Return spaCy Doc for text."""
    nlp = get_nlp(spacy_model)
    return nlp(text)


def open_class_closed_class_ratio(doc: Any) -> float | None:
    """Open-class to closed-class ratio. Returns None if no closed-class tokens."""
    open_c = sum(1 for t in doc if t.pos_ in OPEN_CLASS_POS and not t.is_punct)
    closed_c = sum(1 for t in doc if t.pos_ in CLOSED_CLASS_POS and not t.is_space)
    if closed_c == 0:
        return None
    return open_c / closed_c


def content_word_density(doc: Any) -> float | None:
    """Content words / total tokens (excl. space/punct)."""
    total = sum(1 for t in doc if not t.is_space and not t.is_punct)
    if total == 0:
        return None
    content = sum(1 for t in doc if t.pos_ in OPEN_CLASS_POS)
    return content / total


def pronoun_ratio(doc: Any) -> float | None:
    """PRON tokens / total tokens."""
    total = sum(1 for t in doc if not t.is_space and not t.is_punct)
    if total == 0:
        return None
    pron = sum(1 for t in doc if t.pos_ == "PRON")
    return pron / total


def type_token_ratio(doc: Any, use_lemmas: bool = True) -> float | None:
    """Unique lemmas (or words) / total tokens. Excl. punct/space."""
    tokens = [t.lemma_.lower() if use_lemmas else t.text.lower() for t in doc if not t.is_space and not t.is_punct]
    if not tokens:
        return None
    return len(set(tokens)) / len(tokens)


def moving_average_ttr(doc: Any, window: int = 50, use_lemmas: bool = True) -> float | None:
    """Moving average TTR over windows. Optional; use window=50 as default."""
    tokens = [t.lemma_.lower() if use_lemmas else t.text.lower() for t in doc if not t.is_space and not t.is_punct]
    if not tokens or len(tokens) < window:
        return type_token_ratio(doc, use_lemmas)
    ttrs = []
    for i in range(len(tokens) - window + 1):
        w = tokens[i : i + window]
        ttrs.append(len(set(w)) / len(w))
    return sum(ttrs) / len(ttrs) if ttrs else None


def function_word_omission_heuristic(doc: Any) -> float:
    """
    Heuristic: unusually low rate of DET/ADP/AUX relative to content words.
    Returns a rate in [0,1]; higher = more omission (fewer function words than expected).
    Expected rough ratio: ~1 function (DET/ADP/AUX) per 1–2 content words.
    If ratio is very low, return higher omission score.
    """
    content = sum(1 for t in doc if t.pos_ in OPEN_CLASS_POS and not t.is_punct)
    func = sum(1 for t in doc if t.pos_ in {"DET", "ADP", "AUX"} and not t.is_punct)
    if content == 0:
        return 0.0
    # Expected: func/content around 0.4–0.6 in normal speech; lower -> omission
    ratio = func / content
    # Clip: if ratio < 0.2, score 1.0; if ratio >= 0.5, score 0.0; linear in between
    if ratio >= 0.5:
        return 0.0
    if ratio <= 0.2:
        return 1.0
    return 1.0 - (ratio - 0.2) / 0.3


def dependency_parse_completeness(doc: Any) -> float | None:
    """
    Fraction of tokens (excl. punct/space) that have a head (are in the tree).
    Root has head pointing to self. So completeness = share of non-root tokens with head != self,
    or alternatively: 1 - (orphans / total). We use: proportion of tokens with dep_ not in ['', 'ROOT'] or head.
    Actually: in spaCy every token has a head. Completeness can mean: proportion of tokens
    that are not 'dep' (unclassified) or that have a valid dependency label.
    Simplified: 1 minus (proportion of tokens with dep_ == 'dep' or without governor).
    """
    total = sum(1 for t in doc if not t.is_space and not t.is_punct)
    if total == 0:
        return None
    # "Complete" = has clear dependency; "dep" is the catch-all. So completeness = 1 - (dep_count/total)
    dep_orphan = sum(1 for t in doc if not t.is_space and not t.is_punct and (t.dep_ == "dep" or t.dep_ == ""))
    return 1.0 - (dep_orphan / total)


def morphological_error_heuristic(doc: Any) -> float:
    """
    Lightweight rule set for morphological errors (e.g. verb tense). Placeholder + TODO for later.
    Returns rate in [0,1].
    """
    # TODO: Add rules for verb tense mismatch, agreement, etc.
    errors = 0
    total = sum(1 for t in doc if t.pos_ in {"VERB", "NOUN"} and not t.is_punct)
    if total == 0:
        return 0.0
    # Placeholder: no rules yet
    return float(errors) / total


def get_nouns_and_proper_nouns(doc: Any) -> list[str]:
    """Return list of noun and proper noun lemmas (or text) for semantic distance."""
    return [t.lemma_.lower() for t in doc if t.pos_ in {"NOUN", "PROPN"} and not t.is_punct]
