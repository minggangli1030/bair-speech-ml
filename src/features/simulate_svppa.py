"""
svPPA simulation: degrade transcript to mimic semantic impairment.

Used to generate synthetic svPPA-like samples from nfvPPA/lvPPA transcripts
by reducing semantic specificity (anomia-like) and adding empty speech.
"""

from __future__ import annotations

import re
from typing import Any

# Content-word replacement (anomia simulation): specific -> generic
REPLACEMENTS = [
    (r"\b(boy|girl|kid|child|son|daughter)\b", "person"),
    (r"\b(man|woman|father|mother|dad|mom|lady|guy)\b", "person"),
    (r"\b(dog|puppy)\b", "thing"),
    (r"\b(kite|string|line)\b", "thing"),
    (r"\b(blanket|rug|mat)\b", "thing"),
    (r"\b(basket|picnic basket)\b", "thing"),
    (r"\b(sandwich|food|lunch)\b", "thing"),
    (r"\b(juice|soda|drink)\b", "thing"),
    (r"\b(book|magazine)\b", "thing"),
    (r"\b(ball|tree|grass|sky|lake|water|pond|boat|house|home|path|road|trail)\b", "thing"),
]

# Filled pauses to insert (empty speech)
FILLERS = ["um", "uh", "you know"]


def degrade_transcript(
    text: str,
    replace_ratio: float = 0.7,
    insert_filler_ratio: float = 0.15,
    seed: int | None = None,
) -> str:
    """
    Degrade transcript to simulate svPPA semantic impairment.

    - replace_ratio: fraction of content-word replacements to apply (0–1)
    - insert_filler_ratio: approximate fraction of token positions after which to insert a filler
    """
    if not (text or "").strip():
        return text
    if seed is not None:
        import random
        random.seed(seed)

    text = " " + (text or "").strip() + " "
    tokens = re.findall(r"[a-zA-Z]+", text)

    # Apply replacements with probability replace_ratio per pattern
    import random
    rng = random.Random(seed)
    for pattern, repl in REPLACEMENTS:
        def sub(m):
            return repl if rng.random() < replace_ratio else m.group(0)
        text = re.sub(pattern, sub, text, flags=re.IGNORECASE)

    # Re-tokenize and insert fillers
    tokens = re.findall(r"[a-zA-Z]+", text)
    out = []
    for i, t in enumerate(tokens):
        out.append(t)
        if i < len(tokens) - 1 and rng.random() < insert_filler_ratio:
            out.append(rng.choice(FILLERS))
    return " ".join(out)


def transcript_dict_from_text(text: str) -> dict[str, Any]:
    """Build minimal transcript dict {text, words} for downstream S_sem computation."""
    tokens = re.findall(r"[a-zA-Z]+", (text or "").strip())
    words = [{"word": w, "start": i * 0.5, "end": i * 0.5 + 0.3} for i, w in enumerate(tokens)]
    return {"text": " ".join(tokens), "words": words}
