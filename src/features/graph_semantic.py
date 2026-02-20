"""
Spatiosemantic scene graph: Level 2 semantic scoring for PPA picture description.

Integrates heuristic graph extraction (no LLM) with gold graph evaluation.
Produces S_sem (semantic adequacy index) in [0, 1] from transcript.
"""

from __future__ import annotations

import json
import re
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# -------------------------
# Canonicalization maps (Picnic scene)
# -------------------------

ENTITY_CANON = {
    "boy": {"boy", "kid", "child", "son"},
    "girl": {"girl", "kid", "child", "daughter"},
    "man": {"man", "father", "dad", "guy"},
    "woman": {"woman", "mother", "mom", "lady"},
    "dog": {"dog", "puppy"},
    "kite": {"kite"},
    "kite string": {"string", "kite string", "line"},
    "blanket": {"blanket", "rug", "mat"},
    "picnic basket": {"basket"},
    "food": {"food", "sandwich", "lunch"},
    "drink": {"drink", "juice", "soda"},
    "book": {"book", "magazine"},
    "ball": {"ball"},
    "tree": {"tree"},
    "grass": {"grass", "ground"},
    "sky": {"sky"},
    "lake": {"lake", "water", "pond"},
    "boat": {"boat"},
    "house": {"house", "home"},
    "path": {"path", "road", "trail"},
}

ALLOWED_NODE_TYPES = {"person", "animal", "object", "place", "group", "unknown"}


def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _guess_type(canon_label: str) -> str:
    if canon_label in {"boy", "girl", "man", "woman"}:
        return "person"
    if canon_label == "dog":
        return "animal"
    if canon_label in {"grass", "sky", "lake", "path"}:
        return "place"
    return "object"


def _make_node_id(canon_label: str) -> str:
    return canon_label.replace(" ", "_") + "_1"


def _find_span_evidence(spans: list[dict[str, Any]], keywords: list[str], max_hits: int = 2) -> list[dict[str, Any]]:
    kw = [k.lower() for k in keywords if k]
    out = []
    for s in spans:
        txt = (s.get("text") or "").lower()
        if any(k in txt for k in kw):
            out.append({"start": s["start"], "end": s["end"], "text": s["text"]})
        if len(out) >= max_hits:
            break
    return out


# -------------------------
# Transcript -> participant format
# -------------------------


def transcript_to_participant_format(transcript: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """
    Convert ASR transcript (text, words) to participant format (full_text, spans).
    Words are grouped into spans by segments, or by pause threshold if no segments.
    """
    words = transcript.get("words") or []
    text = transcript.get("text") or " ".join(w.get("word", "") for w in words)

    if not words:
        return text, []

    segments = transcript.get("segments") or []
    if segments and any(seg.get("words") for seg in segments):
        spans = []
        for seg in segments:
            seg_words = seg.get("words") or []
            if not seg_words and seg.get("text", "").strip():
                start = float(seg.get("start", 0))
                end = float(seg.get("end", 0))
                spans.append({"start": start, "end": end, "text": seg["text"].strip()})
            else:
                for w in seg_words:
                    word = (w.get("word") or "").strip()
                    if not word:
                        continue
                    spans.append({
                        "start": float(w.get("start", 0)),
                        "end": float(w.get("end", 0)),
                        "text": word,
                    })
    else:
        pause_threshold = 0.5
        spans = []
        current_start = words[0]["start"]
        current_texts = []
        for i, w in enumerate(words):
            word = (w.get("word") or "").strip()
            if not word:
                continue
            gap = float(w.get("start", 0)) - float(words[i - 1].get("end", 0)) if i > 0 else 0
            if gap >= pause_threshold and current_texts:
                spans.append({
                    "start": current_start,
                    "end": float(words[i - 1].get("end", current_start)),
                    "text": " ".join(current_texts),
                })
                current_start = float(w.get("start", 0))
                current_texts = [word]
            else:
                current_texts.append(word)
        if current_texts:
            spans.append({
                "start": current_start,
                "end": float(words[-1].get("end", current_start)),
                "text": " ".join(current_texts),
            })

    return text, spans


# -------------------------
# Heuristic graph extraction
# -------------------------


def _heuristic_extract(full_text: str, spans: list[dict[str, Any]], allowed_labels: list[str]) -> dict[str, Any]:
    t = _norm(full_text)
    found: dict[str, dict[str, Any]] = {}

    for canon, variants in ENTITY_CANON.items():
        for v in set(variants) | {canon}:
            if re.search(rf"\b{re.escape(v)}\b", t):
                nid = _make_node_id(canon)
                ntype = _guess_type(canon)
                found[canon] = {
                    "id": nid,
                    "type": ntype if ntype in ALLOWED_NODE_TYPES else "unknown",
                    "label": canon,
                }
                break

    nodes = []
    for label, nd in found.items():
        nodes.append({
            **nd,
            "evidence": _find_span_evidence(spans, [label] + list(ENTITY_CANON.get(label, []))),
        })

    def has(label: str) -> bool:
        return label in found

    def nid(label: str) -> str:
        return found[label]["id"]

    edges = []
    attributes = []

    def add_edge(subj_label: str, rel: str, obj_label: str, ev_kw: list[str]):
        edges.append({
            "subj": nid(subj_label),
            "rel": rel,
            "obj": nid(obj_label),
            "evidence": _find_span_evidence(spans, ev_kw),
        })

    def add_attr(node_label: str, attr: str, value: str, ev_kw: list[str]):
        attributes.append({
            "node": nid(node_label),
            "attr": attr,
            "value": value,
            "evidence": _find_span_evidence(spans, ev_kw),
        })

    if any(w in t for w in ["hold", "holding", "carry", "carrying", "grab", "grabbing", "has"]):
        for person in ["boy", "girl", "man", "woman"]:
            if has(person) and has("kite"):
                if has("kite string") and ("string" in t or "line" in t):
                    add_edge(person, "holding", "kite string", [person, "hold", "string", "line", "kite"])
                else:
                    add_edge(person, "holding", "kite", [person, "hold", "kite"])

    if has("kite string") and has("kite") and re.search(r"\b(tied|attached|connected)\b", t):
        add_edge("kite string", "attached_to", "kite", ["string", "kite", "tied", "attached", "connected"])

    if has("blanket") and re.search(r"\b(sit|sits|sitting)\b", t):
        for person in ["woman", "man", "boy", "girl"]:
            if has(person):
                add_edge(person, "sitting_on", "blanket", [person, "sit", "blanket"])

    if has("book") and re.search(r"\b(read|reading|reads)\b", t):
        for person in ["woman", "man", "boy", "girl"]:
            if has(person):
                add_edge(person, "reading", "book", [person, "read", "book", "magazine"])
                break

    if has("dog") and has("ball") and re.search(r"\b(chase|chasing|running after)\b", t):
        add_edge("dog", "chasing", "ball", ["dog", "chase", "ball", "running"])

    if has("boat") and has("lake") and re.search(r"\b(on|in)\b", t):
        add_edge("boat", "on", "lake", ["boat", "lake", "water", "on", "in"])

    if has("dog") and re.search(r"\b(run|running)\b", t):
        add_attr("dog", "activity", "running", ["dog", "run", "running"])
    if has("kite") and re.search(r"\b(fly|flying)\b", t):
        add_attr("kite", "state", "flying", ["kite", "fly", "flying"])

    return {"scene": "picnic", "nodes": nodes, "edges": edges, "attributes": attributes}


# -------------------------
# Graph scoring (completeness = S_sem)
# -------------------------


def _graph_from_spec(spec: dict[str, Any]):
    """Build NetworkX MultiDiGraph from JSON spec."""
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx is required for graph scoring. Install with: pip install networkx") from None

    G = nx.MultiDiGraph(scene=spec.get("scene", ""))

    def ensure_node(nid: str):
        if nid not in G:
            G.add_node(nid, type="unknown", label=_norm(nid), attrs={})

    for n in spec.get("nodes", []):
        nid = n["id"]
        G.add_node(
            nid,
            type=n.get("type", "unknown"),
            label=_norm(n.get("label", nid)),
            attrs={},
        )

    for e in spec.get("edges", []):
        s, o = e["subj"], e["obj"]
        r = _norm(e["rel"])
        ensure_node(s)
        ensure_node(o)
        G.add_edge(s, o, rel=r, kind="edge")

    for a in spec.get("attributes", []):
        nid = a["node"]
        ensure_node(nid)
        attr = _norm(a["attr"])
        val = _norm(str(a["value"]))
        G.nodes[nid]["attrs"].setdefault(attr, set()).add(val)

    return G


def _edges_as_set(G) -> set[tuple[str, str, str]]:
    out: set[tuple[str, str, str]] = set()
    for u, v, data in G.edges(data=True):
        subj = G.nodes[u].get("label", _norm(u))
        obj = G.nodes[v].get("label", _norm(v))
        rel = data.get("rel", "")
        out.add((subj, rel, obj))
    return out


def _attrs_as_set(G) -> set[tuple[str, str, str]]:
    out: set[tuple[str, str, str]] = set()
    for nid, data in G.nodes(data=True):
        nlab = data.get("label", _norm(nid))
        attrs = data.get("attrs", {}) or {}
        for attr, vals in attrs.items():
            for v in vals:
                out.add((nlab, attr, v))
    return out


def _score_graphs(gold: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    """Score predicted graph vs gold. Returns completeness (S_sem) and detailed metrics."""
    gold_G = _graph_from_spec(gold)
    pred_G = _graph_from_spec(pred)

    ge = _edges_as_set(gold_G)
    pe = _edges_as_set(pred_G)
    ga = _attrs_as_set(gold_G)
    pa = _attrs_as_set(pred_G)

    gold_units = ge | ga
    pred_units = pe | pa
    completeness = (len(gold_units & pred_units) / len(gold_units)) if gold_units else 0.0

    tp_e = len(ge & pe)
    fp_e = len(pe - ge)
    fn_e = len(ge - pe)
    prec_e = tp_e / (tp_e + fp_e) if (tp_e + fp_e) else 0.0
    rec_e = tp_e / (tp_e + fn_e) if (tp_e + fn_e) else 0.0
    f1_e = (2 * prec_e * rec_e) / (prec_e + rec_e) if (prec_e + rec_e) else 0.0

    tp_a = len(ga & pa)
    fp_a = len(pa - ga)
    fn_a = len(ga - pa)
    prec_a = tp_a / (tp_a + fp_a) if (tp_a + fp_a) else 0.0
    rec_a = tp_a / (tp_a + fn_a) if (tp_a + fn_a) else 0.0
    f1_a = (2 * prec_a * rec_a) / (prec_a + rec_a) if (prec_a + rec_a) else 0.0

    return {
        "completeness": completeness,
        "s_sem": completeness,
        "edges": {"precision": prec_e, "recall": rec_e, "f1": f1_e, "tp": tp_e, "fp": fp_e, "fn": fn_e},
        "attributes": {"precision": prec_a, "recall": rec_a, "f1": f1_a, "tp": tp_a, "fp": fp_a, "fn": fn_a},
    }


# -------------------------
# Public API
# -------------------------


def compute_s_sem(
    transcript: dict[str, Any],
    gold_graph_path: Path | str | None = None,
) -> tuple[float, dict[str, Any]]:
    """
    Compute S_sem (semantic adequacy index) from transcript via spatiosemantic graph.

    Level 1: transcript -> participant format (full_text, spans)
    Level 2: heuristic graph extraction -> score vs gold -> completeness

    Returns:
        s_sem: float in [0, 1], semantic adequacy (graph completeness)
        details: dict with completeness, edges, attributes metrics
    """
    default_gold = Path(__file__).resolve().parent.parent.parent / "data" / "graph" / "picnic_gold_star.json"
    path = Path(gold_graph_path) if gold_graph_path else default_gold

    if not path.exists():
        logger.warning("Gold graph not found at %s, returning s_sem=NaN", path)
        return float("nan"), {}

    try:
        gold = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load gold graph %s: %s", path, e)
        return float("nan"), {}

    full_text, spans = transcript_to_participant_format(transcript)
    allowed_labels = [n["label"] for n in gold.get("nodes", [])]
    pred = _heuristic_extract(full_text, spans, allowed_labels)
    report = _score_graphs(gold, pred)

    return report["s_sem"], report
