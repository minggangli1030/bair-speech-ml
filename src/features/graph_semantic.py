"""
Spatiosemantic scene graph: Level 2 semantic scoring for PPA picture description.

Integrates heuristic graph extraction (no LLM) with gold graph evaluation.
Produces S_sem (semantic adequacy index) in [0, 1] from transcript.

Key features:
- Content Unit (CU) scoring aligned with clinical practice (Nicholas & Brookshire 1993)
- Improved edge detection with spatial/action relation patterns
- Patient profile generation for interpretability
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
    "family": {"family", "people", "everyone", "they", "them"},
}

ALLOWED_NODE_TYPES = {"person", "animal", "object", "place", "group", "unknown"}

# Spatial relation patterns for broader edge detection
SPATIAL_RELATIONS = {"on", "in", "near", "next to", "beside", "behind", "by", "at", "under", "above", "over"}

# Action verb patterns (lemmas) for relation detection
ACTION_VERBS = {
    "hold": ["hold", "holding", "holds", "held", "has", "have", "got"],
    "sit": ["sit", "sits", "sitting", "sat", "seated"],
    "read": ["read", "reads", "reading"],
    "chase": ["chase", "chases", "chasing", "chased", "running after", "going after", "after"],
    "fly": ["fly", "flies", "flying", "flew"],
    "run": ["run", "runs", "running", "ran"],
    "attach": ["attached", "tied", "connected", "connect"],
    "picnic": ["picnic", "picnicking", "eating", "lunch", "meal"],
}

# Map from action to canonical relation name
ACTION_TO_REL = {
    "hold": "holding",
    "sit": "sitting_on",
    "read": "reading",
    "chase": "chasing",
    "attach": "attached_to",
}


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
    if canon_label == "family":
        return "group"
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
# Gold graph validation
# -------------------------


def validate_gold_graph(gold_graph_path: Path | str) -> dict[str, Any]:
    """
    Validate that a gold graph JSON has the required structure.
    
    Checks:
    - Has nodes, edges, attributes fields
    - Has content_units field with valid references
    - All content_unit edge refs point to valid node IDs
    - All content_unit attr refs point to valid node IDs
    
    Returns:
        dict with "valid": bool, "errors": list[str], "warnings": list[str]
    """
    path = Path(gold_graph_path)
    errors = []
    warnings = []
    
    if not path.exists():
        return {"valid": False, "errors": [f"File not found: {path}"], "warnings": []}
    
    try:
        gold = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return {"valid": False, "errors": [f"Failed to parse JSON: {e}"], "warnings": []}
    
    # Check required fields
    for field in ["nodes", "edges", "attributes"]:
        if field not in gold:
            errors.append(f"Missing required field: {field}")
        elif not isinstance(gold[field], list):
            errors.append(f"Field '{field}' must be a list")
    
    if "content_units" not in gold:
        warnings.append("Missing 'content_units' field - CU scoring will not work")
    elif not isinstance(gold["content_units"], list):
        errors.append("Field 'content_units' must be a list")
    
    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}
    
    # Build set of valid node IDs
    node_ids = {n["id"] for n in gold.get("nodes", []) if "id" in n}
    
    # Validate edges reference valid nodes
    for i, e in enumerate(gold.get("edges", [])):
        if "subj" not in e or "obj" not in e:
            errors.append(f"Edge {i} missing 'subj' or 'obj'")
        else:
            if e["subj"] not in node_ids:
                warnings.append(f"Edge {i} references unknown subj node: {e['subj']}")
            if e["obj"] not in node_ids:
                warnings.append(f"Edge {i} references unknown obj node: {e['obj']}")
    
    # Validate attributes reference valid nodes
    for i, a in enumerate(gold.get("attributes", [])):
        if "node" not in a:
            errors.append(f"Attribute {i} missing 'node' field")
        elif a["node"] not in node_ids:
            warnings.append(f"Attribute {i} references unknown node: {a['node']}")
    
    # Validate content_units
    for i, cu in enumerate(gold.get("content_units", [])):
        if "id" not in cu:
            errors.append(f"Content unit {i} missing 'id'")
        if "kind" not in cu:
            errors.append(f"Content unit {i} missing 'kind'")
        else:
            kind = cu["kind"]
            if kind == "edge":
                for field in ["subj", "obj"]:
                    if field not in cu:
                        errors.append(f"Content unit {cu.get('id', i)} (edge) missing '{field}'")
                    elif cu[field] not in node_ids:
                        warnings.append(f"Content unit {cu.get('id', i)} references unknown {field}: {cu[field]}")
            elif kind == "attr":
                if "node" not in cu:
                    errors.append(f"Content unit {cu.get('id', i)} (attr) missing 'node'")
                elif cu["node"] not in node_ids:
                    warnings.append(f"Content unit {cu.get('id', i)} references unknown node: {cu['node']}")
            else:
                warnings.append(f"Content unit {cu.get('id', i)} has unknown kind: {kind}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "node_count": len(node_ids),
        "edge_count": len(gold.get("edges", [])),
        "attr_count": len(gold.get("attributes", [])),
        "cu_count": len(gold.get("content_units", [])),
    }


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
# Improved heuristic graph extraction
# -------------------------


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words."""
    return re.findall(r"[a-z]+", text.lower())


def _find_entity_positions(tokens: list[str], found_entities: dict[str, dict]) -> dict[str, list[int]]:
    """
    Find token positions where each entity is mentioned.
    Handles multi-word entities like 'kite string' by checking bigrams AND single-word variants.
    Prioritizes longer matches (e.g., 'kite string' over 'kite').
    """
    positions = {label: [] for label in found_entities}
    matched_positions = set()  # Track positions already matched to avoid double-counting
    
    # First pass: multi-word entity bigrams (e.g., "kite string")
    for i in range(len(tokens) - 1):
        bigram = tokens[i] + " " + tokens[i + 1]
        for canon, variants in ENTITY_CANON.items():
            if canon in found_entities and " " in canon:
                if bigram in variants or bigram == canon:
                    positions[canon].append(i)
                    matched_positions.add(i)
                    matched_positions.add(i + 1)
    
    # Second pass: all entities including single-word variants of multi-word entities
    for i, tok in enumerate(tokens):
        if i in matched_positions:
            continue
        for canon, variants in ENTITY_CANON.items():
            if canon in found_entities:
                # Check if this token is a variant (including single-word variants like "string" for "kite string")
                if tok in variants or tok == canon or tok == canon.split()[0]:
                    positions[canon].append(i)
                    matched_positions.add(i)
                    break
    
    return positions


def _detect_spatial_relations(
    tokens: list[str],
    entity_positions: dict[str, list[int]],
    found_entities: dict[str, dict],
) -> list[dict[str, Any]]:
    """
    Detect spatial relations between entities.
    Pattern: entity1 + spatial_preposition + entity2 (within 5-word window).
    """
    edges = []
    entities = list(found_entities.keys())
    
    for i, tok in enumerate(tokens):
        if tok in SPATIAL_RELATIONS or tok == "next" and i + 1 < len(tokens) and tokens[i + 1] == "to":
            rel = tok if tok != "next" else "next to"
            window_start = max(0, i - 5)
            window_end = min(len(tokens), i + 6)
            
            subj_cands = []
            obj_cands = []
            for ent in entities:
                for pos in entity_positions.get(ent, []):
                    if window_start <= pos < i:
                        subj_cands.append((ent, pos))
                    elif i < pos < window_end:
                        obj_cands.append((ent, pos))
            
            if subj_cands and obj_cands:
                subj = max(subj_cands, key=lambda x: x[1])[0]
                obj = min(obj_cands, key=lambda x: x[1])[0]
                if subj != obj:
                    edges.append({
                        "subj": found_entities[subj]["id"],
                        "rel": _norm(rel),
                        "obj": found_entities[obj]["id"],
                        "evidence": [],
                    })
    
    return edges


def _find_verb_positions(tokens: list[str], verb_patterns: list[str]) -> list[int]:
    """Find all positions of verb patterns in token list."""
    positions = []
    for i, tok in enumerate(tokens):
        if tok in verb_patterns:
            positions.append(i)
        # Handle multi-word patterns like "running after"
        if i + 1 < len(tokens):
            bigram = tok + " " + tokens[i + 1]
            if bigram in verb_patterns:
                positions.append(i)
    return positions


def _find_nearest_entity_before(pos: int, entity_positions: dict[str, list[int]], window: int = 5) -> str | None:
    """Find the entity closest to (but before) position pos, within window."""
    best_ent = None
    best_pos = -1
    for ent, positions in entity_positions.items():
        for p in positions:
            if pos - window <= p < pos and p > best_pos:
                best_pos = p
                best_ent = ent
    return best_ent


def _find_nearest_entity_after(pos: int, entity_positions: dict[str, list[int]], window: int = 5) -> str | None:
    """Find the entity closest to (but after) position pos, within window."""
    best_ent = None
    best_pos = float("inf")
    for ent, positions in entity_positions.items():
        for p in positions:
            if pos < p <= pos + window and p < best_pos:
                best_pos = p
                best_ent = ent
    return best_ent


def _detect_action_relations(
    tokens: list[str],
    entity_positions: dict[str, list[int]],
    found_entities: dict[str, dict],
    full_text: str,
    spans: list[dict],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Detect action relations and attributes using DIRECTIONAL word order.
    
    For each verb pattern found:
    1. Find the nearest entity BEFORE the verb (subject)
    2. Find the nearest entity AFTER the verb (object)
    3. Create edge (subj, rel, obj) respecting actual transcript order
    
    This means "dog chasing ball" -> (dog, chasing, ball)
    But "ball chasing dog" -> (ball, chasing, dog) which won't match the gold CU.
    """
    edges = []
    attributes = []
    t = _norm(full_text)
    
    def nid(label: str) -> str:
        return found_entities[label]["id"]
    
    def has(label: str) -> bool:
        return label in found_entities
    
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
    
    def add_directional_edge(verb_patterns: list[str], rel: str, ev_kw: list[str], window: int = 5):
        """Add edge based on actual word order around verb."""
        verb_positions = _find_verb_positions(tokens, verb_patterns)
        for vpos in verb_positions:
            subj = _find_nearest_entity_before(vpos, entity_positions, window)
            obj = _find_nearest_entity_after(vpos, entity_positions, window)
            if subj and obj and subj != obj and has(subj) and has(obj):
                add_edge(subj, rel, obj, ev_kw)
    
    # --- Action relations with directional detection ---
    
    # Holding: "X holding Y" -> (X, holding, Y)
    hold_patterns = ACTION_VERBS["hold"]
    if any(w in t for w in hold_patterns):
        add_directional_edge(hold_patterns, "holding", ["hold", "holding", "has", "have"])
    
    # Sitting: "X sitting on Y" -> (X, sitting_on, Y)
    sit_patterns = ACTION_VERBS["sit"]
    if any(w in t for w in sit_patterns):
        add_directional_edge(sit_patterns, "sitting_on", ["sit", "sitting", "sat", "seated"])
    
    # Reading: "X reading Y" -> (X, reading, Y)
    read_patterns = ACTION_VERBS["read"]
    if any(w in t for w in read_patterns):
        add_directional_edge(read_patterns, "reading", ["read", "reading", "reads"])
    
    # Chasing: "X chasing Y" -> (X, chasing, Y)
    chase_patterns = ACTION_VERBS["chase"]
    if any(w in t for w in chase_patterns):
        add_directional_edge(chase_patterns, "chasing", ["chase", "chasing", "after", "running"])
    
    # Attached: "X attached to Y" -> (X, attached_to, Y)
    attach_patterns = ACTION_VERBS["attach"]
    if any(w in t for w in attach_patterns):
        add_directional_edge(attach_patterns, "attached_to", ["attached", "tied", "connected"])
    
    # --- Spatial relations that are less directional (e.g. "kite in sky") ---
    # These use positional detection from _detect_spatial_relations, but we add some here too
    
    # Kite in sky - check word order
    if has("kite") and has("sky"):
        kite_pos = entity_positions.get("kite", [])
        sky_pos = entity_positions.get("sky", [])
        # Check if "kite" appears before "sky" with "in/up/high" between
        for kp in kite_pos:
            for sp in sky_pos:
                if kp < sp:
                    between = tokens[kp:sp+1]
                    if any(w in between for w in ["in", "up", "high", "air", "flying"]):
                        add_edge("kite", "in", "sky", ["kite", "sky", "in", "up", "high", "air"])
                        break
    
    # Boat on lake - check word order
    if has("boat") and has("lake"):
        boat_pos = entity_positions.get("boat", [])
        lake_pos = entity_positions.get("lake", [])
        for bp in boat_pos:
            for lp in lake_pos:
                if bp < lp:
                    between = tokens[bp:lp+1]
                    if any(w in between for w in ["on", "in", "float", "floating"]):
                        add_edge("boat", "on", "lake", ["boat", "lake", "on", "in", "float"])
                        break
                elif lp < bp:
                    between = tokens[lp:bp+1]
                    if any(w in between for w in ["on", "in", "float", "floating"]):
                        add_edge("boat", "on", "lake", ["boat", "lake", "on", "in", "float"])
                        break
    
    # --- Attributes (not directional) ---
    fly_patterns = ACTION_VERBS["fly"]
    if has("kite") and any(w in t for w in fly_patterns):
        add_attr("kite", "state", "flying", ["kite"] + fly_patterns)
    
    run_patterns = ACTION_VERBS["run"]
    if has("dog") and any(w in t for w in run_patterns):
        add_attr("dog", "activity", "running", ["dog"] + run_patterns)
    
    picnic_patterns = ACTION_VERBS["picnic"]
    if has("family") and any(w in t for w in picnic_patterns):
        add_attr("family", "activity", "picnicking", ["family", "picnic", "eating", "lunch"])
    elif any(w in t for w in picnic_patterns):
        if has("family"):
            add_attr("family", "activity", "picnicking", picnic_patterns)
    
    return edges, attributes


def _heuristic_extract(full_text: str, spans: list[dict[str, Any]], allowed_labels: list[str]) -> dict[str, Any]:
    """
    Extract scene graph from transcript using improved heuristics.
    
    Improvements over original:
    1. Broader spatial relation detection (on, in, near, next to, etc.)
    2. Action relation detection with 5-word window co-occurrence
    3. More verb pattern coverage for edge extraction
    """
    t = _norm(full_text)
    tokens = _tokenize(t)
    found: dict[str, dict[str, Any]] = {}

    # Entity extraction (unchanged)
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

    # Find entity positions for relation detection
    entity_positions = _find_entity_positions(tokens, found)
    
    # Detect spatial relations
    spatial_edges = _detect_spatial_relations(tokens, entity_positions, found)
    
    # Detect action relations and attributes
    action_edges, attributes = _detect_action_relations(tokens, entity_positions, found, full_text, spans)
    
    # Combine edges, deduplicating by (subj, rel, obj)
    seen_edges = set()
    edges = []
    for e in action_edges + spatial_edges:
        key = (e["subj"], e["rel"], e["obj"])
        if key not in seen_edges:
            seen_edges.add(key)
            edges.append(e)

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


def _nodes_as_set(G) -> set[str]:
    """Get set of node labels from graph."""
    return {data.get("label", _norm(nid)) for nid, data in G.nodes(data=True)}


def _build_node_id_to_label(gold: dict[str, Any]) -> dict[str, str]:
    """Build mapping from node ID to canonical label."""
    return {n["id"]: _norm(n.get("label", n["id"])) for n in gold.get("nodes", [])}


def _score_content_units(
    gold: dict[str, Any],
    pred_edges: set[tuple[str, str, str]],
    pred_attrs: set[tuple[str, str, str]],
    id_to_label: dict[str, str],
) -> dict[str, Any]:
    """
    Score against the 12 clinical content units (CU_E01-CU_E09, CU_A01-CU_A03).
    
    Returns:
        dict with matched_cus, missed_cus, s_sem_cu, cu_details
    """
    content_units = gold.get("content_units", [])
    if not content_units:
        return {
            "s_sem_cu": float("nan"),
            "matched_cu_count": 0,
            "total_cu_count": 0,
            "matched_cus": [],
            "missed_cus": [],
            "cu_details": [],
        }
    
    matched = []
    missed = []
    details = []
    
    for cu in content_units:
        cu_id = cu.get("id", "")
        kind = cu.get("kind", "")
        is_matched = False
        
        if kind == "edge":
            subj_label = id_to_label.get(cu.get("subj", ""), "")
            obj_label = id_to_label.get(cu.get("obj", ""), "")
            rel = _norm(cu.get("rel", ""))
            gold_tuple = (subj_label, rel, obj_label)
            is_matched = gold_tuple in pred_edges
            details.append({
                "id": cu_id,
                "kind": "edge",
                "gold": f"{subj_label} --{rel}--> {obj_label}",
                "matched": is_matched,
            })
        elif kind == "attr":
            node_label = id_to_label.get(cu.get("node", ""), "")
            attr = _norm(cu.get("attr", ""))
            value = _norm(str(cu.get("value", "")))
            gold_tuple = (node_label, attr, value)
            is_matched = gold_tuple in pred_attrs
            details.append({
                "id": cu_id,
                "kind": "attr",
                "gold": f"{node_label}.{attr}={value}",
                "matched": is_matched,
            })
        else:
            details.append({"id": cu_id, "kind": kind, "gold": str(cu), "matched": False})
        
        if is_matched:
            matched.append(cu_id)
        else:
            missed.append(cu_id)
    
    total = len(content_units)
    s_sem_cu = len(matched) / total if total > 0 else 0.0
    
    return {
        "s_sem_cu": s_sem_cu,
        "matched_cu_count": len(matched),
        "total_cu_count": total,
        "matched_cus": matched,
        "missed_cus": missed,
        "cu_details": details,
    }


def _score_graphs(gold: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    """
    Score predicted graph vs gold. Returns completeness (S_sem) and detailed metrics.
    
    NEW S_sem formula (clinically grounded):
    - s_sem_nodes (0.4 weight): entity recall against all gold nodes
    - s_sem_cu (0.6 weight): content unit coverage (12 clinical CUs)
    
    This makes S_sem more sensitive to relational content, which is the key
    deficit in svPPA (impaired semantic relationships, preserved word forms).
    
    Also returns legacy metrics (edge_completeness, attr_completeness) for
    backward compatibility.
    """
    gold_G = _graph_from_spec(gold)
    pred_G = _graph_from_spec(pred)

    # Node matching (entities mentioned)
    gn = _nodes_as_set(gold_G)
    pn = _nodes_as_set(pred_G)
    matched_nodes = gn & pn
    node_recall = len(matched_nodes) / len(gn) if gn else 0.0
    
    # Edge matching (all relations, for backward compat)
    ge = _edges_as_set(gold_G)
    pe = _edges_as_set(pred_G)
    matched_edges = ge & pe
    edge_recall = len(matched_edges) / len(ge) if ge else 0.0
    
    # Attribute matching (all attrs, for backward compat)
    ga = _attrs_as_set(gold_G)
    pa = _attrs_as_set(pred_G)
    matched_attrs = ga & pa
    attr_recall = len(matched_attrs) / len(ga) if ga else 0.0
    
    # Content unit scoring (clinical CUs)
    id_to_label = _build_node_id_to_label(gold)
    cu_results = _score_content_units(gold, pe, pa, id_to_label)
    s_sem_cu = cu_results["s_sem_cu"]
    
    # NEW S_sem: 0.4 * node_recall + 0.6 * CU_recall
    # Falls back to old formula if no content_units defined
    if not gold.get("content_units"):
        completeness = 0.6 * node_recall + 0.3 * edge_recall + 0.1 * attr_recall
        s_sem = completeness
    else:
        s_sem = 0.4 * node_recall + 0.6 * s_sem_cu
        completeness = s_sem

    # Edge precision/recall/F1 (for detailed reporting)
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
        "s_sem": s_sem,
        "completeness": completeness,
        # New CU-based scores
        "s_sem_nodes": node_recall,
        "s_sem_cu": s_sem_cu,
        "matched_cu_count": cu_results["matched_cu_count"],
        "total_cu_count": cu_results["total_cu_count"],
        "matched_cus": cu_results["matched_cus"],
        "missed_cus": cu_results["missed_cus"],
        "cu_details": cu_results["cu_details"],
        # Legacy scores (backward compat)
        "node_completeness": node_recall,
        "edge_completeness": edge_recall,
        "attr_completeness": attr_recall,
        "gold_node_count": len(gn),
        "pred_node_count": len(pn),
        "matched_node_count": len(matched_nodes),
        "matched_nodes": list(matched_nodes),
        "missed_nodes": list(gn - pn),
        "gold_edge_count": len(ge),
        "matched_edge_count": len(matched_edges),
        "edges": {"precision": prec_e, "recall": rec_e, "f1": f1_e, "tp": tp_e, "fp": fp_e, "fn": fn_e},
        "attributes": {"precision": prec_a, "recall": rec_a, "f1": f1_a, "tp": tp_a, "fp": fp_a, "fn": fn_a},
        "extracted_graph": pred,
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

    NEW formula: S_sem = 0.4 * node_recall + 0.6 * CU_recall
    This weights clinical content units (12 relational CUs) more heavily than
    simple entity mention, making S_sem more sensitive to relational/semantic
    deficits characteristic of svPPA.

    Returns:
        s_sem: float in [0, 1], semantic adequacy
        details: dict with completeness, CU scores, edges, attributes metrics
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


def get_patient_profile(
    transcript: dict[str, Any],
    gold_graph_path: Path | str | None = None,
    s_ph: float | None = None,
    layer2_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Generate a human-readable patient profile showing semantic assessment details.
    
    This enables per-patient interpretability as described in the paper.
    
    Args:
        transcript: Transcript dict with text and words
        gold_graph_path: Path to gold scene graph JSON
        s_ph: Optional phonological adequacy score (from phoneme_adequacy.py)
        layer2_scores: Optional dict of Layer 2 construct scores
    
    Returns:
        dict with:
        - entities_mentioned: list of gold entities found in transcript
        - entities_missed: list of gold entities not found
        - content_units_covered: list of CU IDs matched
        - content_units_missed: list of CU IDs not matched
        - cu_details: per-CU breakdown (id, kind, gold description, matched)
        - s_sem: semantic adequacy score
        - s_sem_nodes: entity recall component
        - s_sem_cu: content unit coverage component
        - s_ph: phonological adequacy (if provided)
        - delta: S_ph - S_sem (if s_ph provided)
        - layer2: Layer 2 construct scores (if provided)
        - interpretation: human-readable summary
    """
    s_sem, details = compute_s_sem(transcript, gold_graph_path)
    
    profile = {
        "entities_mentioned": details.get("matched_nodes", []),
        "entities_missed": details.get("missed_nodes", []),
        "entity_recall": details.get("node_completeness", 0.0),
        "content_units_covered": details.get("matched_cus", []),
        "content_units_missed": details.get("missed_cus", []),
        "cu_details": details.get("cu_details", []),
        "cu_coverage": details.get("s_sem_cu", 0.0),
        "s_sem": s_sem,
        "s_sem_nodes": details.get("s_sem_nodes", 0.0),
        "s_sem_cu": details.get("s_sem_cu", 0.0),
    }
    
    if s_ph is not None:
        profile["s_ph"] = s_ph
        import math
        if not math.isnan(s_sem) and not math.isnan(s_ph):
            profile["delta"] = s_ph - s_sem
        else:
            profile["delta"] = float("nan")
    
    if layer2_scores:
        profile["layer2"] = layer2_scores
    
    # Generate interpretation
    interp_lines = []
    
    n_entities = len(profile["entities_mentioned"])
    n_missed = len(profile["entities_missed"])
    interp_lines.append(f"Entities: {n_entities} mentioned, {n_missed} missed")
    
    n_cu = len(profile["content_units_covered"])
    n_cu_missed = len(profile["content_units_missed"])
    total_cu = n_cu + n_cu_missed
    interp_lines.append(f"Content Units: {n_cu}/{total_cu} covered ({100*n_cu/total_cu:.0f}%)" if total_cu > 0 else "Content Units: N/A")
    
    interp_lines.append(f"S_sem = {s_sem:.3f}" if not (isinstance(s_sem, float) and s_sem != s_sem) else "S_sem = N/A")
    
    if s_ph is not None:
        delta = profile.get("delta", float("nan"))
        interp_lines.append(f"S_ph = {s_ph:.3f}")
        if not (isinstance(delta, float) and delta != delta):
            if delta > 0.1:
                interp_lines.append(f"Δ = {delta:.3f} (phonology > semantics, svPPA pattern)")
            elif delta < -0.1:
                interp_lines.append(f"Δ = {delta:.3f} (semantics > phonology, nfvPPA pattern)")
            else:
                interp_lines.append(f"Δ = {delta:.3f} (no clear dissociation)")
    
    if layer2_scores:
        high_impairment = [k for k, v in layer2_scores.items() if v > 0.6]
        if high_impairment:
            interp_lines.append(f"High impairment constructs: {', '.join(high_impairment)}")
    
    profile["interpretation"] = "\n".join(interp_lines)
    
    return profile
