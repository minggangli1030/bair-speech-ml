#!/usr/bin/env python3
"""
P1a: Validate S_sem scoring on manual transcripts.

Takes 5 sample transcripts and shows:
1. The extracted Content Units (entities/relations)
2. What matched against the gold graph
3. The S_sem score

This validates that the scoring isn't broken.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.graph_semantic import (
    compute_s_sem,
    _heuristic_extract,
    ENTITY_CANON,
)


# 5 sample transcripts for validation (representing different quality levels)
SAMPLE_TRANSCRIPTS = [
    {
        "id": "perfect_description",
        "description": "Complete description hitting all major CUs",
        "text": "In this picnic scene, we see a family having a picnic. There's a mother in a blue dress pouring water from a pitcher, a father in a white shirt leaning against a tree reading a newspaper, and a little boy playing with a dog. The boy is reaching for a frisbee. There's a picnic blanket with a basket, sandwiches, and cups on it. In the background, there's a lake with sailboats and some trees. A butterfly is flying around and there are birds in the sky. The family has a car parked nearby.",
    },
    {
        "id": "moderate_description",
        "description": "Decent description with some gaps",
        "text": "A family is having a picnic outside. The woman is pouring something and the man is reading. There's a kid with a dog. They have food on a blanket. You can see water in the background.",
    },
    {
        "id": "sparse_description",
        "description": "Very sparse - semantic impairment pattern",
        "text": "There's a... thing happening. People outside. Doing the... you know. The place. With the stuff.",
    },
    {
        "id": "phonological_errors_good_semantics",
        "description": "Phonological errors but good content (svPPA anti-pattern)",
        "text": "The muvver is powring water from a pitser. The faver is reeding the noospaper under a twee. The boy is plaving with the dok and frisbi. They have a pisnic blasket with food and sanwiches.",
    },
    {
        "id": "circumlocution_pattern",
        "description": "svPPA pattern: circumlocution, word-finding difficulty",
        "text": "So there's a... the lady... she's doing the... with the container... putting the wet stuff. And the man is... what do you call it... looking at the paper thing. The small one, the child... running around with the... four legs... animal.",
    },
]


def main():
    print("=" * 70)
    print("P1a: S_sem VALIDATION ON MANUAL TRANSCRIPTS")
    print("=" * 70)
    
    gold_graph_path = Path("data/graph/picnic_gold_star.json")
    if not gold_graph_path.exists():
        print(f"ERROR: Gold graph not found at {gold_graph_path}")
        return
    
    # Load gold graph for reference
    with open(gold_graph_path) as f:
        gold = json.load(f)
    
    print(f"\nGold graph: {len(gold.get('nodes', []))} nodes, {len(gold.get('edges', []))} edges")
    print(f"Expected entities: {[n['label'] for n in gold.get('nodes', [])][:10]}...")
    print()
    
    all_results = []
    
    for sample in SAMPLE_TRANSCRIPTS:
        print("-" * 70)
        print(f"SAMPLE: {sample['id']}")
        print(f"Description: {sample['description']}")
        print(f"\nText: \"{sample['text'][:150]}{'...' if len(sample['text']) > 150 else ''}\"")
        
        # Create transcript dict
        transcript = {
            "text": sample["text"],
            "segments": [{"text": sample["text"], "speaker": "Participant"}],
        }
        
        # Compute S_sem
        s_sem, details = compute_s_sem(transcript, gold_graph_path)
        
        # Show extracted entities
        extracted = details.get("extracted_graph", {})
        pred_nodes = [n["label"] for n in extracted.get("nodes", [])]
        
        print(f"\nExtracted entities ({len(pred_nodes)}): {pred_nodes[:15]}{'...' if len(pred_nodes) > 15 else ''}")
        
        # Show matched entities
        matched = details.get("matched_nodes", [])
        print(f"Matched to gold ({len(matched)}): {matched[:15]}{'...' if len(matched) > 15 else ''}")
        
        # Show score breakdown
        node_score = details.get("node_completeness", 0)
        edge_score = details.get("edge_completeness", 0)
        
        print(f"\nScoring:")
        print(f"  Node completeness: {node_score:.3f} ({details.get('matched_node_count', 0)}/{details.get('gold_node_count', 0)})")
        print(f"  Edge completeness: {edge_score:.3f} ({details.get('matched_edge_count', 0)}/{details.get('gold_edge_count', 0)})")
        print(f"  S_sem = {s_sem:.3f}")
        
        # Expected vs actual
        expected_pattern = {
            "perfect_description": "HIGH (0.5+)",
            "moderate_description": "MEDIUM (0.2-0.5)",
            "sparse_description": "VERY LOW (<0.1)",
            "phonological_errors_good_semantics": "HIGH (0.5+) - content is good despite phonetic errors",
            "circumlocution_pattern": "LOW (0.1-0.3) - circumlocution indicates semantic difficulty",
        }
        print(f"  Expected: {expected_pattern.get(sample['id'], 'N/A')}")
        
        all_results.append({
            "id": sample["id"],
            "s_sem": s_sem,
            "extracted_count": len(pred_nodes),
            "matched_count": len(matched),
        })
        print()
    
    # Summary
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"{'Sample':<35} {'S_sem':>8} {'Extracted':>10} {'Matched':>10}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['id']:<35} {r['s_sem']:>8.3f} {r['extracted_count']:>10} {r['matched_count']:>10}")
    
    # Sanity checks
    print("\nSANITY CHECKS:")
    
    # Check 1: perfect_description should have highest score
    scores = {r["id"]: r["s_sem"] for r in all_results}
    if scores["perfect_description"] >= max(scores.values()) - 0.01:
        print("✓ 'perfect_description' has highest or near-highest score")
    else:
        print("✗ WARNING: 'perfect_description' doesn't have highest score!")
    
    # Check 2: sparse_description should have lowest
    if scores["sparse_description"] <= min(scores.values()) + 0.01:
        print("✓ 'sparse_description' has lowest or near-lowest score")
    else:
        print("✗ WARNING: 'sparse_description' doesn't have lowest score!")
    
    # Check 3: phonological errors shouldn't hurt semantic score much
    if scores["phonological_errors_good_semantics"] >= 0.2:
        print("✓ Phonological errors don't overly penalize semantic content")
    else:
        print("✗ WARNING: Phonological errors may be hurting semantic scoring")
    
    # Check 4: circumlocution pattern should be detectable
    if scores["circumlocution_pattern"] < scores["moderate_description"]:
        print("✓ Circumlocution pattern scores lower than moderate description")
    else:
        print("? Circumlocution pattern may not be fully captured")
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
