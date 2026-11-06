"""Offline scene-graph demonstration using fictional descriptions only."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.graph_semantic import compute_s_sem

def main():
    descriptions = {
        "brief": "A man is here.",
        "detailed": "The man is reading a book on the blanket. The boy is flying a kite. The dog is chasing the girl near the lake.",
    }
    for name, text in descriptions.items():
        score, details = compute_s_sem({"text": text})
        print(json.dumps({"example": name, "semantic_score": round(score, 4),
                          "matched_content_units": details.get("matched_cu_count"),
                          "matched_nodes": sorted(details.get("matched_nodes", []))}))

if __name__ == "__main__":
    main()
