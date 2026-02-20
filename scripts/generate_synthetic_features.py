"""Generate a synthetic features CSV for testing train/val/test split when no real audio is available."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features.layer1 import LAYER1_NAMES
from src.features.layer2 import LAYER2_NAMES

def main() -> None:
    out = Path("outputs/features.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    n_subjects = 24  # 12 per class for stratification
    rows = []
    for i in range(n_subjects):
        label = i % 2  # alternate 0, 1
        # Slight mean shift by label so the classifier has signal
        shift = 0.2 if label == 1 else -0.2
        l1 = {name: float(rng.normal(shift, 0.5)) for name in LAYER1_NAMES}
        # Clamp ratios to [0,1] where needed
        for k in ["content_word_density", "pronoun_usage_ratio", "type_token_ratio", "abandoned_utterance_rate",
                  "function_word_omission_rate", "morphological_error_rate", "dependency_parse_completeness",
                  "filled_pause_frequency"]:
            if k in l1:
                l1[k] = np.clip(l1[k], 0, 1)
        l1["wpm"] = max(50, l1["wpm"] + 100)
        l1["articulation_rate_phones_per_sec"] = max(2, l1["articulation_rate_phones_per_sec"] + 4)
        l1["utterance_count"] = max(1, l1["utterance_count"] + 5)
        l1["mlu"] = max(1, l1["mlu"] + 6)
        from src.features.layer2 import compute_layer2_features
        _, l2 = compute_layer2_features(l1)
        row = {"subject_id": f"syn_{i:02d}", "label": label}
        row.update(l1)
        row.update(l2)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"Wrote {out} with {len(df)} subjects (labels: {df['label'].value_counts().to_dict()})")

if __name__ == "__main__":
    main()
