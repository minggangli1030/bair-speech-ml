#!/bin/bash
# Full pipeline with fixed Layer 2: merge, binary classifier, multiclass classifier.
# Run from project root.
set -e
cd "$(dirname "$0")/.."

ALPHA=${ALPHA:-0.5}
FEATURES="outputs/ppa_dissociation_all.csv"
L2_FIXED="outputs/layer2_fixed.csv"
OUT_FEATURES="outputs/ppa_dissociation_fixed_l2.csv"

echo "=== Step 0: Merge fixed Layer 2 into dissociation features (alpha=$ALPHA) ==="
python3 - <<PY
import pandas as pd
import numpy as np
from pathlib import Path

alpha = $ALPHA
df = pd.read_csv("$FEATURES")
l2 = pd.read_csv("$L2_FIXED")
df = df.merge(l2[["subject_id", "session", "s_sem_layer2_adequacy_fixed"]], on=["subject_id", "session"], how="left")
df["s_sem_layer2_adequacy"] = df["s_sem_layer2_adequacy_fixed"].fillna(df["s_sem_layer2_adequacy"])
df["s_sem"] = alpha * df["s_sem_graph"] + (1 - alpha) * df["s_sem_layer2_adequacy"]
df["delta"] = df["s_ph"] - df["s_sem"]
df.to_csv("$OUT_FEATURES", index=False)
print(f"Merged, recomputed s_sem (alpha={alpha}) and delta. Saved to $OUT_FEATURES")
PY

echo ""
echo "=== Step 1: Binary classifier (control vs svPPA) ==="
python scripts/train_dissociation_classifier.py \
  --features "$OUT_FEATURES" \
  --output outputs/classifier_results_fixed_l2.json

echo ""
echo "=== Step 2: Multiclass classifier ==="
python scripts/train_multiclass_classifier.py \
  --features "$OUT_FEATURES" \
  --output outputs/multiclass_results_fixed_l2.json

echo ""
echo "Done. Outputs: $OUT_FEATURES, classifier_results_fixed_l2.json, multiclass_results_fixed_l2.json"
