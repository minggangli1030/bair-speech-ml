#!/bin/bash
# Run the full pipeline using Manual Diarization data, then analysis and classifiers.
# Outputs are written with _manual suffix so we can compare with the original (before) results.
#
# Requirements:
#   - HF_TOKEN for pyannote diarization (same as run_full_pipeline.sh)
#   - Optional: CODE_DIR for SylWav phoneme decoding
#
# Usage:
#   export HF_TOKEN="your_token"
#   ./scripts/run_manual_diarization_pipeline.sh

set -e
cd "$(dirname "$0")/.."

MANUAL_DIR="$(pwd)/data/Manual Diarization"
MANIFEST_MANUAL="data/manifest_manual_diarization.csv"
PIPELINE_OUT_MANUAL="data/manual_diarization_pipeline_outputs"

if [ ! -f "$MANIFEST_MANUAL" ]; then
  echo "Building manifest from Manual Diarization folder..."
  python3 scripts/build_manual_diarization_manifest.py
fi

echo "=== Step 1: Diarization + ASR + SylWav pipeline (Manual Diarization data) ==="
echo "  Data dir: $MANUAL_DIR"
echo "  Output:   $PIPELINE_OUT_MANUAL"
echo ""

if [ -z "$HF_TOKEN" ]; then
  echo "Warning: HF_TOKEN not set. Skipping pipeline step (diarization)."
  echo "  Set HF_TOKEN and re-run to process audio, or ensure $PIPELINE_OUT_MANUAL already has *_feature_extraction.txt files."
else
  python3 -m src.cli ppa-pipeline \
    --manifest "$MANIFEST_MANUAL" \
    --data-dir "$MANUAL_DIR" \
    --output-dir "$PIPELINE_OUT_MANUAL" \
    --phoneme-code-dir "${CODE_DIR:-}" \
    --no-skip-existing
fi

# If manual pipeline output dir is empty, copy from original pipeline outputs (same stems) so analysis can run
ORIG_PIPELINE="$(pwd)/data/ppa_pipeline_outputs"
if [ ! -d "$PIPELINE_OUT_MANUAL" ] || [ -z "$(ls -A "$PIPELINE_OUT_MANUAL"/*_feature_extraction.txt 2>/dev/null)" ]; then
  if [ -d "$ORIG_PIPELINE" ]; then
    echo ""
    echo "Copying pipeline outputs from $ORIG_PIPELINE for stems in manual manifest..."
    mkdir -p "$PIPELINE_OUT_MANUAL"
    python3 -c "
import pandas as pd
from pathlib import Path
manifest = pd.read_csv(\"$MANIFEST_MANUAL\")
orig = Path(\"$ORIG_PIPELINE\")
dest = Path(\"$PIPELINE_OUT_MANUAL\")
for _, row in manifest.iterrows():
    stem = Path(row['audio_path']).stem
    for ext in ['_feature_extraction.txt', '.rttm']:
        src = orig / (stem + ext)
        if src.exists():
            (dest / (stem + ext)).write_bytes(src.read_bytes())
"
    echo "Done. Re-run with HF_TOKEN to regenerate from Manual Diarization audio."
  fi
fi

echo ""
echo "=== Step 2: Population-normed Layer 2 (manual data) ==="
python3 scripts/fix_layer2_population_norm.py \
  --manifest "$MANIFEST_MANUAL" \
  --pipeline-dir "$PIPELINE_OUT_MANUAL" \
  --output outputs/layer2_fixed_manual.csv

echo ""
echo "=== Step 3: Full analysis (S_ph, S_sem, delta) — manual data ==="
python3 scripts/run_full_analysis.py \
  --manifest "$MANIFEST_MANUAL" \
  --data-dir "$MANUAL_DIR" \
  --pipeline-output-dir "$PIPELINE_OUT_MANUAL" \
  --output outputs/ppa_dissociation_manual.csv \
  --stats-output outputs/ppa_stats_manual.json

echo ""
echo "=== Step 4: Merge fixed Layer 2 → ppa_dissociation_fixed_l2_manual.csv ==="
python3 -c "
import pandas as pd
df = pd.read_csv('outputs/ppa_dissociation_manual.csv')
l2 = pd.read_csv('outputs/layer2_fixed_manual.csv')
l2_cols = ['subject_id','session'] + [c for c in l2.columns if c.startswith('l2_') and c != 'l2_sem_impairment']
if 's_sem_layer2_adequacy_fixed' in l2.columns:
    l2_cols.append('s_sem_layer2_adequacy_fixed')
df = df.merge(l2[l2_cols], on=['subject_id','session'], how='left')
alpha = 0.5
if 's_sem_layer2_adequacy_fixed' in df.columns:
    df['s_sem_layer2_adequacy'] = df['s_sem_layer2_adequacy_fixed'].fillna(df.get('s_sem_layer2_adequacy'))
df['s_sem'] = alpha * df['s_sem_graph'] + (1 - alpha) * df['s_sem_layer2_adequacy']
df['delta'] = df['s_ph'] - df['s_sem']
df.to_csv('outputs/ppa_dissociation_fixed_l2_manual.csv', index=False)
print('Saved outputs/ppa_dissociation_fixed_l2_manual.csv')
"

echo ""
echo "=== Step 5: Control vs svPPA classifier (manual data) ==="
python3 scripts/train_dissociation_classifier.py \
  --features outputs/ppa_dissociation_fixed_l2_manual.csv \
  --task control_vs_svppa \
  --output outputs/classifier_control_vs_svppa_manual.json

echo ""
echo "=== Step 6: svPPA vs (nfv+lv) real data (manual) ==="
python3 scripts/train_sv_vs_nflf_real.py \
  --features outputs/ppa_dissociation_fixed_l2_manual.csv \
  --layer2-fixed outputs/layer2_fixed_manual.csv \
  --output outputs/sv_vs_nflf_real_results_manual.json

echo ""
echo "Done. Run scripts/compare_manual_diarization_results.py for before vs after comparison."
