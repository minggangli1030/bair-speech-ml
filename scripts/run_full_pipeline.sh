#!/bin/bash
# Run SylWav + diarization on all audio in data/ and then full dissociation analysis.
#
# Requirements:
#   - HF_TOKEN: Hugging Face token for pyannote diarization
#     Get one at: https://huggingface.co/settings/tokens
#     Accept pyannote model terms: https://huggingface.co/pyannote/speaker-diarization-3.1
#
# Usage:
#   export HF_TOKEN="your_token"
#   ./scripts/run_full_pipeline.sh
#
# Or: HF_TOKEN="your_token" ./scripts/run_full_pipeline.sh

set -e
cd "$(dirname "$0")/.."

if [ -z "$HF_TOKEN" ]; then
  echo "Error: HF_TOKEN must be set for pyannote diarization."
  echo "  export HF_TOKEN=\"your_huggingface_token\""
  echo "  Get a token at: https://huggingface.co/settings/tokens"
  exit 1
fi

DATA_DIR="$(pwd)/data"
OUTPUT_DIR="$DATA_DIR/ppa_pipeline_outputs"
CODE_DIR="${CODE_DIR:-/data/sarikapasumarthy/code}"

echo "=== Step 1: Diarization + ASR + SylWav phoneme pipeline ==="
echo "  Data dir: $DATA_DIR"
echo "  Output:   $OUTPUT_DIR"
echo "  SylWav:   $CODE_DIR"
echo "  Files:    254 audio files"
echo ""

python -m src.cli ppa-pipeline \
  --manifest data/manifest_ppa_full.csv \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --phoneme-code-dir "$CODE_DIR" \
  --no-skip-existing

echo ""
echo "=== Step 2: Full dissociation analysis ==="
python scripts/run_full_analysis.py \
  --manifest data/manifest_ppa_full.csv \
  --data-dir "$DATA_DIR" \
  --pipeline-output-dir "$OUTPUT_DIR" \
  --output outputs/ppa_dissociation_all.csv \
  --stats-output outputs/ppa_stats_all.json

echo ""
echo "=== Done. Results in outputs/ppa_dissociation_all.csv and outputs/ppa_stats_all.json ==="
