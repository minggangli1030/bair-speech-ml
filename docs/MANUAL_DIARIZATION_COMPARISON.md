# Manual Diarization Pipeline & Before/After Comparison

This document describes how to run the full pipeline on the **Manual Diarization** data and compare results with the original pipeline.

## Overview

- **Manual Diarization** folder: `data/Manual Diarization/` — contains manually diarized (or curated) audio in subfolders:
  - `picnics-controls-enhanced copy/`
  - `picnics-svppa-enhanced copy/`
  - `picnics-lvppa-enhanced copy/`
  - `picnics-nfvppa-enhanced copy/`

- The pipeline is run on this data; outputs are written with a `_manual` suffix so they can be compared with the original (before) results.

## Quick Start

1. **Build the manifest** (if not already done):
   ```bash
   python3 scripts/build_manual_diarization_manifest.py
   ```
   Creates `data/manifest_manual_diarization.csv` (254 rows).

2. **Run the full pipeline** (diarization + analysis + classifiers):
   ```bash
   export HF_TOKEN="your_huggingface_token"   # required for pyannote diarization
   ./scripts/run_manual_diarization_pipeline.sh
   ```
   If `HF_TOKEN` is not set, the script will copy existing pipeline outputs from `data/ppa_pipeline_outputs` for the same file stems so that analysis and classifier steps still run (same features, manual manifest only).

3. **Compare before vs after**:
   ```bash
   python3 scripts/compare_manual_diarization_results.py
   ```

## Outputs (after run)

| Output | Description |
|--------|-------------|
| `data/manifest_manual_diarization.csv` | Manifest for Manual Diarization (254 files) |
| `data/manual_diarization_pipeline_outputs/` | `*_feature_extraction.txt` and `.rttm` per file |
| `outputs/layer2_fixed_manual.csv` | Population-normed Layer 2 for manual data |
| `outputs/ppa_dissociation_manual.csv` | S_ph, S_sem, Δ per sample |
| `outputs/ppa_stats_manual.json` | Summary stats |
| `outputs/ppa_dissociation_fixed_l2_manual.csv` | Merged L2 + composite S_sem |
| `outputs/classifier_control_vs_svppa_manual.json` | Control vs svPPA classifier results |
| `outputs/sv_vs_nflf_real_results_manual.json` | svPPA vs (nfv+lv) results |

## Comparison summary (typical run)

The comparison script reports:

1. **Control vs svPPA**  
   - Metrics: Accuracy, F1, ROC-AUC for the main dissociation model (S_ph + S_sem or S_ph + S_sem + Δ).  
   - Before = original pipeline; After = pipeline on Manual Diarization data (or same pipeline outputs with manual manifest).

2. **svPPA vs (nfv+lv)**  
   - Best model (by p-value): accuracy, F1, ROC-AUC, Mann–Whitney p-value, effect size.

3. **Sample counts**  
   - Number of samples and how many have SylWav/transcript in each run.

For a **true** before/after comparison using re-diarized and re-ASR’d manual audio, set `HF_TOKEN` and run `run_manual_diarization_pipeline.sh` so that Step 1 runs on the Manual Diarization audio; then run the comparison again.
