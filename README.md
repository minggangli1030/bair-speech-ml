# Phonetic-Semantic Dissociation Index for PPA Classification

End-to-end pipeline for **Primary Progressive Aphasia (PPA)** speech analysis using the phonetic-semantic dissociation index (Δ = S_ph - S_sem).

## Key Results

| Model | Accuracy | F1 | AUC |
|-------|----------|-----|-----|
| S_sem only | 73.1% | 0.775 | 0.810 |
| S_ph + S_sem | 83.6% | 0.864 | 0.919 |
| **S_ph + S_sem + Δ** | **85.1%** | **0.875** | **0.921** |

## Overview

- **S_sem (Semantic Adequacy)**: Content Unit coverage against gold scene graph (Picnic task)
- **S_ph (Phonological Adequacy)**: Phoneme accuracy from SylWav vs G2P-converted transcripts
- **Δ (Dissociation Index)**: S_ph - S_sem, capturing the phonology-semantics dissociation

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Data

**Manifest CSV** (`data/manifest_ppa_full.csv`):
- 254 samples: 25 controls, 42 svPPA, 109 lvPPA, 78 nfvPPA
- Columns: `subject_id`, `session`, `audio_path`, `label`, `subtype`, `task`

## Usage

### 1. Run Full Pipeline (Diarization + ASR + Phoneme)

Requires `HF_TOKEN` for pyannote diarization:

```bash
export HF_TOKEN="your_huggingface_token"
export CODE_DIR=/path/to/code  # Optional: for SylWav phoneme decoding

python -m src.cli ppa-pipeline \
  --manifest data/manifest_ppa_full.csv \
  --data-dir /path/to/PPA/enhanced \
  --output-dir data/ppa_pipeline_outputs
```

### 2. Run Dissociation Analysis

```bash
python scripts/run_full_analysis.py \
  --manifest data/manifest_ppa_full.csv \
  --data-dir /path/to/PPA/enhanced \
  --pipeline-output-dir data/ppa_pipeline_outputs \
  --output outputs/ppa_dissociation.csv \
  --stats-output outputs/ppa_stats.json
```

### 3. Train Classifier

```bash
# Binary: control vs svPPA
python scripts/train_dissociation_classifier.py \
  --features outputs/ppa_dissociation_with_controls.csv \
  --output outputs/classifier_results.json

# Multi-class: all subtypes
python scripts/train_multiclass_classifier.py \
  --features outputs/ppa_dissociation_all.csv
```

### 4. Validate S_sem Scoring

```bash
python scripts/validate_s_sem.py
```

## Project Structure

```
semantic-evaluation/
├── src/
│   ├── cli.py                    # CLI entry points
│   ├── analyze.py                # Dissociation analysis orchestration
│   ├── config.py                 # Configuration
│   ├── features/
│   │   ├── graph_semantic.py     # S_sem: scene graph scoring
│   │   ├── phoneme_adequacy.py   # S_ph: phonological adequacy
│   │   ├── dissociation.py       # Δ computation + statistics
│   │   ├── layer1.py             # 20 Layer 1 proxy features
│   │   └── layer2.py             # 10 Layer 2 constructs
│   ├── pipeline/
│   │   └── runner.py             # Diarization + ASR + phoneme pipeline
│   ├── io/                       # Data loading and caching
│   ├── models/                   # ASR and phonetic interfaces
│   └── train/                    # Baseline classifier
├── scripts/
│   ├── run_full_analysis.py      # Main analysis script
│   ├── train_dissociation_classifier.py
│   ├── train_multiclass_classifier.py
│   ├── run_ablation_study.py
│   └── validate_s_sem.py
├── data/
│   ├── manifest_ppa_full.csv     # Full PPA dataset manifest
│   ├── graph/picnic_gold_star.json  # Gold scene graph
│   └── ppa_pipeline_outputs/     # Pipeline outputs (ASR + phonemes)
├── outputs/
│   ├── ppa_dissociation_all.csv  # S_sem for all 254 samples
│   ├── ppa_dissociation_with_controls.csv  # Full features (67 samples)
│   └── classifier_results_full.json
├── requirements.txt
└── README.md
```

## Feature Definitions

### Dissociation Features
- **S_sem**: Weighted completeness (60% nodes, 30% edges, 10% attributes) vs gold graph
- **S_ph**: 1 - Phoneme Error Rate (Levenshtein distance between SylWav and G2P)
- **Δ**: S_ph - S_sem (positive = phonology > semantics)

### Layer 1 (20 features)
WPM, articulation rate, pause metrics, MLU, TTR, pronoun ratio, function word omission, etc.

### Layer 2 (10 constructs)
Reduced speech rate, agrammatism, paragrammatism, anomia, empty speech, semantic/phonemic paraphasias, etc.

## Statistical Validation

Effect sizes (Cliff's δ) for control vs svPPA:
- S_sem: δ = 0.65 (large)
- S_ph: δ = 0.79 (large)
- 100% of svPPA samples show Δ > 0 (mean = 0.209)
