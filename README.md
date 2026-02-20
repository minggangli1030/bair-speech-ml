# svPPA Speech Analysis Pipeline

End-to-end inference and baseline training for **semantic variant Primary Progressive Aphasia (svPPA)** speech analysis.

## Overview

- **Input**: Per-subject audio (wav/mp3) of elicited speech tasks, optional metadata/labels.
- **Output**:
  1. **Layer 1**: 20 proxy features (speech rate, MLU, lexical/morphosyntactic, etc.)
  2. **Layer 2**: 10 specialized constructs (reduced speech rate, agrammatism, anomia, etc.)
  3. **Baseline**: Logistic regression (svPPA vs control) with explainability (coefficients, per-subject top features).

## Setup

```bash
cd "/Users/minggangli/Workspace/Speech/Semantic Evaluation"
python -m venv .venv
source .venv/bin/activate   # or: .venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Optional: `sentence-transformers` for semantic distance (Layer 1 feature 20). If not installed, that feature is stubbed (NaN).

## Data Format

**Manifest CSV** (`data/manifest.csv`):

| subject_id | audio_path        | label | task  |
|------------|-------------------|-------|-------|
| sub01      | sub01_task1.wav   | 1     | picture |
| sub01      | sub01_task2.wav   | 1     | story  |
| sub02      | sub02_task1.wav   | 0     | picture |

- `subject_id`, `audio_path` required.
- `label`: optional; 0 = control, 1 = svPPA (or impaired). Used for training.
- Multiple rows per subject: features are aggregated (mean) to subject-level.

## Usage

Run from the **project root** (or set `PYTHONPATH=.`).

### 1. Preprocess (ASR + phonetic cache)

```bash
python -m src.cli preprocess \
  --manifest data/manifest.csv \
  --cache-dir data/cache \
  --asr whisper \
  --whisper-model base
```

- Skips files that already have cached transcript/phonetic (use `--no-skip-existing` to rerun).
- Phonetic encoder: stub by default; use `--phonetic-encoder mylab.phonetic:LabEncoder` when available.

### 2. Extract features (Layer 1 + Layer 2)

```bash
python -m src.cli extract \
  --manifest data/manifest.csv \
  --cache-dir data/cache \
  --output outputs/features.csv
```

- Reads cached transcripts/phonetics, computes 20 Layer 1 and 10 Layer 2 features per subject.
- Writes `outputs/features.csv` with columns: `subject_id`, `label`, plus all feature names.

### 3. Train baseline (logistic regression)

**Option A – no split (all data for training + CV):**
```bash
python -m src.cli train \
  --features outputs/features.csv \
  --model-dir outputs/models \
  --target label \
  --report outputs/train_report.json
```

**Option B – stratified train/val/test (recommended when you have limited data):**
```bash
python -m src.cli train \
  --features outputs/features.csv \
  --model-dir outputs/models \
  --target label \
  --split 0.6,0.2,0.2 \
  --report outputs/train_report.json
```
- Fit is on the **train** portion only; **val** and **test** are held out.
- Saves `outputs/models/splits/train.csv`, `val.csv`, `test.csv` with subject-level rows so the split is reproducible.
- Report includes train/val/test accuracy, F1, ROC-AUC, and confusion matrices.

- Uses Layer 2 features by default; add `--layer1` to also use Layer 1. Saves model, scaler, and feature schema under `outputs/models/`.

### 4. Predict (per-subject probabilities + explainability)

```bash
python -m src.cli predict \
  --features outputs/features.csv \
  --model-dir outputs/models \
  --output outputs/predictions.csv \
  --top-k 5
```

- Outputs `subject_id`, `prob_svPPA`, `predicted`, and top contributing features per subject.

## Project Structure

```
src/
  config.py         # Paths and model settings (dataclasses)
  io/
    dataset.py      # Load manifest, iterate subjects, load audio
    cache.py       # Disk cache for transcripts/phonetics
  models/
    asr.py         # ASR interface + Whisper backend (pluggable)
    phonetic.py    # Phonetic encoder interface + stub
  features/
    layer1.py      # 20 Layer 1 proxy features
    layer2.py      # 10 Layer 2 constructs from Layer 1
    utils.py       # spaCy POS, dependency, tokenization helpers
  train/
    baseline.py   # Logistic regression train/eval/predict + reporting
  cli.py          # Entrypoints: preprocess, extract, train, predict
scripts/          # Optional helper scripts
requirements.txt
```

## Layer 1 (20) and Layer 2 (10)

- **Layer 1**: WPM, articulation rate, pauses, MLU, clause count, TTR, pronoun ratio, function word omission, dependency completeness, word-finding pause, filled pause frequency, semantic distance to expected entities, etc.
- **Layer 2**: Reduced speech rate, reduced length/complexity, agrammatism, paragrammatism, anomia, empty speech, semantic paraphasias, phonemic paraphasias (placeholder), self-correction, overall communication impairment. Each is a deterministic score in [0,1] from standardized Layer 1 z-scores (configurable weights).

## Caching and logging

- Transcripts and phonetic outputs are stored under `cache_dir/transcripts/` and `cache_dir/phonetics/` keyed by subject and audio path. Re-running preprocess skips existing files unless `--no-skip-existing` is set.
- Logging goes to stderr; level INFO by default.

## TODOs (marked in code)

- **Phonetic encoder**: Integrate lab-provided model; stub used when not available.
- **Semantic distance**: Requires `sentence-transformers` and optional task-level expected entities; otherwise NaN.
- **Morphological error rate**: Heuristic baseline; TODO for richer rule set.

## Minimal test (feature functions)

**Requires spaCy.** From project root:

```bash
pip install spacy
python -m spacy download en_core_web_sm
python scripts/test_features.py
```

If spaCy or `en_core_web_sm` is missing, the script prints the install command and exits. The test runs unit-test-style checks on Layer 1/Layer 2 with a small synthetic transcript (no audio).

## Synthetic data (no audio)

To try the full train/predict pipeline without real audio or transcripts:

```bash
python scripts/generate_synthetic_features.py   # creates outputs/features.csv with 24 synthetic subjects
python -m src.cli train --features outputs/features.csv --model-dir outputs/models --split 0.6,0.2,0.2
python -m src.cli predict --features outputs/features.csv --model-dir outputs/models --output outputs/predictions.csv
```
