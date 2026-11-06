# Reproduction and data contracts

## Offline contract checks

Install `requirements-core.txt`, run `python scripts/demo_semantic_scoring.py`, then `python -m pytest -q tests`. These paths use fictional text and temporary fixtures. They neither load recorded speech nor download models.

## Audio pipeline

For full experiments, install `requirements.txt` and provision Whisper, spaCy resources, a pyannote-authorized environment, and the optional external phoneme decoder as required by the selected path. Use your own authorized manifest and inputs. Never put tokens in command history or source; prefer environment configuration.

```bash
python -m src.cli ppa-pipeline   --manifest data/manifest_ppa_full.csv   --data-dir data/audio   --output-dir data/ppa_pipeline_outputs
```

The generic manifest loader requires `subject_id` and `audio_path`; labels and task fields are optional there. The full paper analysis additionally expects its original cohort/session metadata and feature outputs. Check each script's CLI and joins rather than assuming every experiment accepts the same schema.

`HF_TOKEN` provides model access to pyannote; `CODE_DIR` locates separately supplied phoneme-decoder code. A missing decoder can trigger a stub path, so inspect provenance and missing features before interpreting scores.

## Historical paper workflow

`bash scripts/run_paper_pipeline.sh` applies population normalization, extracts scores, merges fixed Layer 2 features, and runs the two original classification tasks. It requires the original local manifest and inference outputs. It is not the offline demonstration and was not rerun during portfolio preparation. The aggregate numbers in the README are historical results.

## Data boundary

The author confirmed that the existing research data are cleared for the repository to remain public. For source-focused browsing, recordings, subject-level manifests, caches, and generated outputs are removed from the current Git index while retained locally. No history rewrite or data deletion is performed. `data/graph/picnic_gold_star.json` is a reference scene specification and remains tracked.
