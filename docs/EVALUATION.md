# Evaluation boundaries

## Historical outcomes

The previous README and local aggregate result files report 85.1% accuracy / 0.932 ROC-AUC on control-versus-svPPA and 82.1% / 0.734 for svPPA versus the combined non-semantic variants. The project also contains ten-seed reports, ablations, and balanced/multiclass variants. These are different experimental summaries, not interchangeable estimates of one deployed system.

## Issues identified in source review

- `scripts/train_dissociation_classifier.py`, `scripts/train_sv_vs_nflf_real.py`, and the generic `train_baseline` path fit a scaler before constructing CV predictions. Generic feature construction also imputes before CV. Held-out rows therefore influence preprocessing.
- Population normalization must be audited for train-only fitting within each evaluation split.
- Repeated recordings/sessions require an explicit subject-disjoint evaluation contract. Row-stratified CV alone does not establish that contract.
- Cached artifact keys do not identify model/version/input contents; stale-cache reuse is a reproducibility risk.
- Small-cohort selection, class imbalance, ASR/diarization errors, and approximate phoneme mapping can affect results.

The documentation cleanup does not re-run patient analyses or silently replace scientific outputs. A follow-up evaluation should use train-fitted transformations inside a scikit-learn pipeline, subject-grouped splits where needed, predefined endpoints, and uncertainty across independently defined runs. Current metrics should not be presented as validated clinical performance.

## What the portfolio checks establish

Synthetic tests cover cache separation/round trips, manifest validation, graph evidence, and missing-reference handling. The offline example runs without audio or external models. These are software contract checks; they do not validate diagnosis, generalization, or the historical classification figures.
