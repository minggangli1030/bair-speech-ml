# Experiment and tooling index

| Purpose | Entry point |
| --- | --- |
| Offline synthetic graph example | `demo_semantic_scoring.py` |
| Audio-to-inference outputs | `run_full_pipeline.sh`; CLI `python -m src.cli ppa-pipeline` |
| Original paper analysis | `run_paper_pipeline.sh` |
| Feature/scoring pass | `run_full_analysis.py` |
| Population normalization | `fix_layer2_population_norm.py` |
| Binary dissociation comparison | `train_dissociation_classifier.py` |
| PPA-only binary comparison | `train_sv_vs_nflf_real.py` |
| Multi-seed / balanced / multiclass studies | `run_multi_seed_*.py`, `train_*multiclass*.py`, `train_sv_vs_nflf_balanced.py` |
| Ablations | `run_ablation_study.py`, `run_ablation_alpha_sweep.py` |
| Error inspection | `run_failure_case_analysis.py` |
| Manual diarization comparison | `build_manual_diarization_manifest.py`, `compare_manual_diarization_results.py`, `run_manual_diarization_pipeline.sh` |
| Figures and tables | `generate_*.py`, `plot_longitudinal_svppa_vs_nflv.py`, `dataset_duration_table.py` |

Run commands from the repository root. Most analysis scripts require cohort-specific generated data; use `--help` to inspect their arguments. Related experiment scripts are retained rather than merged because they differ in cohort, balancing, targets, or comparison design. See [evaluation boundaries](../docs/EVALUATION.md) before interpreting outputs.
