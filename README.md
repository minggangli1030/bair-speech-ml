# IPSS-PPA: Interpretable Phonological–Semantic Scoring for Automated Primary Progressive Aphasia Subtyping

Code repository for the Interspeech 2026 submission: **IPSS-PPA** — an interpretable, fully automated pipeline for PPA subtype classification from WAB Picnic picture description speech.

**Paper:** "Interpretable Phonological–Semantic Scoring for Automated Primary Progressive Aphasia Subtyping"

---

## Dataset

| Group | Recordings | Label |
|-------|-----------|-------|
| Healthy controls | 25 | 0 |
| svPPA | 42 | 1 |
| nfvPPA | 78 | 2 |
| lvPPA | 109 | 3 |
| **Total** | **254** | |

## Key Results

| Task | Best Model | Accuracy | AUC | Effect Size |
|------|------------|----------|-----|-------------|
| **Control vs svPPA** (n=67) | S_ph + S_sem | 85.1% | 0.932 | δ = 0.79 (large) |
| **svPPA vs nfv+lv** (n=229) | Full + L2 + Graph (RF) | 82.1% | 0.734 | δ = 0.47 (medium) |

**Core innovation:** Task-agnostic acoustic and linguistic features are mapped to ten clinician-aligned constructs (anomia, agrammatism, reduced speech rate, etc.), then integrated into phonological and semantic adequacy scores. The pipeline produces clinician-readable evidence chains suitable for clinical screening.

---

## Result Files for Paper

**Prerequisites:** Pipeline outputs already generated (diarization + ASR + SylWav) in `data/ppa_pipeline_outputs/`, and manifest `data/manifest_ppa_full.csv`.

| File | Contents | Use in Paper |
|------|----------|--------------|
| `outputs/classifier_results.json` | Control vs svPPA classifier metrics | Table 3: Binary classification |
| `outputs/sv_vs_nflf_real_results.json` | svPPA vs (nfv+lv) classifier metrics | Table 4: Subtype discrimination |
| `outputs/ppa_stats.json` | Statistical comparisons (Mann-Whitney, Cliff's δ) | Effect sizes, p-values |
| `outputs/population_l1_stats.json` | Population normalization parameters | Methodology (Layer 2) |
| `outputs/cu_combined_heatmap_paper.png` | Entity/CU mention rates by group | Figure: Semantic content coverage |

---

## Pipeline Overview

```
Audio → Diarization → ASR + SylWav Phonemes
                          ↓
              ┌──────────────────────┐
              │  S_ph (phonological) │ ← SylWav vs G2P edit distance
              │  S_sem (semantic)    │ ← Scene graph + Layer 2 constructs
              │  Δ = S_ph − S_sem    │ ← Dissociation index
              └──────────────────────┘
                          ↓
              Classification (LR / RF)
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/semantic-evaluation.git
cd semantic-evaluation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Reproducing Results

**Prerequisites:** Pipeline outputs (diarization + ASR + SylWav) in `data/ppa_pipeline_outputs/` and manifest `data/manifest_ppa_full.csv`.

```bash
# 1. Fix Layer 2 with population normalization
python scripts/fix_layer2_population_norm.py

# 2. Run full analysis (S_ph, S_sem, Δ)
python scripts/run_full_analysis.py

# 3. Train classifiers
python scripts/train_dissociation_classifier.py --task control_vs_svppa
python scripts/train_sv_vs_nflf_real.py

# 4. Failure case analysis (Section 6 of paper)
python scripts/run_failure_case_analysis.py
```

Or run the master script:
```bash
bash scripts/run_paper_pipeline.sh
```

---

## Project Structure

```
semantic-evaluation/
├── README.md
├── requirements.txt
├── docs/
│   ├── METHODOLOGY.md           # Pipeline methodology
│   ├── METHODOLOGY_DETAILED.md  # Detailed procedural specification
│   └── FAILURE_CASE_ANALYSIS.md # Section 6: Correct/misclassified examples
├── data/
│   └── graph/picnic_gold_star.json  # Gold scene graph (12 content units)
├── scripts/
│   ├── run_paper_pipeline.sh        # Master script
│   ├── fix_layer2_population_norm.py
│   ├── run_full_analysis.py
│   ├── train_dissociation_classifier.py
│   ├── train_sv_vs_nflf_real.py
│   ├── run_ablation_*.py            # Ablation studies
│   └── run_failure_case_analysis.py # Section 6 failure case analysis
├── src/features/
│   ├── graph_semantic.py        # S_sem (scene graph + CU scoring)
│   ├── phoneme_adequacy.py      # S_ph (phoneme error rate)
│   ├── layer1.py / layer2.py    # Clinical constructs
│   └── dissociation.py          # Δ computation, statistics
└── outputs/
    ├── classifier_results.json       # Control vs svPPA
    ├── sv_vs_nflf_real_results.json  # svPPA vs nfv+lv
    ├── ppa_stats.json                # Statistical comparisons
    └── population_l1_stats.json      # L1 population stats
```

---

## Feature Definitions

| Feature | Formula | Description |
|---------|---------|-------------|
| **S_ph** | 1 − PER | Phonological adequacy (SylWav vs G2P) |
| **S_sem** | 0.5 × S_graph + 0.5 × S_L2 | Composite semantic adequacy |
| **S_sem_graph** | 0.4 × node_recall + 0.6 × CU_recall | Scene graph completeness (12 CUs) |
| **S_sem_L2** | 1 − mean(all 10 L2 constructs) | Communication adequacy from Layer 2 |
| **Δ** | S_ph − S_sem | Dissociation index |

**Layer 2 constructs:** reduced_speech_rate, reduced_length_complexity, agrammatism, paragrammatism, anomia, empty_speech, semantic_paraphasias, phonemic_paraphasias_neologisms, self_correction, overall_communication_impairment

---

## Statistical Methods

- **Mann-Whitney U test:** Non-parametric group comparison (scipy.stats.mannwhitneyu, two-sided)
- **Cliff's delta:** Effect size (|δ| < 0.147 negligible, 0.147–0.33 small, 0.33–0.474 medium, ≥0.474 large)
- **Cross-validation:** Stratified 5-fold, random_state=42
- **Classifiers:** Logistic Regression (balanced), Random Forest (n=100, max_depth=6)

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{pasumarthy2026issppa,
  title={Interpretable Phonological–Semantic Scoring for Automated Primary Progressive Aphasia Subtyping},
  author={Pasumarthy, Sarika and Li, Minggang and Lian, Jiachen and Guo, Chenxu and Yang, Emma and Kurteff, Lynn and Ezzes, Zoe and Keegan-Rodewald, Willa and Vonk, Jet M.J. and Ramkrishnan, Siddarth and Antonicelli, Giada and Miller, Zachary A. and Gorno-Tempini, Maria Luisa and Anumanchipalli, Gopala},
  booktitle={Proc. Interspeech 2026},
  year={2026}
}
```

---

## License

[To be determined]

---

## References

- Hickok, G., & Poeppel, D. (2007). The cortical organization of speech processing. Nature Reviews Neuroscience.
- Nicholas, L. E., & Brookshire, R. H. (1993). A system for quantifying the informativeness and efficiency of the connected speech of adults with aphasia. JSLHR.
