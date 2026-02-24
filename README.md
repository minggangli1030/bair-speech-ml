# Automated PPA Classification from Speech: Phonetic, Semantic, and Clinical-Layer Features

Single pipeline and scripts for the **publishable story**: (1) **control vs svPPA** and (2) **svPPA vs non-semantic PPA (nfv+lv)** using automated S_ph, S_sem (spatiosemantic scene graph + Layer 2), and population-normed clinical Layer 2 features.

---

## Main results

| Task | Best model | Accuracy | AUC | p (real data) |
|------|------------|----------|-----|----------------|
| **Control vs svPPA** | S_ph + S_sem (+ Δ) | **~85%** | **~0.92** | &lt; 0.001 |
| **svPPA vs (nfv+lv)** | Full clinical RF (S_ph, S_sem, scene graph, Layer 2, acoustic) | **~82%** | **~0.72** | **&lt; 0.00001** |

- **S_sem** = composite of **spatiosemantic scene graph** (picnic content) and **Layer 2 semantic adequacy** (anomia, empty speech, semantic paraphasias).
- **S_ph** = phonological adequacy from SylWav vs G2P.
- **Layer 2** = population-normed (not per-sample z-scored); includes agrammatism, reduced speech rate, anomia, etc.

---

## Reproducibility: one run order

**Prerequisites:** Pipeline outputs already generated (diarization + ASR + SylWav) in `data/ppa_pipeline_outputs/`, and manifest `data/manifest_ppa_full.csv`.

```bash
# From repo root
cd /path/to/semantic-evaluation

# 1. Fix Layer 2 (population normalization)
python scripts/fix_layer2_population_norm.py \
  --manifest data/manifest_ppa_full.csv \
  --pipeline-dir data/ppa_pipeline_outputs \
  --output outputs/layer2_fixed.csv

# 2. Build feature table (S_ph, S_sem, delta, etc.) and merge fixed Layer 2
python scripts/run_full_analysis.py \
  --manifest data/manifest_ppa_full.csv \
  --pipeline-output-dir data/ppa_pipeline_outputs \
  --output outputs/ppa_dissociation_all.csv \
  --stats-output outputs/ppa_stats.json

# Merge fixed Layer 2 into features (composite S_sem and delta)
python -c "
import pandas as pd
from pathlib import Path
df = pd.read_csv('outputs/ppa_dissociation_all.csv')
l2 = pd.read_csv('outputs/layer2_fixed.csv')
l2_cols = ['subject_id','session'] + [c for c in l2.columns if c.startswith('l2_') and c != 'l2_sem_impairment']
if 's_sem_layer2_adequacy_fixed' in l2.columns:
    l2_cols.append('s_sem_layer2_adequacy_fixed')
df = df.merge(l2[l2_cols], on=['subject_id','session'], how='left')
alpha = 0.5
if 's_sem_layer2_adequacy_fixed' in df.columns:
    df['s_sem_layer2_adequacy'] = df['s_sem_layer2_adequacy_fixed'].fillna(df.get('s_sem_layer2_adequacy'))
df['s_sem'] = alpha * df['s_sem_graph'] + (1-alpha) * df['s_sem_layer2_adequacy']
df['delta'] = df['s_ph'] - df['s_sem']
df.to_csv('outputs/ppa_dissociation_fixed_l2.csv', index=False)
print('Saved outputs/ppa_dissociation_fixed_l2.csv')
"

# 3. Control vs svPPA
python scripts/train_dissociation_classifier.py \
  --features outputs/ppa_dissociation_fixed_l2.csv \
  --task control_vs_svppa \
  --output outputs/classifier_control_vs_svppa.json

# 4. svPPA vs (nfv+lv) on real data only
python scripts/train_sv_vs_nflf_real.py \
  --features outputs/ppa_dissociation_fixed_l2.csv \
  --layer2-fixed outputs/layer2_fixed.csv \
  --output outputs/sv_vs_nflf_real_results.json

# 5. (Optional) LLM baseline — requires OPENAI_API_KEY
# python scripts/run_llm_baseline.py --task binary --output outputs/llm_baseline_predictions.csv
```

Or run the master script (does steps 1–4; assumes pipeline outputs and manifest exist):

```bash
bash scripts/run_paper_pipeline.sh
```

---

## Project layout (condensed)

```
semantic-evaluation/
├── README.md                          # This file
├── requirements.txt
├── data/
│   ├── manifest_ppa_full.csv          # Sample manifest (subject_id, session, subtype, audio_path)
│   ├── graph/picnic_gold_star.json   # Gold scene graph for S_sem
│   └── ppa_pipeline_outputs/         # Pipeline outputs (*_feature_extraction.txt)
├── scripts/
│   ├── run_paper_pipeline.sh          # One script: Layer 2 fix → features → both classifiers
│   ├── fix_layer2_population_norm.py  # Population-normed Layer 2 (required)
│   ├── run_full_analysis.py           # S_ph, S_sem (scene graph + Layer 2), delta
│   ├── train_dissociation_classifier.py  # Control vs svPPA (and optional sv vs nflv with sim)
│   ├── train_sv_vs_nflf_real.py      # svPPA vs nfv+lv, real data, full clinical + scene graph
│   ├── run_llm_baseline.py            # LLM baseline (Phan et al. style)
│   ├── run_ablation_alpha_sweep.py   # Ablation: alpha sweep (scene graph vs Layer 2 weight)
│   └── run_ablation_study.py          # Ablation: model variants (S_sem only, +S_ph, +delta, L2, etc.)
├── src/
│   ├── features/
│   │   ├── graph_semantic.py          # Spatiosemantic scene graph → S_sem_graph
│   │   ├── phoneme_adequacy.py        # S_ph from SylWav vs G2P
│   │   ├── layer1.py / layer2.py     # Layer 1/2 constructs
│   │   └── dissociation.py            # Δ and stats
│   ├── pipeline/runner.py             # Diarization + ASR + phoneme pipeline
│   └── ...
└── outputs/
    ├── layer2_fixed.csv               # Population-normed Layer 2
    ├── population_l1_stats.json       # For reproducibility
    ├── ppa_dissociation_all.csv       # From run_full_analysis
    ├── ppa_dissociation_fixed_l2.csv # Main feature table (S_ph, S_sem, delta, L2, …)
    ├── classifier_control_vs_svppa.json
    └── sv_vs_nflf_real_results.json   # sv vs nflv results (p, accuracy, AUC)
```

**Optional (not part of the main story):** `scripts/generate_simulated_svppa.py`, `scripts/train_multiclass_classifier.py`, `scripts/run_full_pipeline.sh` (full diarization + SylWav pipeline). **Ablations:** `scripts/run_ablation_alpha_sweep.py` (alpha sweep for composite S_sem), `scripts/run_ablation_study.py` (model-variant ablations).

---

## Feature definitions

- **S_ph:** Phonological adequacy (1 − phoneme error rate), SylWav vs G2P.
- **S_sem_graph:** Spatiosemantic scene graph completeness vs gold picnic graph.
- **S_sem:** Composite = α × S_sem_graph + (1−α) × Layer2_semantic_adequacy; Layer2 adequacy = 1 − mean(anomia, empty_speech, semantic_paraphasias).
- **Δ:** S_ph − S_sem.
- **Layer 2 (fixed):** Population-normed constructs (agrammatism, reduced_speech_rate, anomia, empty_speech, semantic_paraphasias, etc.).

---

## References

- Phan et al. (2024). Proc. Interspeech 2024. [LLM for content assessment](https://www.isca-archive.org/interspeech_2024/phan24_interspeech.html).
