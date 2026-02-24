#!/bin/bash
# Single pipeline for the paper: Layer 2 fix → features → control vs svPPA → sv vs nflv (real).
# Run from repo root. Requires: data/manifest_ppa_full.csv, data/ppa_pipeline_outputs/*.txt
set -e
cd "$(dirname "$0")/.."

echo "=== 1. Population-normed Layer 2 ==="
python scripts/fix_layer2_population_norm.py \
  --manifest data/manifest_ppa_full.csv \
  --pipeline-dir data/ppa_pipeline_outputs \
  --output outputs/layer2_fixed.csv

echo ""
echo "=== 2. Full analysis (S_ph, S_sem, delta) ==="
python scripts/run_full_analysis.py \
  --manifest data/manifest_ppa_full.csv \
  --pipeline-output-dir data/ppa_pipeline_outputs \
  --output outputs/ppa_dissociation_all.csv \
  --stats-output outputs/ppa_stats.json

echo ""
echo "=== 3. Merge fixed Layer 2 → ppa_dissociation_fixed_l2.csv ==="
python -c "
import pandas as pd
df = pd.read_csv('outputs/ppa_dissociation_all.csv')
l2 = pd.read_csv('outputs/layer2_fixed.csv')
l2_cols = ['subject_id','session'] + [c for c in l2.columns if c.startswith('l2_') and c != 'l2_sem_impairment']
if 's_sem_layer2_adequacy_fixed' in l2.columns:
    l2_cols.append('s_sem_layer2_adequacy_fixed')
df = df.merge(l2[l2_cols], on=['subject_id','session'], how='left')
alpha = 0.5
if 's_sem_layer2_adequacy_fixed' in df.columns:
    df['s_sem_layer2_adequacy'] = df['s_sem_layer2_adequacy_fixed'].fillna(df.get('s_sem_layer2_adequacy'))
df['s_sem'] = alpha * df['s_sem_graph'] + (1 - alpha) * df['s_sem_layer2_adequacy']
df['delta'] = df['s_ph'] - df['s_sem']
df.to_csv('outputs/ppa_dissociation_fixed_l2.csv', index=False)
print('Saved outputs/ppa_dissociation_fixed_l2.csv')
"

echo ""
echo "=== 4. Control vs svPPA ==="
python scripts/train_dissociation_classifier.py \
  --features outputs/ppa_dissociation_fixed_l2.csv \
  --task control_vs_svppa \
  --output outputs/classifier_control_vs_svppa.json

echo ""
echo "=== 5. svPPA vs (nfv+lv) real data ==="
python scripts/train_sv_vs_nflf_real.py \
  --features outputs/ppa_dissociation_fixed_l2.csv \
  --layer2-fixed outputs/layer2_fixed.csv \
  --output outputs/sv_vs_nflf_real_results.json

echo ""
echo "Done. Optional: set OPENAI_API_KEY and run scripts/run_llm_baseline.py --task binary"
