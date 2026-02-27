# sv vs nflv: With vs Without Synthetic Augmentation

## Setup
- **Real data:** 42 svPPA, 187 nfv+lv (229 PPA samples with s_ph, s_sem, delta).
- **Simulated svPPA:** 187 samples from `generate_simulated_svppa.py` (degraded semantic channel only; S_ph from original nfv/lv, S_sem recomputed after content→generic + fillers).

## Comparison (dissociation classifier, LR, 5-fold CV)

| Setting | Model | N (eval) | Acc | F1 | AUC |
|--------|--------|----------|-----|-----|-----|
| **No augmentation** | S_ph_S_sem_Delta | 229 real | 0.520 | 0.276 | 0.493 |
| **No augmentation** | Dissociation_Plus | 229 real | **0.598** | **0.313** | **0.620** |
| **With augmentation** (train on real+sim, test on **real only**) | S_ph_S_sem_Delta | 229 real | 0.555 | 0.239 | 0.452 |
| **With augmentation** (train on real+sim, test on **real only**) | Dissociation_Plus | 229 real | 0.550 | 0.248 | 0.497 |

## Finding
**Augmentation did not improve real-data performance.** When the model is trained on real + simulated svPPA and evaluated **only on real** samples, accuracy and AUC are similar or slightly worse than training on real data only. So the **paper’s main sv vs nflv result should remain real-only** (no simulated augmentation). The synthetic setup is a controlled semantic-degradation experiment; it does not yield a better real-world classifier in this setup.

## How to reproduce
```bash
# 1. Generate simulated svPPA
python scripts/generate_simulated_svppa.py --features outputs/ppa_dissociation_fixed_l2.csv --output outputs/simulated_svppa.csv

# 2. No augmentation
python scripts/train_dissociation_classifier.py --task sv_vs_nflf --features outputs/ppa_dissociation_fixed_l2.csv --output outputs/classifier_sv_vs_nflf_no_aug.json

# 3. With augmentation (includes real-only evaluation at end of output)
python scripts/train_dissociation_classifier.py --task sv_vs_nflf --features outputs/ppa_dissociation_fixed_l2.csv --simulated-csv outputs/simulated_svppa.csv --output outputs/classifier_sv_vs_nflf_with_aug.json
```

## For the paper
- **LLM baseline:** Report as weakest baseline (e.g. ~37% AUC sv vs nflv vs your pipeline ~73% real-only); no phonology grounding.
- **Synthetic augmentation:** Report that you compared with vs without; augmentation did **not** improve real-only performance. Frame the synthesis as *linguistically grounded semantic degradation* (controlled); main result stays **real data only**. If you still want to mention it, position it as a negative result / ablation rather than as a data-augmentation contribution that improves the main metric.
