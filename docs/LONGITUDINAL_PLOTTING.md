# Longitudinal comparison: svPPA vs lv+nfv

Single script and figure for comparing longitudinal change between svPPA and lv+nfv (nfvPPA + lvPPA).

## Data and script

- **Data:** `outputs/ppa_dissociation_full.csv` (must have `subtype`, `subject_id`, `session`, `s_ph`, `s_sem`).
- **Script:** `scripts/plot_longitudinal_svppa_vs_nflv.py`

The script keeps only subjects with **2+ visits**, computes **years from first visit** per subject, and aggregates at 3 time bins (0, 1, 2 years). Y-axis is **% change from initial visit** (per-subject baseline). Mean trend is plotted for each group (no shaded band).

## Run

```bash
python scripts/plot_longitudinal_svppa_vs_nflv.py
```

- **Output:** `outputs/figures/longitudinal_svppa_vs_nflv.png`
- **Options:** `--data`, `--output`, `--format` (png/pdf/svg)

## Figure

Three panels: **S_ph**, **S_sem**, **Total score**. Each shows mean % change over time for **svPPA** (blue) and **lv+nfv** (orange), so you can compare underlying patterns between groups.
