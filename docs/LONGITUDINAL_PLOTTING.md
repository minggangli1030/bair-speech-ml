# Longitudinal Plotting — Full Detail for Score Change Over Time

This document gives everything you need to plot longitudinal graphs (e.g. scores vs **years from first visit**) for repeated svPPA patients and to compare their scores over time.

---

## 1. Data source and generation

- **Plot-ready CSV:** `outputs/longitudinal_plot_data.csv`  
  One row per **(subject_id, session)** for every svPPA subject who has **more than one session** (repeated speakers).

- **How to generate it:**  
  Run the longitudinal analysis script (it writes the plot CSV automatically):

  ```bash
  python scripts/longitudinal_analysis.py
  ```

  Optional: use a different feature table and output path:

  ```bash
  python scripts/longitudinal_analysis.py \
    --features outputs/ppa_dissociation_fixed_l2.csv \
    --output-plot-csv outputs/longitudinal_plot_data.csv
  ```

  If there are **no** svPPA subjects with multiple sessions, the script exits cleanly and does **not** write `longitudinal_plot_data.csv` (section omitted).

---

## 2. Plot data columns (all necessary info for plotting)

| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | string | Participant ID (e.g. `sub-5468`). Use to group series and to filter one subject. |
| `session` | string | Session ID (e.g. `ses-20070313`). |
| `session_date` | string | Date of session in `YYYY-MM-DD` (parsed from `session`). |
| `years_from_first_visit` | float | **X-axis for plots.** Time in years from that subject’s first visit. First visit = 0.0. |
| `visit_number` | int | Visit index for this subject (1, 2, 3, …). |
| `n_visits` | int | Total number of visits for this subject (same for all rows of that subject). |
| `time_span_years` | float | Total time span from first to last visit in years (same for all rows of that subject). |
| `transcript_len` | float | Transcript length (e.g. word count) for that visit. |
| `s_ph` | float | Phonological adequacy [0, 1]. Higher = better. |
| `s_sem` | float | Semantic adequacy [0, 1]. Higher = better. |
| `s_sem_graph` | float | Graph-based semantic score [0, 1]. |
| `s_sem_layer2_adequacy` | float | Layer‑2 adequacy component [0, 1]. |
| `delta` | float | Dissociation index: **S_ph − S_sem**. Can be negative. |
| `phoneme_error_rate` | float | Phoneme error rate [0, 1]. Higher = more errors (1 − S_ph). |
| `silence_ratio` | float | Proportion of silence [0, 1]. |

**For “years from first visit”:**

- Session IDs are `ses-YYYYMMDD`. The script parses the date and, per subject, sets the first visit’s date to 0 and computes `years_from_first_visit = (current_session_date - first_session_date) / 365` (using the same day-count logic as in the script).
- So **X-axis** = `years_from_first_visit`; **Y-axis** = any of the score columns above (or multiple series on one plot with a legend).

---

## 3. Subject and visit summary (who has longitudinal data)

From the current dataset, these svPPA subjects have repeated sessions and appear in the plot CSV:

| Subject ID | n_visits | time_span_years | Sessions (example) |
|------------|----------|-----------------|---------------------|
| sub-5468   | 3        | ~2.25           | ses-20070313, ses-20080403, ses-20090403 |
| sub-10880  | 2        | ~2.99           | ses-20101102, ses-20131028 |
| sub-13138  | 3        | ~3.23           | ses-20120110, ses-20140304, ses-20150401 |
| sub-13919  | 2        | ~1.21           | ses-20120314, ses-20130528 |
| sub-15607  | 2        | ~2.13           | ses-20140225, ses-20160412 |
| sub-17020  | 3        | ~5.04           | ses-20140415, ses-20160926, ses-20190430 |
| sub-21609  | 2        | ~2.09           | ses-20160712, ses-20180813 |
| sub-24565  | 2        | ~1.03           | ses-20170823, ses-20180807 |
| sub-25434  | 2        | ~1.97           | ses-20171207, ses-20181126 |
| sub-26076  | 2        | ~1.05           | ses-20171011, ses-20181031 |

(Exact `time_span_years` and session lists are in `longitudinal_plot_data.csv`.)

---

## 4. How to plot (this repo’s script)

**Prerequisites:** `pip install matplotlib pandas`

**Plot all longitudinal subjects** (one figure per subject, saved under `outputs/longitudinal_plots/`):

```bash
python scripts/plot_longitudinal.py
```

**Plot one subject** (e.g. sub-5468):

```bash
python scripts/plot_longitudinal.py --subject sub-5468
```

**Custom paths and format:**

```bash
python scripts/plot_longitudinal.py \
  --data outputs/longitudinal_plot_data.csv \
  --subject sub-5468 \
  --output-dir outputs/longitudinal_plots \
  --format png
```

The script produces:

- **Panel 1:** Adequacy scores over **years from first visit**: S_ph, S_sem, Δ.
- **Panel 2:** Phoneme error rate and silence ratio over time; transcript length on a secondary Y-axis (if present).

So you get **all necessary info** (time axis, subject ID, visit count, time span, and every score column) from the CSV and the script uses them to build the graphs.

---

## 5. Plotting in your own tool (e.g. Excel, R, Python)

- **X-axis:** `years_from_first_visit`
- **Y-axis (one or more series):** any of `s_ph`, `s_sem`, `delta`, `phoneme_error_rate`, `silence_ratio`, `transcript_len`, etc.
- **Group by:** `subject_id` — one line per subject (or one plot per subject).
- **Caption / title:** Use `subject_id`, `n_visits`, and `time_span_years` (e.g. “sub-5468, 3 visits, 2.33 years”).

Example (pseudo):

- Filter to one `subject_id`.
- Sort by `years_from_first_visit`.
- Plot `years_from_first_visit` vs `s_ph`, `s_sem`, `phoneme_error_rate`, etc., with markers at each visit.

---

## 6. Interpretation (short)

- **S_ph, S_sem:** Lower over time can indicate worsening phonological/semantic production.
- **phoneme_error_rate:** Higher over time = more phoneme-level errors.
- **delta = S_ph − S_sem:** Negative and decreasing can reflect increasing dissociation (phonology relatively more preserved than semantics), as often seen in svPPA.

Overall pattern can be summarized as progressive change in speech production (e.g. “progressive deterioration with phoneme and semantic scores declining over time”), consistent with the longitudinal design.

---

## 7. Files reference

| File | Purpose |
|------|--------|
| `outputs/longitudinal_plot_data.csv` | Plot-ready table: years from first visit + all scores per visit. |
| `outputs/longitudinal_results.json` | Per-subject summary stats (first/last, change, slope). |
| `outputs/longitudinal_results.md` | Short text report. |
| `scripts/longitudinal_analysis.py` | Builds plot CSV and results; omit cleanly if no longitudinal data. |
| `scripts/plot_longitudinal.py` | Plots score change over years from first visit (one figure per subject). |
