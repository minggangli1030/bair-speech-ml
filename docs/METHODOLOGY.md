# Pipeline Methodology (Detailed)

This document specifies the methodology for each component of the PPA classification pipeline.

---

## 1. Data and inputs

- **Manifest:** `data/manifest_ppa_full.csv` with columns `subject_id`, `session`, `subtype`, `label`, `audio_path`, (optional) `task`. Subtypes: `control`, `svppa`, `nfvppa`, `lvppa`. Labels: 0=control, 1=svPPA, 2=lvPPA, 3=nfvPPA.
- **Pipeline outputs:** For each row, the pipeline expects a file `data/ppa_pipeline_outputs/{stem}_feature_extraction.txt` where `stem = Path(audio_path).stem`. Each file is a text report containing:
  - **Segment table:** For each segment, `Timestamp`, `Speaker` (Participant vs Clinician), `ASR Transcription`, `Phonemic Encoding` (SylWav phoneme sequence). Only **Participant** segments are used for transcript and phonemes.
- **Gold scene graph:** `data/graph/picnic_gold_star.json` — reference scene graph for the picnic picture (nodes = entities, edges = relations, attributes). Used to compute spatiosemantic completeness.

---

## 2. Pipeline output parsing

- **Script / step:** Parsing is done inside `parse_pipeline_output()` in `src/features/phoneme_adequacy.py` and reused in `fix_layer2_population_norm.py`, `run_full_analysis.py`, `generate_simulated_svppa.py`.
- **Method:** Regex over the pipeline output file to extract per-segment: `Timestamp` (start–end), `Speaker`, `ASR Transcription`, `Phonemic Encoding`. Participant-only segments are concatenated to form:
  - `transcript`: full participant transcript (space-joined).
  - `phonemes`: space-joined SylWav phoneme sequence.
  - `segments`: list of `{start, end, speaker, transcript, phonemes}`.
- **Transcript dict for downstream:** From `segments`, a word-level structure is built: words from Participant segments, timestamps interpolated within segment duration; leading `"s "` stripped from segment transcript. Output: `{ "text": str, "words": [ { "word", "start", "end" } ] }`.

---

## 3. Layer 1 features (20 proxies)

- **Script:** `src/features/layer1.py`; called from `fix_layer2_population_norm.py` (and implicitly wherever Layer 2 is computed).
- **Input:** Transcript dict `{ text, words }` with word-level timestamps.
- **Method:**
  - **Utterances:** Words are split into utterances by pause threshold (default 0.5 s). Pause = gap between `words[i].end` and `words[i+1].start`.
  - **Layer 1 names (20):** `wpm`, `articulation_rate_phones_per_sec`, `mean_pause_duration`, `pause_to_speech_ratio`, `initial_response_latency`, `mlu`, `median_utterance_length`, `clause_count_per_utterance`, `abandoned_utterance_rate`, `utterance_count`, `open_class_closed_class_ratio`, `content_word_density`, `type_token_ratio`, `pronoun_usage_ratio`, `function_word_omission_rate`, `morphological_error_rate`, `dependency_parse_completeness`, `word_finding_pause_duration`, `filled_pause_frequency`, `semantic_distance_to_expected`.
  - **Constants:** `WORD_FINDING_PAUSE_THRESHOLD = 1.0` s; `MIN_PAUSE_GAP = 0.3` s; filled pauses: `{"um","uh","er","ah","eh","hm","hmm","uhm","umhm"}`.
  - **NLP:** Uses `get_nlp()` (spaCy or fallback), open/closed class ratio, content-word density, pronoun ratio, TTR, function-word omission heuristic, dependency completeness, morphological error heuristic, nouns/proper nouns, semantic distance (when expected entities provided; in this pipeline often disabled: `use_semantic_distance=False`).
- **Output:** One dict per sample with 20 float values (or NaN) keyed by `LAYER1_NAMES`.

---

## 4. Layer 2 (population-normed) — fix script

- **Script:** `scripts/fix_layer2_population_norm.py`.
- **Purpose:** Replace per-sample z-scoring of Layer 1 with **population-level** z-scoring so Layer 2 reflects between-subject variation.
- **Steps:**
  1. **Load pipeline outputs** for all manifest rows; build transcript dict; compute **Layer 1** for each sample (`compute_layer1_features(..., use_semantic_distance=False)`). Drop samples with no transcript.
  2. **Population stats:** Over the (n × 20) matrix of L1 features, compute `population_mean` and `population_std` per feature (column-wise `nanmean` / `nanstd`). Replace zero std with 1.0.
  3. **Layer 2 per sample:** For each sample, form the 20-dim L1 vector, z-score using **population** mean/std: `zvec = (vec - pop_mean) / pop_std`. Then for each of the 10 Layer 2 constructs, compute a **weighted average of z-scores** over the prescribed L1 indices (see `layer2.py` `DEFAULT_LAYER2_WEIGHTS`), then map to [0,1] with `(tanh(z)+1)/2`.
  4. **Semantic adequacy (fixed):** `s_sem_layer2_adequacy_fixed = 1 - mean(anomia, empty_speech, semantic_paraphasias)` (Layer 2 construct names); `l2_sem_impairment` = that mean.
- **Outputs:** `outputs/layer2_fixed.csv` (per subject/session: all `l2_*` columns, `s_sem_layer2_adequacy_fixed`, `l2_sem_impairment`); `outputs/population_l1_stats.json` (`feature_names`, `population_mean`, `population_std`, `n_samples`).

**Layer 2 construct definitions (from `layer2.py`):** 10 constructs, each = weighted sum of z-scored L1 features then sigmoid to [0,1]:
- `reduced_speech_rate` (wpm, articulation_rate, mean_pause_duration, pause_to_speech_ratio, initial_response_latency)
- `reduced_length_complexity` (mlu, median_utterance_length, clause_count_per_utterance, utterance_count)
- `agrammatism` (clause_count_per_utterance, function_word_omission_rate, morphological_error_rate, dependency_parse_completeness)
- `paragrammatism` (morphological_error_rate, dependency_parse_completeness, type_token_ratio)
- `anomia` (word_finding_pause_duration, filled_pause_frequency, mean_pause_duration, abandoned_utterance_rate)
- `empty_speech` (utterance_count, open_class_closed_class_ratio, pronoun_usage_ratio, type_token_ratio)
- `semantic_paraphasias` (semantic_distance_to_expected, type_token_ratio, utterance_count)
- `phonemic_paraphasias_neologisms` (morphological_error_rate, filled_pause_frequency)
- `self_correction` (filled_pause_frequency, abandoned_utterance_rate, initial_response_latency)
- `overall_communication_impairment` (wpm, mlu, utterance_count, open_class_closed_class_ratio, semantic_distance_to_expected)

---

## 5. Full analysis (S_ph, S_sem, Δ)

- **Script:** `scripts/run_full_analysis.py`.
- **Inputs:** Manifest, pipeline output directory, optional gold graph path. Optional `--s-sem-alpha` (default 0.4 in script; merge step in pipeline uses 0.5 — see below).
- **Per sample:**

  **5.1 Transcript and phonemes**  
  From pipeline file: `parse_pipeline_output` → `transcript`, `phonemes`, `segments`. If no pipeline output and not `--skip-asr`, Whisper ASR can be run (cached under `data/cache_ppa/transcripts/`).

  **5.2 S_ph (phonological adequacy)**  
  - **Source:** `src/features/phoneme_adequacy.py` `compute_s_ph(transcript, phonemes)`.
  - **Method:** Expected phonemes = G2P of transcript (word-by-word dictionary lookup + character fallback). Observed = SylWav phoneme string parsed to ARPAbet list; SIL removed for alignment. **Edit distance** = Levenshtein between expected and observed (no SIL). **PER = edit_distance / max(len(expected), len(observed))**.
  - **S_ph = 1 − PER**, clipped to [0, 1]. Also returned: `phoneme_error_rate` (= PER), `silence_ratio` = proportion of SIL in observed, `phonemes_per_word`.
  - S_ph is computed only when pipeline provides both transcript and non-placeholder phonemes.

  **5.3 S_sem_graph (spatiosemantic scene graph)**  
  - **Source:** `src/features/graph_semantic.py` `compute_s_sem(transcript_dict, gold_graph_path)`.
  - **Method:** Transcript → participant format (full text + spans with timestamps). Heuristic extraction:
    - **Entity extraction:** Canonical map (e.g. boy/girl/man/woman/dog/kite/blanket/ball/tree/grass/sky/lake/boat/house/family…) with variant matching.
    - **Spatial relation detection:** Patterns like "on", "in", "near", "next to", "beside", "behind", "by" within 5-word window of two entity mentions.
    - **Action relation detection:** Verb patterns (hold/holding/has, sit/sitting, read/reading, chase/chasing, fly/flying, run/running, attach/tied) with co-occurrence within 5-word window.
    - **Attribute detection:** State (flying) and activity (running, picnicking) patterns.
  - **Content Unit (CU) scoring:** The gold graph (`picnic_gold_star.json`) defines 12 clinical content units (CU_E01–CU_E09 = edges, CU_A01–CU_A03 = attributes) aligned with standard SLP CU scoring (Nicholas & Brookshire 1993). **s_sem_cu = matched_CUs / 12**.
  - **S_sem formula (clinically grounded):** `S_sem = 0.4 * node_recall + 0.6 * s_sem_cu`. This weights relational content (CUs) more heavily than simple entity mention, making S_sem more sensitive to semantic deficits characteristic of svPPA.
  - **Legacy scores (backward compat):** `node_completeness`, `edge_completeness`, `attr_completeness` are still returned.

  **5.4 Composite S_sem (in run_full_analysis)**  
  - **Formula:** `S_sem = alpha * s_sem_graph + (1 - alpha) * s_sem_layer2_adequacy`.
  - **In this script:** Layer 2 is computed **without** population normalization (internal z-score per sample) unless a separate merge is used. So the “full analysis” CSV may contain pre-fix Layer 2; the **paper pipeline** overwrites this in the merge step (see §6) with fixed Layer 2.

  **5.5 Dissociation index**  
  - **Δ = S_ph − S_sem** (only where both are non-NaN).

- **Outputs:** `outputs/ppa_dissociation_all.csv` (per subject/session: s_ph, s_sem, s_sem_graph, s_sem_layer2_adequacy, delta, phoneme_error_rate, silence_ratio, etc.); `outputs/ppa_stats.json` (e.g. pairwise S_sem comparisons, svPPA Δ stats). Statistics use **Mann-Whitney U** (two-sided) and **Cliff’s delta** (effect size: negligible &lt;0.147, small &lt;0.33, medium &lt;0.474, large ≥0.474).

---

## 6. Merge: fixed Layer 2 into feature table

- **Where:** In `scripts/run_paper_pipeline.sh` (inline Python) and equivalently in README.
- **Method:** Read `ppa_dissociation_all.csv` and `layer2_fixed.csv`. Merge on `(subject_id, session)` all columns from `layer2_fixed` that start with `l2_` (excluding `l2_sem_impairment` if desired) plus `s_sem_layer2_adequacy_fixed`. Set `s_sem_layer2_adequacy` from `s_sem_layer2_adequacy_fixed` where available. **Composite S_sem:** `s_sem = alpha * s_sem_graph + (1 - alpha) * s_sem_layer2_adequacy` with **alpha = 0.5**. **Delta:** `delta = s_ph - s_sem`.
- **Output:** `outputs/ppa_dissociation_fixed_l2.csv` (canonical feature table for classifiers).

---

## 7. Control vs svPPA classifier

- **Script:** `scripts/train_dissociation_classifier.py` with `--task control_vs_svppa`.
- **Data:** Rows with `label` in {0, 1} from the feature CSV (control=0, svPPA=1). Requires `s_sem` (and for models 2–4, `s_ph`, `delta`; optional extras and Layer 2).
- **Models (ablation):**  
  1. S_sem only.  
  2. S_ph + S_sem (samples with s_ph &gt; 0).  
  3. S_ph + S_sem + Δ.  
  4. Dissociation + phoneme_error_rate, silence_ratio (if present).  
  5. Layer 2 only (all `reduced_*` and named L2 columns).  
  6. Layer 2 + dissociation (S_ph, S_sem, Δ + L2).
- **Training:** **Logistic regression** (L2, C=1.0, `class_weight="balanced"`, max_iter=1000, solver="lbfgs", random_state=42). **Stratified 5-fold CV:** `StandardScaler` fit on train, transform train/test; median imputation for NaN per feature. Metrics: accuracy, precision, recall, F1, ROC-AUC; confusion matrix; coefficients and top 5 features.
- **Statistics:** Mann-Whitney U and Cliff’s delta for s_sem, s_ph, delta (control vs svPPA).
- **Output:** JSON with per-model metrics and summary table.

---

## 8. svPPA vs (nfv+lv) — dissociation classifier (optional simulated)

- **Script:** Same `train_dissociation_classifier.py` with `--task sv_vs_nflf`.
- **Data:** Rows with `subtype` in {svppa, nfvppa, lvppa}; binary label: svPPA=1, nfv+lv=0. Optionally merge **simulated svPPA** from `--simulated-csv`: rows appended with `binary_label=1`, `subtype=simulated_svppa`; must have `s_ph`, `s_sem`; other columns filled with NaN.
- **Models:** Same sequence as in §7 (S_sem only, S_ph+S_sem, +Δ, +extras, Layer2 only, Layer2+dissociation).
- **Real-only evaluation when simulated is used:** For models 3 and 4, an additional evaluation is run: **Stratified 5-fold CV over real samples only**; in each fold, **train** = real train indices + **all** simulated indices, **test** = real test indices. Metrics (Acc, F1, AUC) are reported **only on real test** samples. This answers “does augmentation improve real-world performance?”

---

## 9. svPPA vs (nfv+lv) — real-only, full clinical (main paper result)

- **Script:** `scripts/train_sv_vs_nflf_real.py`.
- **Data:** Feature table merged with `layer2_fixed` on `(subject_id, session)`. Rows restricted to `subtype` in {svppa, nfvppa, lvppa}; drop rows with missing s_sem or s_ph. **No simulated samples.**
- **Feature sets (examples):** base (s_ph, s_sem); +s_sem_graph; +delta; +s_sem_graph and delta; +acoustic (phoneme_error_rate, silence_ratio); +L2 semantic (l2_anomia, l2_empty_speech, l2_semantic_paraphasias); +L2 grammar (l2_agrammatism, l2_reduced_speech_rate); +all L2; Full_clinical (base+delta+acoustic+L2 semantic+L2 gram); Full_clinical_graph; Full_all_L2; Full_all_L2_graph. Only sets with ≥2 available columns are run.
- **Classifiers:** **Logistic regression** (class_weight="balanced", max_iter=2000) and **Random forest** (n_estimators=100, max_depth=6, class_weight="balanced", random_state=42).
- **Evaluation:** **Stratified 5-fold CV.** Median imputation, StandardScaler, then cross_val_predict for labels and predict_proba (class 1 = svPPA). Metrics: accuracy, F1, ROC-AUC. **Mann-Whitney U** on **predicted P(svPPA)** (real sv vs real nflv) with **Cliff’s delta** and effect-size label.
- **Univariate:** Mann-Whitney and Cliff’s delta for single features (e.g. s_sem, s_ph, delta, s_sem_graph, l2_anomia, l2_reduced_speech_rate) on real sv vs nflv.
- **Output:** `outputs/sv_vs_nflf_real_results.json` (per-model metrics, best p-value, best result).

---

## 10. Simulated svPPA (synthetic augmentation)

- **Script:** `scripts/generate_simulated_svppa.py`.
- **Purpose:** Generate synthetic “svPPA-like” samples by **degrading only the semantic channel**; S_ph is taken from the **original** nfv/lv sample (same subject/session).
- **Input:** Manifest (nfvPPA and lvPPA only), pipeline outputs, feature CSV (for S_ph and optional phoneme_error_rate, silence_ratio lookup), `population_l1_stats.json` (for Layer 2 with population norm), gold graph path, alpha (default 0.5), replace_ratio (default 0.7), seed.
- **Degradation (`src/features/simulate_svppa.py`):**  
  - **Content replacement:** Regex patterns replace specific words with generic terms (e.g. boy/girl/child→“person”, man/woman→“person”, dog→“thing”, kite/blanket/ball/…→“thing”) with probability **replace_ratio** per pattern.  
  - **Filled pauses:** After each token, insert one of `["um","uh","you know"]` with probability **insert_filler_ratio** (default in script 0.12).  
  - **Random seed:** `seed + hash(subject_id, session)` so reproducible per sample.
- **Recomputation:** From degraded text, build transcript dict; compute **s_sem_graph** (heuristic graph vs gold); compute **Layer 2** with **population** mean/std from `population_l1_stats.json`; **s_sem_layer2_adequacy** = 1 − mean(anomia, empty_speech, semantic_paraphasias); **s_sem = alpha * s_sem_graph + (1−alpha) * layer2_adequacy**; **S_ph** and (if present) phoneme_error_rate, silence_ratio **copied from original** nfv/lv row; **delta = S_ph − s_sem**.
- **Output:** `outputs/simulated_svppa.csv` (one row per processed nfv/lv sample: subject_id, session, subtype=simulated_svppa, label=1, s_ph, s_sem, s_sem_graph, delta, etc.). **Interpretation:** Simulated samples are **controlled semantic degradation**; S_ph is borrowed from real nfv/lv (preserved-phonology assumption for svPPA).

---

## 11. LLM baseline

- **Script:** `scripts/run_llm_baseline.py`.
- **Input:** Manifest, pipeline output dir (to get transcript per row via `get_transcript_from_pipeline`). Optional `--task binary` (svPPA vs nflv) or `--task multiclass` (control, svPPA, nfvPPA, lvPPA); `--model` (default gpt-4o-mini); `--output`; `--limit`.
- **Method:** For each manifest row, locate `{stem}_feature_extraction.txt`, parse pipeline output, extract participant transcript. Skip if transcript length &lt; 20. **API call:** OpenAI Chat Completions with system prompt (binary or multiclass), user message = “Transcript:\n\n{text[:4000]}”, max_tokens=20.
- **Parsing:** Response is normalized; first alphabetic token (or presence of keywords) is mapped:  
  - **Binary:** “svPPA”/“semantic” → svPPA; “nflv”/“non-semantic” → nflv; else unknown.  
  - **Multiclass:** “control” → control; “svPPA” → svPPA; “nfvPPA” → nfvPPA; “lvPPA” → lvPPA; else unknown.
- **Metrics:** Binary: accuracy and F1 on rows with valid true and pred binary labels. Multiclass: accuracy, macro-F1, classification report (precision/recall/F1 per class) on rows where both true subtype and prediction are in {control, svPPA, nfvPPA, lvPPA}.
- **Output:** CSV with subject_id, session, subtype, label, prediction, true_binary, pred_binary.

---

## 12. Ablation: alpha sweep (composite S_sem)

- **Script:** `scripts/run_ablation_alpha_sweep.py`.
- **Data:** Feature table; optionally merged with `layer2_fixed` so `s_sem_layer2_adequacy` is population-normed. Filter: control and svPPA only; require s_sem_graph, s_sem_layer2_adequacy, s_ph non-NaN.
- **Method:** For each **alpha** in a comma-separated list (default 0, 0.1, …, 1.0):  
  - **S_sem(α) = α * s_sem_graph + (1−α) * s_sem_layer2_adequacy**; **Δ(α) = s_ph − S_sem(α)**.  
  - **Mann-Whitney U** on Δ(α) (control vs svPPA); record p-value and Cliff’s delta.  
  - **Classifier:** Logistic regression on S_sem(α) only; 5-fold stratified CV; accuracy and F1.
- **Output:** JSON with per-alpha p-value, effect size, accuracy, F1; and which alpha (if any) achieves p &lt; 0.05.

---

## 13. Ablation: model variants

- **Script:** `scripts/run_ablation_study.py`.
- **Data:** Same feature CSV (and optional Layer 2 CSV) as dissociation classifier; task = control_vs_svppa or sv_vs_nflf.
- **Method:** Runs the same model sequence as in §7 (S_sem only, S_ph+S_sem, S_ph+S_sem+Δ, +extras, Layer2 only, Layer2+dissociation) via `train_and_evaluate()` (LR, 5-fold CV, StandardScaler, median imputation). No simulated data in this script.
- **Output:** JSON with per-model accuracy, F1, ROC-AUC; printed summary table.

---

## 14. Patient profile and gold graph validation

- **`get_patient_profile(transcript, gold_graph_path, s_ph, layer2_scores)`** in `graph_semantic.py`:
  - Returns a human-readable dict showing:
    - `entities_mentioned` / `entities_missed`: which gold entities were found vs missing.
    - `content_units_covered` / `content_units_missed`: which CU IDs matched vs missing.
    - `cu_details`: per-CU breakdown (id, kind, gold description, matched).
    - `s_sem`, `s_sem_nodes`, `s_sem_cu`: semantic adequacy scores.
    - `s_ph`, `delta`: if s_ph passed in, computes Δ = S_ph − S_sem.
    - `layer2`: Layer 2 construct scores if provided.
    - `interpretation`: human-readable summary (e.g. "Δ = 0.36 (phonology > semantics, svPPA pattern)").
  - Enables per-patient interpretability profiles for the paper.

- **`validate_gold_graph(gold_graph_path)`** in `graph_semantic.py`:
  - Validates that the gold graph JSON has `nodes`, `edges`, `attributes` fields.
  - Checks `content_units` field exists and all CU references point to valid node IDs.
  - Returns `{"valid": bool, "errors": list, "warnings": list, "node_count", "edge_count", "cu_count"}`.
  - Raises informative errors if structure is invalid.

---

---

## 15. Statistical conventions

- **Tests:** **Mann-Whitney U** (two-sided) via `scipy.stats.mannwhitneyu` for group comparisons (non-parametric).
- **Effect size:** **Cliff’s delta** = (concordant − discordant) / (n1×n2). Interpretation: |δ| &lt; 0.147 negligible; 0.147–0.33 small; 0.33–0.474 medium; ≥ 0.474 large.
- **Cross-validation:** **Stratified K-fold** (K=5), shuffle=True, random_state=42. Scaling and imputation are fit on train fold only when applicable.
- **Reported metrics:** Accuracy, precision, recall, F1 (binary or macro), ROC-AUC; where relevant, Mann-Whitney p and Cliff’s delta on predicted probabilities or raw features.

---

## 16. Run order (paper pipeline)

1. `fix_layer2_population_norm.py` → `layer2_fixed.csv`, `population_l1_stats.json`  
2. `run_full_analysis.py` → `ppa_dissociation_all.csv`, `ppa_stats.json`  
3. Merge (fixed L2, α=0.5) → `ppa_dissociation_fixed_l2.csv`  
4. `train_dissociation_classifier.py --task control_vs_svppa` → control vs svPPA results  
5. `train_sv_vs_nflf_real.py` → sv vs nflv real-only results (main table)  
Optional: `run_llm_baseline.py`; `generate_simulated_svppa.py` + dissociation classifier with `--simulated-csv` for augmentation comparison; `run_ablation_alpha_sweep.py`; `run_ablation_study.py`.
