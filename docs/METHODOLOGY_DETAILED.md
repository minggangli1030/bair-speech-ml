# Detailed Pipeline Methodology

This document provides a complete procedural specification of the PPA classification pipeline, including exact variable names, formulas, statistical test definitions, and what is compared at each step.

---

## Table of Contents

1. [Data Inputs](#1-data-inputs)
2. [Pipeline Output Parsing](#2-pipeline-output-parsing)
3. [Layer 1 Features (20 Proxies)](#3-layer-1-features-20-proxies)
4. [Layer 2 Constructs (Population-Normed)](#4-layer-2-constructs-population-normed)
5. [S_ph: Phonological Adequacy](#5-s_ph-phonological-adequacy)
6. [S_sem: Semantic Adequacy](#6-s_sem-semantic-adequacy)
7. [Dissociation Index (Δ)](#7-dissociation-index-δ)
8. [Statistical Tests](#8-statistical-tests)
9. [Classification Models](#9-classification-models)
10. [Pipeline Execution Order](#10-pipeline-execution-order)

---

## 1. Data Inputs

### 1.1 Manifest File

**File:** `data/manifest_ppa_full.csv`

**Columns:**
| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | string | Unique participant identifier |
| `session` | string | Session identifier |
| `subtype` | string | One of: `control`, `svppa`, `nfvppa`, `lvppa` |
| `label` | int | 0=control, 1=svPPA, 2=lvPPA, 3=nfvPPA |
| `audio_path` | string | Path to audio file |

### 1.2 Pipeline Output Files

**Location:** `data/ppa_pipeline_outputs/{stem}_feature_extraction.txt`

Where `stem = Path(audio_path).stem`

**Contents:** Text report with per-segment table containing:
- `Timestamp` (start–end)
- `Speaker` (Participant or Clinician)
- `ASR Transcription`
- `Phonemic Encoding` (SylWav phoneme sequence)

### 1.3 Gold Scene Graph

**File:** `data/graph/picnic_gold_star.json`

**Structure:**
```json
{
  "nodes": [21 entities],      // e.g., {"id": "boy_1", "type": "person", "label": "boy"}
  "edges": [23 relations],     // e.g., {"subj": "boy_1", "rel": "holding", "obj": "string_1"}
  "attributes": [6 attrs],     // e.g., {"node": "kite_1", "attr": "state", "value": "flying"}
  "content_units": [12 CUs]    // Clinical content units (9 edges + 3 attributes)
}
```

**Content Units (CUs):**

| ID | Kind | Description |
|----|------|-------------|
| CU_E01 | edge | blanket on grass |
| CU_E02 | edge | woman sitting_on blanket |
| CU_E03 | edge | man sitting_on blanket |
| CU_E04 | edge | woman reading book |
| CU_E05 | edge | boy holding string |
| CU_E06 | edge | string attached_to kite |
| CU_E07 | edge | kite in sky |
| CU_E08 | edge | dog chasing ball |
| CU_E09 | edge | boat on lake |
| CU_A01 | attr | kite flying |
| CU_A02 | attr | dog running |
| CU_A03 | attr | family picnicking |

---

## 2. Pipeline Output Parsing

**Function:** `parse_pipeline_output()` in `src/features/phoneme_adequacy.py`

### 2.1 Procedure

1. Read pipeline output file
2. Regex extract per-segment: `Timestamp`, `Speaker`, `ASR Transcription`, `Phonemic Encoding`
3. Filter to **Participant-only** segments (exclude Clinician)
4. Concatenate participant segments

### 2.2 Outputs

| Variable | Type | Description |
|----------|------|-------------|
| `transcript` | str | Space-joined participant text |
| `phonemes` | str | Space-joined SylWav phoneme sequence |
| `segments` | list | `[{start, end, speaker, transcript, phonemes}, ...]` |

### 2.3 Transcript Dict Format

For downstream processing, segments are converted to:

```python
transcript_dict = {
    "text": "full participant transcript",
    "words": [
        {"word": "the", "start": 0.0, "end": 0.2},
        {"word": "boy", "start": 0.2, "end": 0.5},
        ...
    ]
}
```

Word timestamps are interpolated from segment duration.

---

## 3. Layer 1 Features (20 Proxies)

**Script:** `src/features/layer1.py`

### 3.1 Input

`transcript_dict` with word-level timestamps

### 3.2 Utterance Segmentation

Words are split into utterances by **pause threshold**:
- `PAUSE_THRESHOLD = 0.5` seconds
- Pause = gap between `words[i].end` and `words[i+1].start`
- If pause ≥ threshold → new utterance

### 3.3 Feature Definitions

| # | Feature Name | Formula / Description |
|---|--------------|----------------------|
| 1 | `wpm` | `(word_count / total_duration) * 60` |
| 2 | `articulation_rate_phones_per_sec` | `phoneme_count / speech_duration` (non-silence) |
| 3 | `mean_pause_duration` | `mean(pause_durations)` where pause ≥ 0.3s |
| 4 | `pause_to_speech_ratio` | `total_pause_time / total_speech_time` |
| 5 | `initial_response_latency` | `words[0].start` (time to first word) |
| 6 | `mlu` | Mean Length of Utterance: `mean(words_per_utterance)` |
| 7 | `median_utterance_length` | `median(words_per_utterance)` |
| 8 | `clause_count_per_utterance` | Clauses detected via spaCy / utterance count |
| 9 | `abandoned_utterance_rate` | Utterances without terminal punctuation / total |
| 10 | `utterance_count` | Total number of utterances |
| 11 | `open_class_closed_class_ratio` | Open class words / closed class words |
| 12 | `content_word_density` | Content words / total words |
| 13 | `type_token_ratio` | `unique_words / total_words` (TTR) |
| 14 | `pronoun_usage_ratio` | Pronouns / total words |
| 15 | `function_word_omission_rate` | Heuristic: missing determiners, aux verbs |
| 16 | `morphological_error_rate` | Heuristic: verb form errors |
| 17 | `dependency_parse_completeness` | % of tokens with valid dependency head |
| 18 | `word_finding_pause_duration` | Mean pause ≥ 1.0s before content words |
| 19 | `filled_pause_frequency` | Count of {um, uh, er, ah, eh, hm, hmm} / word_count |
| 20 | `semantic_distance_to_expected` | Cosine distance to expected topic (disabled: returns NaN) |

### 3.4 Constants

```python
WORD_FINDING_PAUSE_THRESHOLD = 1.0  # seconds
MIN_PAUSE_GAP = 0.3                  # minimum gap to count as pause
FILLED_PAUSES = {"um", "uh", "er", "ah", "eh", "hm", "hmm", "uhm", "umhm"}
```

---

## 4. Layer 2 Constructs (Population-Normed)

**Script:** `scripts/fix_layer2_population_norm.py`

### 4.1 Purpose

Convert Layer 1 features to clinician-aligned constructs using **population-level** z-scoring (not per-sample).

### 4.2 Procedure

#### Step 1: Compute Layer 1 for All Samples

```python
for each sample in manifest:
    L1[sample] = compute_layer1_features(transcript_dict)
# Result: (n_samples × 20) matrix
```

#### Step 2: Compute Population Statistics

```python
for each feature f in LAYER1_NAMES:
    population_mean[f] = nanmean(L1[:, f])
    population_std[f] = nanstd(L1[:, f])
    if population_std[f] == 0:
        population_std[f] = 1.0
```

Output saved to: `outputs/population_l1_stats.json`

#### Step 3: Z-Score Each Sample Against Population

```python
for each sample:
    z_vec[sample] = (L1[sample] - population_mean) / population_std
```

#### Step 4: Compute Layer 2 Constructs

For each of 10 constructs, compute weighted average of relevant z-scores, then map to [0, 1]:

```python
L2[construct] = (tanh(weighted_z_avg) + 1) / 2
```

### 4.3 Layer 2 Construct Definitions

| Construct | Layer 1 Features Used |
|-----------|----------------------|
| `l2_reduced_speech_rate` | wpm, articulation_rate, mean_pause_duration, pause_to_speech_ratio, initial_response_latency |
| `l2_reduced_length_complexity` | mlu, median_utterance_length, clause_count_per_utterance, utterance_count |
| `l2_agrammatism` | clause_count_per_utterance, function_word_omission_rate, morphological_error_rate, dependency_parse_completeness |
| `l2_paragrammatism` | morphological_error_rate, dependency_parse_completeness, type_token_ratio |
| `l2_anomia` | word_finding_pause_duration, filled_pause_frequency, mean_pause_duration, abandoned_utterance_rate |
| `l2_empty_speech` | utterance_count, open_class_closed_class_ratio, pronoun_usage_ratio, type_token_ratio |
| `l2_semantic_paraphasias` | semantic_distance_to_expected, type_token_ratio, utterance_count |
| `l2_phonemic_paraphasias_neologisms` | morphological_error_rate, filled_pause_frequency |
| `l2_self_correction` | filled_pause_frequency, abandoned_utterance_rate, initial_response_latency |
| `l2_overall_communication_impairment` | wpm, mlu, utterance_count, open_class_closed_class_ratio, semantic_distance_to_expected |

### 4.4 Communication Adequacy from Layer 2

```python
s_sem_layer2_adequacy = 1 - mean(all 10 L2 constructs)
```

Where the 10 constructs are:
- `l2_reduced_speech_rate`
- `l2_reduced_length_complexity`
- `l2_agrammatism`
- `l2_paragrammatism`
- `l2_anomia`
- `l2_empty_speech`
- `l2_semantic_paraphasias`
- `l2_phonemic_paraphasias_neologisms`
- `l2_self_correction`
- `l2_overall_communication_impairment`

**Interpretation:** Higher = better overall communication adequacy; lower values indicate impairment. Using all 10 constructs (rather than just 3 semantic-specific ones) improves the dissociation index discrimination.

### 4.5 Output

**File:** `outputs/layer2_fixed.csv`

Columns: `subject_id`, `session`, `subtype`, `l2_anomia`, `l2_empty_speech`, ..., `s_sem_layer2_adequacy_fixed`

---

## 5. S_ph: Phonological Adequacy

**Script:** `src/features/phoneme_adequacy.py`

### 5.1 Input

- `transcript`: ASR text from participant
- `phonemes`: SylWav phoneme string from audio

### 5.2 Procedure

#### Step 1: Generate Expected Phonemes (G2P)

```python
for each word in transcript:
    if word in CMU_DICT:
        expected_phonemes.extend(CMU_DICT[word])
    else:
        expected_phonemes.extend(character_fallback(word))
```

#### Step 2: Parse Observed Phonemes

```python
observed = phonemes.split()
observed = [p for p in observed if p != "SIL"]  # remove silence
```

#### Step 3: Compute Edit Distance

```python
edit_distance = levenshtein(expected_phonemes, observed_phonemes)
```

#### Step 4: Compute Phoneme Error Rate (PER)

```python
PER = edit_distance / max(len(expected), len(observed))
```

#### Step 5: Compute S_ph

```python
S_ph = 1 - PER
S_ph = clip(S_ph, 0, 1)
```

### 5.3 Additional Outputs

| Variable | Formula |
|----------|---------|
| `phoneme_error_rate` | PER (same as 1 - S_ph) |
| `silence_ratio` | count(SIL) / len(raw_phonemes) |
| `phonemes_per_word` | len(observed) / word_count |

### 5.4 Interpretation

- **S_ph ≈ 1.0**: Near-perfect phonological production (phonemes match expected)
- **S_ph ≈ 0.5**: ~50% phoneme errors
- **S_ph ≈ 0.0**: Complete phonological breakdown

---

## 6. S_sem: Semantic Adequacy

**Script:** `src/features/graph_semantic.py`

### 6.1 Overview

S_sem measures how completely a patient's transcript covers the semantic content of the picnic picture.

### 6.2 Step 1: Entity Extraction

**Canonical Entity Map:**

```python
ENTITY_CANON = {
    "boy": ["boy", "son", "child", "kid", "little boy"],
    "girl": ["girl", "daughter", "little girl"],
    "man": ["man", "father", "dad", "husband"],
    "woman": ["woman", "mother", "mom", "wife", "lady"],
    "dog": ["dog", "puppy", "pet"],
    "kite": ["kite"],
    "kite string": ["kite string", "string", "line"],
    "blanket": ["blanket", "sheet", "cloth", "mat"],
    ...  # 21 entities total
}
```

**Procedure:**
```python
for each word in transcript:
    for canon_entity, variants in ENTITY_CANON:
        if word in variants:
            found_entities[canon_entity] = {"id": f"{canon_entity}_1", "label": canon_entity}
```

### 6.3 Step 2: Relation Extraction (DIRECTIONAL)

Relations are extracted respecting **word order** in the transcript:

#### Spatial Relations

```python
SPATIAL_RELATIONS = ["on", "in", "near", "next to", "beside", "behind", "by", "under", "above"]

for each spatial_word found at position i:
    subj = nearest_entity_BEFORE(i, window=5)
    obj = nearest_entity_AFTER(i, window=5)
    if subj and obj and subj != obj:
        edges.append((subj, relation, obj))
```

#### Action Relations

```python
ACTION_VERBS = {
    "hold": ["hold", "holding", "has", "have", "got"],
    "sit": ["sit", "sitting", "sat", "seated"],
    "read": ["read", "reading", "reads"],
    "chase": ["chase", "chasing", "chased", "running after", "going after", "after"],
    ...
}

for each verb found at position i:
    subj = nearest_entity_BEFORE(i, window=5)
    obj = nearest_entity_AFTER(i, window=5)
    if subj and obj and subj != obj:
        edges.append((subj, verb_relation, obj))
```

**CRITICAL: Directionality**

- "boy holding string" → extracts `(boy, holding, string)` ✓
- "string holding boy" → extracts `(string, holding, boy)` ✗ (won't match gold CU)

This ensures nonsensical statements are correctly penalized.

### 6.4 Step 3: Attribute Extraction

```python
if "kite" in entities and any(w in text for w in ["fly", "flying", "flies"]):
    attributes.append(("kite", "state", "flying"))

if "dog" in entities and any(w in text for w in ["run", "running", "runs"]):
    attributes.append(("dog", "activity", "running"))

if "family" in entities and any(w in text for w in ["picnic", "picnicking", "eating"]):
    attributes.append(("family", "activity", "picnicking"))
```

### 6.5 Step 4: Content Unit (CU) Scoring

Compare extracted edges/attributes against the **12 clinical content units**:

```python
matched_cus = []
for cu in gold_graph["content_units"]:
    if cu["kind"] == "edge":
        gold_tuple = (cu["subj_label"], cu["rel"], cu["obj_label"])
        if gold_tuple in extracted_edges:
            matched_cus.append(cu["id"])
    elif cu["kind"] == "attr":
        gold_tuple = (cu["node_label"], cu["attr"], cu["value"])
        if gold_tuple in extracted_attrs:
            matched_cus.append(cu["id"])

s_sem_cu = len(matched_cus) / 12
```

### 6.6 Step 5: Node Recall

```python
gold_node_labels = {node["label"] for node in gold_graph["nodes"]}
matched_nodes = extracted_entities.keys() & gold_node_labels
s_sem_nodes = len(matched_nodes) / len(gold_node_labels)  # recall against 21 nodes
```

### 6.7 Step 6: Final S_sem Formula

```python
S_sem_graph = 0.4 * s_sem_nodes + 0.6 * s_sem_cu
```

**Rationale:** Content units (relational content) are weighted more heavily than simple entity mentions because:
1. CUs capture clinically meaningful semantic content (Nicholas & Brookshire 1993)
2. svPPA patients may mention entities but fail to produce correct relations
3. Relational deficits are more diagnostically relevant

### 6.8 Composite S_sem (with Layer 2)

```python
S_sem = alpha * S_sem_graph + (1 - alpha) * s_sem_layer2_adequacy
# Default alpha = 0.5
```

### 6.9 Output Variables

| Variable | Description |
|----------|-------------|
| `s_sem` | Final composite semantic adequacy |
| `s_sem_graph` | Graph-based semantic adequacy |
| `s_sem_nodes` | Entity recall (0.0–1.0) |
| `s_sem_cu` | Content unit coverage (0.0–1.0) |
| `matched_cus` | List of matched CU IDs |
| `missed_cus` | List of missed CU IDs |
| `s_sem_layer2_adequacy` | Layer 2-based semantic adequacy |

---

## 7. Dissociation Index (Δ)

### 7.1 Formula

```python
Δ = S_ph - S_sem
```

### 7.2 Interpretation

| Δ Value | Interpretation |
|---------|----------------|
| Δ >> 0 | Phonology preserved, semantics impaired → **svPPA pattern** |
| Δ ≈ 0 | Balanced phonology and semantics |
| Δ << 0 | Semantics preserved, phonology impaired → nfvPPA pattern |

### 7.3 Theoretical Basis

Based on the dual-stream model (Hickok & Poeppel 2007):
- **Dorsal stream**: phonological processing
- **Ventral stream**: semantic processing

svPPA selectively impairs ventral stream → preserved S_ph, impaired S_sem → Δ > 0

---

## 8. Statistical Tests

### 8.1 Mann-Whitney U Test

**Purpose:** Non-parametric comparison of two groups (does not assume normality)

**Implementation:**
```python
from scipy.stats import mannwhitneyu

def mann_whitney_u(group1_values, group2_values):
    # Remove NaN values
    x = group1_values[~np.isnan(group1_values)]
    y = group2_values[~np.isnan(group2_values)]
    
    stat, p_value = mannwhitneyu(x, y, alternative="two-sided")
    return stat, p_value
```

**Where it's used and on WHAT variables:**

| Comparison | Variable Being Compared | Script |
|------------|------------------------|--------|
| Control vs svPPA | `s_sem` values | `run_full_analysis.py` |
| Control vs svPPA | `s_ph` values | `train_dissociation_classifier.py` |
| Control vs svPPA | `delta` values | `train_dissociation_classifier.py` |
| svPPA vs lvPPA | `s_sem` values | `run_full_analysis.py` |
| svPPA vs nfvPPA | `s_sem` values | `run_full_analysis.py` |
| svPPA vs nflv | `predicted P(svPPA)` from classifier | `train_sv_vs_nflf_real.py` |
| svPPA vs nflv | Individual features (`l2_anomia`, etc.) | `train_sv_vs_nflf_real.py` |

**Example from actual results:**

```
s_sem (control vs svppa):
  Group 1 (control): n=25, values=[0.502, 0.515, ...]
  Group 2 (svppa): n=42, values=[0.441, 0.423, ...]
  
  Mann-Whitney U = 935.0
  p-value = 1.10e-07 (highly significant)
```

### 8.2 Cliff's Delta (Effect Size)

**Purpose:** Quantify the magnitude of difference between groups

**Formula:**
```python
def cliffs_delta(x, y):
    concordant = 0  # x[i] > y[j]
    discordant = 0  # x[i] < y[j]
    
    for xi in x:
        for yj in y:
            if xi > yj:
                concordant += 1
            elif xi < yj:
                discordant += 1
    
    delta = (concordant - discordant) / (len(x) * len(y))
    return delta  # range: [-1, +1]
```

**Interpretation (Romano et al. 2006):**

| |δ| | Effect Size |
|-----|-------------|
| < 0.147 | Negligible |
| 0.147 – 0.33 | Small |
| 0.33 – 0.474 | Medium |
| ≥ 0.474 | Large |

**Example from actual results:**

```
s_sem (control vs svppa):
  Cliff's δ = 0.781 (large effect)
  
  Interpretation: In 78.1% of pairwise comparisons,
  control s_sem > svppa s_sem
```

### 8.3 Statistical Tests on Predicted Probabilities

In `train_sv_vs_nflf_real.py`, after training classifiers:

```python
# Get predicted probability of svPPA class
y_prob = cross_val_predict(model, X, y, cv=5, method="predict_proba")[:, 1]

# Split by true label
prob_sv = y_prob[y == 1]    # P(svPPA) for actual svPPA samples
prob_nflv = y_prob[y == 0]  # P(svPPA) for actual nfv+lv samples

# Mann-Whitney: are predicted probabilities different between groups?
U, p = mann_whitney_u(prob_sv, prob_nflv)
delta = cliffs_delta(prob_sv, prob_nflv)
```

**Interpretation:** If p < 0.05 and δ is medium/large, the classifier successfully discriminates between groups.

---

## 9. Classification Models

### 9.1 Task 1: Control vs svPPA

**Script:** `train_dissociation_classifier.py --task control_vs_svppa`

**Data Filter:**
```python
df = df[df["label"].isin([0, 1])]  # 0=control, 1=svPPA
y = df["label"].values
```

**Feature Sets (Ablation):**

| Model | Features |
|-------|----------|
| S_sem_only | `[s_sem]` |
| S_ph_S_sem | `[s_ph, s_sem]` |
| S_ph_S_sem_Delta | `[s_ph, s_sem, delta]` |
| Dissociation_Plus | `[s_ph, s_sem, delta, phoneme_error_rate, silence_ratio]` |
| Layer2_Only | `[l2_anomia, l2_empty_speech, ..., l2_overall_communication_impairment]` |
| Layer2_Plus_Dissociation | `[s_ph, s_sem, delta] + all L2 features` |

**Classifier:**
```python
LogisticRegression(
    C=1.0,
    class_weight="balanced",
    max_iter=1000,
    solver="lbfgs",
    random_state=42
)
```

**Evaluation:**
```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, test_idx in cv.split(X, y):
    # Fit scaler on train only
    scaler = StandardScaler().fit(X[train_idx])
    X_train = scaler.transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])
    
    # Median imputation for NaN
    X_train = np.nan_to_num(X_train, nan=np.nanmedian(X_train, axis=0))
    X_test = np.nan_to_num(X_test, nan=np.nanmedian(X_train, axis=0))
    
    model.fit(X_train, y[train_idx])
    y_pred[test_idx] = model.predict(X_test)
    y_prob[test_idx] = model.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
auc = roc_auc_score(y, y_prob)
```

### 9.2 Task 2: svPPA vs (nfvPPA + lvPPA)

**Script:** `train_sv_vs_nflf_real.py`

**Data Filter:**
```python
df = df[df["subtype"].isin(["svppa", "nfvppa", "lvppa"])]
y = (df["subtype"] == "svppa").astype(int)  # 1=svPPA, 0=nfv+lv
```

**Feature Sets:**

| Model | Features |
|-------|----------|
| S_ph_S_sem | `[s_ph, s_sem]` |
| S_ph_S_sem_graph | `[s_ph, s_sem, s_sem_graph]` |
| S_ph_S_sem_Delta | `[s_ph, s_sem, delta]` |
| +Acoustic | `+ [phoneme_error_rate, silence_ratio]` |
| +L2_semantic | `+ [l2_anomia, l2_empty_speech, l2_semantic_paraphasias]` |
| +L2_gram | `+ [l2_agrammatism, l2_reduced_speech_rate]` |
| +L2_all | `+ all 10 L2 constructs` |
| Full_clinical | Base + delta + acoustic + L2_semantic + L2_gram |
| Full_all_L2 | Base + delta + acoustic + all L2 |

**Classifiers:**
```python
classifiers = {
    "LR": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
    "RF": RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42)
}
```

### 9.3 Metrics Reported

| Metric | Formula |
|--------|---------|
| Accuracy | `correct_predictions / total_samples` |
| Precision | `TP / (TP + FP)` |
| Recall | `TP / (TP + FN)` |
| F1 | `2 * (Precision * Recall) / (Precision + Recall)` |
| ROC-AUC | Area under ROC curve (probability-based) |
| Mann-Whitney p | On predicted P(svPPA) between groups |
| Cliff's δ | On predicted P(svPPA) between groups |

---

## 10. Pipeline Execution Order

### 10.1 Main Paper Pipeline

```bash
# Step 1: Fix Layer 2 with population normalization
python scripts/fix_layer2_population_norm.py
# Output: outputs/layer2_fixed.csv, outputs/population_l1_stats.json

# Step 2: Run full analysis (S_ph, S_sem, Δ)
python scripts/run_full_analysis.py
# Output: outputs/ppa_dissociation_full.csv, outputs/ppa_stats.json

# Step 3: Merge fixed Layer 2 into feature table
python -c "
import pandas as pd
df = pd.read_csv('outputs/ppa_dissociation_full.csv')
l2 = pd.read_csv('outputs/layer2_fixed.csv')
df = df.merge(l2, on=['subject_id', 'session'], how='left', suffixes=('', '_l2'))

# Use fixed Layer 2 adequacy
df['s_sem_layer2_adequacy'] = df['s_sem_layer2_adequacy_fixed'].fillna(df['s_sem_layer2_adequacy'])

# Recompute composite S_sem with alpha=0.5
alpha = 0.5
df['s_sem'] = alpha * df['s_sem_graph'] + (1 - alpha) * df['s_sem_layer2_adequacy']
df['delta'] = df['s_ph'] - df['s_sem']

df.to_csv('outputs/ppa_dissociation_fixed_l2.csv', index=False)
"

# Step 4: Control vs svPPA classifier
python scripts/train_dissociation_classifier.py \
    --task control_vs_svppa \
    --features outputs/ppa_dissociation_fixed_l2.csv
# Output: outputs/classifier_results.json

# Step 5: svPPA vs nfv+lv classifier (main paper result)
python scripts/train_sv_vs_nflf_real.py
# Output: outputs/sv_vs_nflf_real_results.json
```

### 10.2 Optional Scripts

```bash
# LLM Baseline
python scripts/run_llm_baseline.py --task multiclass --model gpt-4o

# Synthetic Data Augmentation
python scripts/generate_simulated_svppa.py

# Ablation: Alpha sweep for composite S_sem
python scripts/run_ablation_alpha_sweep.py

# Ablation: Model variants
python scripts/run_ablation_study.py
```

---

## Appendix A: Complete Variable Reference

### Output CSV Columns (`ppa_dissociation_full.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | str | Participant ID |
| `session` | str | Session ID |
| `subtype` | str | control/svppa/nfvppa/lvppa |
| `label` | int | 0/1/2/3 |
| `s_ph` | float | Phonological adequacy [0,1] |
| `s_sem` | float | Composite semantic adequacy [0,1] |
| `s_sem_graph` | float | Graph-based semantic adequacy [0,1] |
| `s_sem_nodes` | float | Node recall [0,1] |
| `s_sem_cu` | float | Content unit coverage [0,1] |
| `s_sem_layer2_adequacy` | float | Layer 2-based semantic adequacy [0,1] |
| `delta` | float | S_ph - S_sem |
| `phoneme_error_rate` | float | 1 - S_ph |
| `silence_ratio` | float | Proportion of silence |
| `l2_anomia` | float | Anomia construct [0,1] |
| `l2_empty_speech` | float | Empty speech construct [0,1] |
| `l2_semantic_paraphasias` | float | Semantic paraphasias [0,1] |
| `l2_agrammatism` | float | Agrammatism construct [0,1] |
| `l2_reduced_speech_rate` | float | Reduced speech rate [0,1] |
| `l2_reduced_length_complexity` | float | Reduced length/complexity [0,1] |
| `l2_paragrammatism` | float | Paragrammatism [0,1] |
| `l2_phonemic_paraphasias_neologisms` | float | Phonemic paraphasias [0,1] |
| `l2_self_correction` | float | Self-correction behavior [0,1] |
| `l2_overall_communication_impairment` | float | Overall impairment [0,1] |

### Statistics JSON (`ppa_stats.json`)

```json
{
  "s_sem_comparisons": [
    {
      "group_1": "control",
      "group_2": "svppa",
      "n_1": 25,
      "n_2": 42,
      "u_statistic": 935.0,
      "p_value": 1.1e-07,
      "cliffs_delta": 0.781,
      "effect_size": "large"
    },
    ...
  ],
  "svppa_dissociation": {
    "n": 42,
    "delta_mean": -0.090,
    "delta_std": 0.059,
    "delta_positive_pct": 4.76,
    "s_ph_mean": 0.351,
    "s_sem_mean": 0.441
  }
}
```

---

## Appendix B: Key Formulas Summary

| Metric | Formula |
|--------|---------|
| S_ph | `1 - (edit_distance / max(len(expected), len(observed)))` |
| S_sem_nodes | `matched_entities / 21` |
| S_sem_cu | `matched_content_units / 12` |
| S_sem_graph | `0.4 * S_sem_nodes + 0.6 * S_sem_cu` |
| S_sem_L2 | `1 - mean(all 10 L2 constructs)` |
| S_sem (composite) | `0.5 * S_sem_graph + 0.5 * S_sem_L2` |
| Δ | `S_ph - S_sem` |
| L2 construct | `(tanh(weighted_z_avg) + 1) / 2` |
| Cliff's δ | `(concordant - discordant) / (n1 * n2)` |

---

*Document generated: 2026-02-27*
*Pipeline version: 2.0 (with directional graph scoring)*
