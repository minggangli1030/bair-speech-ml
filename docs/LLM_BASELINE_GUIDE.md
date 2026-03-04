# LLM Baseline for PPA Classification — Full Prompt & Guide

This guide documents the LLM evaluation baseline: prompts, metrics (Accuracy, F1, AUC), and how to run it.

---

## 1. Tasks

| Task | Description | Labels |
|------|-------------|--------|
| **Control vs svPPA** | Binary: healthy control vs semantic variant PPA | `control`, `svPPA` |
| **svPPA vs nflv** | Binary: semantic PPA vs non-semantic (nfvPPA + lvPPA) | `svPPA`, `nflv` |

Samples are **balanced** per task (roughly 50/50 per class up to `--limit`). Controls are only used in the first task; the second uses only PPA subtypes.

---

## 2. Metrics

- **Accuracy (Acc):** Fraction of correct predictions.
- **F1:** Binary F1 score (positive class = svPPA for both tasks).
- **AUC (ROC-AUC):** Area under the ROC curve. Computed from binary predictions (no probability API); useful for comparison with other classifiers.

---

## 3. Full Prompts

### 3.1 Task 1: Control vs svPPA

**Zero-shot (instruction only):**

```
You are a clinical linguist. Below is a speech transcript from a picture description task (picnic scene). The speaker may be a healthy control or have semantic variant PPA (svPPA).

Classify the transcript into exactly ONE of:
- control (healthy, normal picture description)
- svPPA (semantic variant PPA: word-finding difficulty, empty speech, semantic errors)

Reply with only one token: control or svPPA
```

**Few-shot (instruction + examples):**

```
You are a clinical linguist. Below is a speech transcript from a picture description task (picnic scene). The speaker may be a healthy control or have semantic variant PPA (svPPA).

Classify the transcript into exactly ONE of:
- control (healthy, normal picture description)
- svPPA (semantic variant PPA: word-finding difficulty, empty speech, semantic errors)

Reply with only one token: control or svPPA

Examples:
Transcript: There's a family at a picnic. The mother is spreading a blanket. The father is holding a basket. Two children are playing with a ball near the trees.
Classification: control

Transcript: There's the thing with the... the people and the... I don't know what you call it. The thing for sitting on. And something for the food.
Classification: svPPA
```

**User message (per sample):**  
`Transcript:\n\n{participant transcript text (up to 4000 chars)}`

---

### 3.2 Task 2: svPPA vs nflv

**Zero-shot (instruction only):**

```
You are a clinical linguist. Below is a speech transcript from a picture description task (picnic scene). The speaker may have primary progressive aphasia (PPA). 

Classify the transcript into exactly ONE of:
- svPPA (semantic variant: word-finding difficulty, empty speech, semantic errors)
- nflv (non-semantic PPA: nfvPPA or lvPPA: grammar/speech rate/phonological issues)

Reply with only one token: svPPA or nflv
```

**Few-shot (instruction + examples):**

```
You are a clinical linguist. Below is a speech transcript from a picture description task (picnic scene). The speaker may have primary progressive aphasia (PPA). 

Classify the transcript into exactly ONE of:
- svPPA (semantic variant: word-finding difficulty, empty speech, semantic errors)
- nflv (non-semantic PPA: nfvPPA or lvPPA: grammar/speech rate/phonological issues)

Reply with only one token: svPPA or nflv

Examples:
Transcript: They're having a... the outdoor thing. With the... I forget the word. The thing you put food on. And the person is... the lady is doing something with the cloth.
Classification: svPPA

Transcript: The fam... the family is at the picnic. He is... he is put- putting the basket. She is spread the blanket. The child are play.
Classification: nflv
```

**User message (per sample):**  
`Transcript:\n\n{participant transcript text (up to 4000 chars)}`

---

## 4. How to Run

### Prerequisites

- **API key:** Set `DASHSCOPE_API_KEY` (Qwen/DashScope). Never commit the key; use `.env` or export.
- **Data:** Manifest `data/manifest_ppa_full.csv` and pipeline outputs under `data/ppa_pipeline_outputs/` (one `*_feature_extraction.txt` per audio with transcript).

### Commands

**Both tasks, 50 samples per task, few-shot (recommended):**
```bash
export DASHSCOPE_API_KEY=your_key
python scripts/run_llm_baseline.py --task both --limit 50 --few-shot
```

**One task only:**
```bash
# Control vs svPPA
python scripts/run_llm_baseline.py --task control_vs_svppa --limit 50 --few-shot

# svPPA vs nflv
python scripts/run_llm_baseline.py --task binary --limit 50 --few-shot
```

**Full manifest (no limit):**
```bash
python scripts/run_llm_baseline.py --task both --limit 0 --few-shot
```

**Zero-shot (no examples in prompt):**
```bash
python scripts/run_llm_baseline.py --task both --limit 50
```

**Larger/faster model:**
```bash
python scripts/run_llm_baseline.py --task both --limit 50 --few-shot --model qwen3.5-plus
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--task` | `both` | `control_vs_svppa`, `binary`, `multiclass`, or `both` |
| `--limit` | `50` | Max samples per task; `0` = use all eligible |
| `--few-shot` | off | Add 2 example transcript→label pairs to the prompt |
| `--model` | `qwen-turbo` | e.g. `qwen-turbo`, `qwen3.5-plus` |
| `--manifest` | `data/manifest_ppa_full.csv` | Manifest CSV |
| `--pipeline-dir` | `data/ppa_pipeline_outputs` | Dir with `*_feature_extraction.txt` |
| `--output` | `outputs/llm_baseline_predictions.csv` | Output CSV |

---

## 5. Output

- **Console:** For each task, prints `Acc`, `F1`, and `AUC`, then a short summary for both tasks. Run parameters and the full system prompt are printed for reproducibility.
- **CSV:** `outputs/llm_baseline_predictions.csv` with columns: `subject_id`, `session`, `subtype`, `label`, `prediction`, `true_binary`, `pred_binary`, `run_task`.

---

## 6. Example Summary

```
========== Summary (both tasks) ==========
  Control vs svPPA: n=50  Acc=0.560  F1=0.694  AUC=0.560
  svPPA vs nflv: n=50  Acc=0.500  F1=0.444  AUC=0.500
```

---

## 7. Reference

Phan et al. (2024) Interspeech — LLM for content assessment / classification.  
https://www.isca-archive.org/interspeech_2024/phan24_interspeech.html
