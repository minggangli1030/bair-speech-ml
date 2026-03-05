# Diarization and Semi-Automation Efficiency Gains

This document summarizes the experiment and results quantifying the efficiency gains from the semi-automated diarization workflow in IPSS-PPA.

## Workflow

The pipeline uses pyannote.audio for speaker diarization. Unambiguous single-speaker segments are accepted automatically; segments flagged as mixed-speaker or low-confidence are manually corrected by trained annotators. This semi-automated design reduces the manual annotation burden relative to fully manual diarization.

---

## Experiment 1: Recording-Level Pass-Through Rate

**Method:** Count how many of the 254 recordings required any manual correction versus those accepted fully automatically.

**Result:**

| Metric | Value |
|--------|-------|
| Total recordings | 254 |
| Accepted automatically (no manual intervention) | 174 |
| Required annotator correction | 80 |
| **Automatic pass-through rate** | **68.5%** |

**Efficiency interpretation:** Because manual effort scales with the number of recordings requiring review (not total corpus duration), 68.5% of recordings incur zero manual diarization time. This yields an estimated **68.5% reduction in annotation time** relative to fully manual diarization of the complete corpus.

---

## Experiment 2: Speech-Duration-Based Effort Estimation

**Method:** Parse all 254 diarized `*_feature_extraction.txt` files and sum segment durations by speaker label (Participant vs Clinician). Under the assumption that manual review is concentrated on clinician/non-patient speech (where pyannote errors are most likely), the proportion of clinician speech approximates the fraction of the diarized timeline requiring human attention.

**Procedure:**
- Regex extraction of per-segment: start time, end time, speaker label
- Sum durations for Participant and Clinician segments across all files
- Compute clinician fraction = Clinician_sec / Total_sec

**Result:**

| Metric | Value |
|--------|-------|
| Participant speech (total) | 12,879.1 s |
| Clinician speech (total) | 2,613.3 s |
| Total diarized speech | 15,492.5 s |
| **Clinician fraction** | **16.9%** |

**Efficiency interpretation:** If manual diarization effort scales with the amount of speech that requires human labeling, and the semi-automated workflow limits human review to clinician/overlap regions (~16.9% of the diarized timeline), then hands-on diarization time is reduced by approximately **83%** relative to fully manual labeling of all speech segments.

---

## Summary for Paper

| Experiment | Metric | Efficiency gain |
|------------|--------|-----------------|
| Recording-level | 68.5% auto pass-through | ~68.5% reduction in recordings needing manual diarization |
| Duration-based | 16.9% clinician fraction | ~83% reduction in hands-on diarization time |

**Suggested wording for paper:**  
*"We quantified the efficiency of semi-automation by (1) recording-level pass-through: 174 of 254 recordings (68.5%) were accepted automatically without manual diarization; and (2) duration-based estimation: clinician speech comprised 16.9% of total diarized speech (2,613 s of 15,492 s), suggesting an ~80–85% reduction in hands-on diarization time relative to fully manual labeling."*

---

## Reproduction

To reproduce the duration-based experiment:

```bash
cd /data/sarikapasumarthy/semantic-evaluation
python - << 'PY'
from pathlib import Path
import re

dir_path = Path('data/ppa_pipeline_outputs')
pattern = re.compile(r"^\s*\d+\s+([0-9:.]+) \u2192 ([0-9:.]+)\s+(Participant|Clinician)")

participant_sec = 0.0
clinician_sec = 0.0
for fp in dir_path.glob('*_feature_extraction.txt'):
    for line in fp.open():
        m = pattern.match(line)
        if not m:
            continue
        start, end, speaker = m.groups()
        def to_sec(t):
            mm, ss = t.split(':')
            return int(mm) * 60 + float(ss)
        dur = to_sec(end) - to_sec(start)
        if speaker == 'Participant':
            participant_sec += dur
        else:
            clinician_sec += dur

total = participant_sec + clinician_sec
print("Participant:", participant_sec, "s")
print("Clinician:", clinician_sec, "s")
print("Total:", total, "s")
print("Clinician fraction:", clinician_sec / total)
PY
```
