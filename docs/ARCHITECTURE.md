# Architecture and engineering tradeoffs

## Separate expensive inference from evaluation

`src/models/asr.py` defines an abstract transcription interface and a lazy-loaded Whisper implementation. Phonetic decoding is an optional external dependency with a stub path; stub availability does not establish a valid phonological measurement. `src/pipeline/runner.py` coordinates diarization, ASR, phoneme decoding, and reuse of existing output files.

`src/io/cache.py` hashes subject ID, input path, and artifact suffix into deterministic cache names. This avoids redundant work for repeated invocations at the same path, but does not hash file contents or model versions. Replacing audio or changing model parameters requires deliberate cache invalidation. Cache files are not automatically anonymized or transactionally written.

## Interpretable feature boundary

Layer 1 exposes acoustic and linguistic proxies. Layer 2 maps those proxies to named constructs. Graph scoring canonicalizes scene entities and uses relation patterns to recover content units; its output includes matched/missed nodes and units for inspection. The graph path is configurable, but entity dictionaries and rules remain scene-specific. This is not a learned general-purpose knowledge graph.

The phonological module uses edit distance against a small dictionary/heuristic grapheme-to-phoneme mapping. Coverage and approximation error limit interpretation; the public offline example exercises graph semantics only.

## Experiment boundary

`src/train/baseline.py` separates feature schemas, imputation medians, a scaler, and classifier artifacts. Per-feature contributions describe the fitted linear model; they are not causal explanations. Research scripts expose ablations, model families, seeds, and task-specific analyses. Historical evaluation limitations are documented separately rather than silently changing archived metrics.

## Repository layout

`src/` contains reusable modules, `scripts/` experiment entry points, `tests/` synthetic offline contracts, `docs/` methods/evidence, and `data/graph/` the reference scene specification. Generated data and outputs are kept local. Duplicate-looking experiment scripts are retained when they encode distinct cohorts or analyses; [the script index](../scripts/README.md) explains how to navigate them.
