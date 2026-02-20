"""Configuration dataclasses for svPPA pipeline paths and model settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PathConfig:
    """Paths for data, cache, and outputs."""

    manifest_path: Path = Path("data/manifest.csv")
    audio_dir: Path = Path("data/audio")
    cache_dir: Path = Path("data/cache")
    output_dir: Path = Path("outputs")
    model_dir: Path = Path("outputs/models")

    def __post_init__(self) -> None:
        for p in (self.cache_dir, self.output_dir, self.model_dir):
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class ASRConfig:
    """ASR backend and Whisper-specific settings."""

    backend: str = "whisper"
    whisper_model: str = "base"
    device: str = "cpu"
    language: str | None = None
    # Extra kwargs passed to backend
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PhoneticConfig:
    """Phonetic encoder (external callable) settings."""

    enabled: bool = True
    # Module path or callable name for lab-provided encoder
    encoder_path: str | None = None
    # Stub returns empty/placeholder if encoder not available
    use_stub_if_unavailable: bool = True


@dataclass
class FeatureConfig:
    """Feature extraction and Layer 2 settings."""

    # spaCy model for POS/dependency
    spacy_model: str = "en_core_web_sm"
    # Sentence embedding model for semantic distance (optional)
    sentence_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    use_semantic_distance: bool = True  # False to stub and return NaN
    # Spatiosemantic graph (Level 2): S_sem from picnic scene graph
    use_graph_semantic: bool = True
    gold_graph_path: Path | None = None  # Default: data/graph/picnic_gold_star.json
    # Layer 2: mapping from construct index to (Layer 1 indices, weights)
    # Default equal weights; override in config or code
    layer2_weights: dict[int, list[tuple[int, float]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.layer2_weights:
            # Default: 10 constructs, each with equal-weight Layer 1 indices
            self.layer2_weights = {
                0: [(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0)],   # Reduced speech rate
                1: [(5, 1.0), (6, 1.0), (7, 1.0), (9, 1.0)],             # Reduced length/complexity
                2: [(7, 1.0), (14, 1.0), (15, 1.0), (16, 1.0)],          # Agrammatism
                3: [(15, 1.0), (16, 1.0), (12, 1.0)],                    # Paragrammatism
                4: [(17, 1.0), (18, 1.0), (2, 1.0), (8, 1.0)],           # Anomia
                5: [(10, 1.0), (11, 1.0), (13, 1.0), (12, 1.0)],         # Empty speech
                6: [(19, 1.0), (12, 1.0), (10, 1.0)],                    # Semantic paraphasias
                7: [(15, 1.0), (18, 1.0)],                                # Phonemic paraphasias
                8: [(18, 1.0), (8, 1.0), (4, 1.0)],                      # Self-correction
                9: [(0, 1.0), (5, 1.0), (10, 1.0), (11, 1.0), (19, 1.0)], # Overall impairment
            }


@dataclass
class TrainConfig:
    """Logistic regression training settings."""

    target_column: str = "label"
    use_layer1: bool = False  # If True, concatenate Layer 1 + Layer 2 for training
    use_layer2: bool = True
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    penalty: str = "l2"
    C: float = 1.0
    class_weight: str = "balanced"
    max_iter: int = 1000


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""

    paths: PathConfig = field(default_factory=PathConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    phonetic: PhoneticConfig = field(default_factory=PhoneticConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    # Optional: sample rate for audio normalization
    target_sample_rate: int = 16000
    # Optional VAD/segmentation
    vad_enabled: bool = False
    vad_min_silence_ms: int = 300
    vad_min_speech_ms: int = 250
