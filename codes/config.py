from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class SplitConfig:
    train_size: float = 0.70
    validation_size: float = 0.10
    test_size: float = 0.20
    random_state: int = 42
    stratify: bool = True


@dataclass(slots=True)
class DataConfig:
    data_path: Path
    input_column: str = "Text"
    label_column: str = "type"
    max_input_length: int = 512
    max_target_length: int = 128
    prompt_style: str = "human_explanation"
    target_text_column: Optional[str] = None
    rename_drug_columns: bool = True
    split: SplitConfig = field(default_factory=SplitConfig)


@dataclass(slots=True)
class ClassifierConfig:
    encoder_name: str = "thomas-sounack/BioClinical-ModernBERT-large"
    num_heads: int = 4
    num_experts: int = 8
    top_k: int = 2
    num_classes: int = 6
    dropout: float = 0.1
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    load_balancing_weight: float = 0.01
    batch_size: int = 8
    epochs: int = 5


@dataclass(slots=True)
class GeneratorConfig:
    encoder_name: str = "thomas-sounack/BioClinical-ModernBERT-large"
    decoder_name: str = "t5-base"
    num_heads: int = 4
    num_experts: int = 8
    top_k: int = 2
    dropout: float = 0.1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    epochs: int = 3
    max_new_tokens: int = 64
    freeze_encoder: bool = True


@dataclass(slots=True)
class LoRAFineTuningConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q", "k", "v", "o", "wi", "wo")
