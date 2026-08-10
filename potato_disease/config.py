"""Typed config loading for training/evaluation runs, backed by per-architecture YAML files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    name: str
    num_classes: int = 3
    pretrained: bool = False
    freeze_backbone: bool = False


@dataclass
class DataConfig:
    data_dir: str = "data/processed"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2


@dataclass
class TrainConfig:
    epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"
    seed: int = 42
    device: str = "auto"


@dataclass
class OutputConfig:
    checkpoint_dir: str = "checkpoints"
    experiment_dir: str = "experiments"


@dataclass
class Config:
    model: ModelConfig
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        return cls(
            model=ModelConfig(**raw["model"]),
            data=DataConfig(**raw.get("data", {})),
            train=TrainConfig(**raw.get("train", {})),
            output=OutputConfig(**raw.get("output", {})),
        )
