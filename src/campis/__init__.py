"""CAMPIS: clasificacion reproducible de enfermedades de hojas de papa."""

from .config import (
    AppConfig,
    AugmentationConfig,
    DataConfig,
    ModelConfig,
    PathsConfig,
    SourceConfig,
    TrainingConfig,
    load_config,
)

__all__ = [
    "AppConfig",
    "AugmentationConfig",
    "DataConfig",
    "ModelConfig",
    "PathsConfig",
    "SourceConfig",
    "TrainingConfig",
    "load_config",
]

__version__ = "1.0.0"
