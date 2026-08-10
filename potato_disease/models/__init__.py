"""Model registry: architectures register themselves via @register_model, built via get_model()."""

from typing import Callable

import torch.nn as nn

_MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str) -> Callable:
    def decorator(builder: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        _MODEL_REGISTRY[name] = builder
        return builder

    return decorator


def get_model(name: str, **kwargs) -> nn.Module:
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Modelo desconocido '{name}'. Disponibles: {list(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[name](**kwargs)


# Importar los módulos registra cada arquitectura como efecto secundario.
from potato_disease.models import efficientnet, mobilenet, resnet, simple_cnn  # noqa: E402,F401
