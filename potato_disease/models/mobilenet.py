"""MobileNetV2 transfer-learning wrapper."""

import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from potato_disease.models import register_model


@register_model("mobilenet_v2")
def build_mobilenet_v2(
    num_classes: int = 3, pretrained: bool = True, freeze_backbone: bool = True, **_kwargs
) -> nn.Module:
    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = mobilenet_v2(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    last_in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(last_in_features, num_classes)
    return model
