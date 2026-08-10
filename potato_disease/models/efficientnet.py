"""EfficientNet-B0 transfer-learning wrapper."""

import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from potato_disease.models import register_model


@register_model("efficientnet_b0")
def build_efficientnet_b0(
    num_classes: int = 3, pretrained: bool = True, freeze_backbone: bool = True, **_kwargs
) -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    last_in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(last_in_features, num_classes)
    return model
