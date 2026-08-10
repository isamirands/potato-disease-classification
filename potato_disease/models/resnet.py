"""ResNet18 transfer-learning wrapper."""

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from potato_disease.models import register_model


@register_model("resnet18")
def build_resnet18(
    num_classes: int = 3, pretrained: bool = True, freeze_backbone: bool = True, **_kwargs
) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
