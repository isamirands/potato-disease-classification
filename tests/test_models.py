import pytest
import torch

from potato_disease.models import _MODEL_REGISTRY, get_model

NUM_CLASSES = 3


@pytest.mark.parametrize("model_name", sorted(_MODEL_REGISTRY.keys()))
def test_model_forward_pass(model_name):
    model = get_model(model_name, num_classes=NUM_CLASSES, pretrained=False, freeze_backbone=False)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, NUM_CLASSES)


def test_get_model_unknown_raises():
    with pytest.raises(ValueError):
        get_model("not_a_real_model")
