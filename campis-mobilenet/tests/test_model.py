from pathlib import Path
import pytest
torch=pytest.importorskip("torch")
from campis.datasets.mappings import load_crops
from campis.models import MultiCropMobileNet
def test_output_dimensions():
    crops=load_crops(Path(__file__).parents[1]/"configs/crops.yaml"); model=MultiCropMobileNet(crops,pretrained=False); x=torch.rand(2,3,224,224)
    assert {c:model(x,c).shape[1] for c in crops}=={"tomato":5,"maize":4,"coffee":2,"banana":3,"rice":4}
def test_mixed_crop_batch():
    crops=load_crops(Path(__file__).parents[1]/"configs/crops.yaml"); out=MultiCropMobileNet(crops,pretrained=False)(torch.rand(3,3,224,224),["tomato","coffee","tomato"]); assert out["tomato"][1].shape==(2,5) and out["coffee"][1].shape==(1,2)
