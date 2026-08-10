from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from potato_disease.config import Config
from potato_disease.data.dataset import get_dataloaders

CLASSES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]


def _make_fake_split(root: Path, split: str, images_per_class: int = 4) -> None:
    for class_name in CLASSES:
        class_dir = root / split / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(images_per_class):
            arr = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
            Image.fromarray(arr).save(class_dir / f"img_{i}.jpg")


@pytest.fixture
def fake_processed_dir(tmp_path):
    for split in ("train", "val", "test"):
        _make_fake_split(tmp_path, split)
    return tmp_path


def test_get_dataloaders(fake_processed_dir):
    cfg = Config.from_yaml(Path(__file__).parent.parent / "configs" / "simple_cnn.yaml")
    cfg.data.data_dir = str(fake_processed_dir)
    cfg.data.batch_size = 2
    cfg.data.num_workers = 0

    train_loader, val_loader, test_loader, classes = get_dataloaders(cfg)

    assert set(classes) == set(CLASSES)

    images, labels = next(iter(train_loader))
    assert images.shape[0] == 2
    assert images.shape[1] == 3
    assert labels.shape[0] == 2
