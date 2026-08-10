from pathlib import Path

import pytest

from potato_disease.config import Config

CONFIG_DIR = Path(__file__).parent.parent / "configs"
CONFIG_PATHS = sorted(CONFIG_DIR.glob("*.yaml"))


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=lambda p: p.stem)
def test_config_loads(config_path):
    cfg = Config.from_yaml(config_path)

    assert cfg.model.name
    assert cfg.model.num_classes > 0
    assert cfg.data.image_size > 0
    assert cfg.train.epochs > 0


def test_found_all_configs():
    names = {p.stem for p in CONFIG_PATHS}
    assert names == {"simple_cnn", "resnet18", "mobilenet_v2", "efficientnet_b0"}
