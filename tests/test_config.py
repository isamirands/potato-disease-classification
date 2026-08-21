from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

try:
    from ._bootstrap import PROJECT_ROOT
except ImportError:  # Permite ``unittest discover -s tests``.
    from _bootstrap import PROJECT_ROOT

from campis.config import AppConfig, load_config


_TEMP_CONFIG = """
[paths]
project_root = "project"
artifacts_dir = "artifacts"
manifest_path = "artifacts/manifest.csv"

[data]
classes = ["early_blight", "healthy", "late_blight"]
extensions = ["JPG", ".PNG"]
verify_images = false
balance_strategy = "source_class"
samples_per_source_class = 4
hash_duplicates = true
group_campis_by_minutes = 5
max_images_per_source_class = 0

[data.ratios]
train = 0.70
val = 0.15
test = 0.15

[[data.sources]]
name = "synthetic"
path = "source"
camera_type = "test-camera"

[model]
image_size = [96, 96]
dense_units = [16]
dropout = 0.0
weights = "none"
fine_tune_fraction = 0.0

[training]
batch_size = 2
head_epochs = 1
fine_tune_epochs = 0
head_lr = 0.001
fine_tune_lr = 0.00001
patience = 0
num_workers = 0

[augmentation]
flip = false
rotation = 0.0
translation = 0.0
zoom = 0.0
contrast = 0.0
"""


class ConfigTests(unittest.TestCase):
    def _temporary_project(self, root: Path) -> Path:
        project = root / "project"
        source = project / "source"
        for class_name in ("early_blight", "healthy", "late_blight"):
            (source / class_name).mkdir(parents=True)
        return project

    def _write_config(self, root: Path, contents: str = _TEMP_CONFIG) -> Path:
        path = root / "config.toml"
        path.write_text(textwrap.dedent(contents).strip() + "\n", encoding="utf-8")
        return path

    @unittest.skipUnless(
        (PROJECT_ROOT / "datasets" / "dataset_campis").is_dir()
        and (PROJECT_ROOT / "datasets" / "dataset_irish").is_dir(),
        "Los datasets reales no están disponibles en este checkout.",
    )
    def test_real_default_and_smoke_configs_load(self) -> None:
        expected = {
            "default.toml": ("imagenet", (224, 224)),
            "smoke.toml": (None, (96, 96)),
        }

        for filename, (weights, image_size) in expected.items():
            with self.subTest(config=filename):
                config = load_config(PROJECT_ROOT / "configs" / filename)
                self.assertIsInstance(config, AppConfig)
                self.assertEqual(config.model.weights, weights)
                self.assertEqual(config.model.image_size, image_size)
                self.assertEqual(
                    [source.name for source in config.data.sources],
                    ["campis", "irish"],
                )
                self.assertTrue(config.paths.project_root.is_absolute())

    def test_temporary_config_resolves_paths_and_normalizes_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = self._temporary_project(root)

            config = load_config(self._write_config(root))

            self.assertEqual(config.paths.project_root, project.resolve())
            self.assertEqual(config.paths.artifacts_dir, (project / "artifacts").resolve())
            self.assertEqual(config.data.extensions, (".jpg", ".png"))
            self.assertEqual(config.data.sources[0].path, (project / "source").resolve())
            self.assertIsNone(config.model.weights)

    def test_invalid_ratios_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._temporary_project(root)
            invalid = _TEMP_CONFIG.replace("test = 0.15", "test = 0.20")

            with self.assertRaisesRegex(ValueError, "sumar 1.0"):
                load_config(self._write_config(root, invalid))

    def test_unknown_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._temporary_project(root)
            invalid = _TEMP_CONFIG + "\nunexpected = true\n"

            with self.assertRaisesRegex(ValueError, "Claves desconocidas"):
                load_config(self._write_config(root, invalid))

    def test_fine_tuning_epochs_require_a_positive_fraction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._temporary_project(root)
            invalid = _TEMP_CONFIG.replace("fine_tune_epochs = 0", "fine_tune_epochs = 1")

            with self.assertRaisesRegex(ValueError, "fine_tune_epochs debe ser 0"):
                load_config(self._write_config(root, invalid))


if __name__ == "__main__":
    unittest.main()
