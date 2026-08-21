from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # Permite ``unittest discover -s tests``.
    import _bootstrap  # type: ignore[no-redef]  # noqa: F401

from PIL import Image

from campis.audit import (
    verify_tensorflow_decoding,
)  # noqa: F401
from campis.audit import audit_dataset, format_summary, summarize_scan
from campis.config import (
    AppConfig,
    AugmentationConfig,
    DataConfig,
    ModelConfig,
    PathsConfig,
    SourceConfig,
    TrainingConfig,
)
from campis.data import ImageRecord, ScanResult


CLASSES = ("early_blight", "healthy", "late_blight")


def _audit_config(root: Path) -> AppConfig:
    source_root = root / "sources"
    return AppConfig(
        paths=PathsConfig(
            project_root=root,
            artifacts_dir=root / "artifacts",
            manifest_path=root / "artifacts" / "manifest.csv",
        ),
        data=DataConfig(
            classes=CLASSES,
            extensions=(".png",),
            seed=7,
            ratios={"train": 0.6, "val": 0.2, "test": 0.2},
            verify_images=True,
            balance_strategy="none",
            samples_per_source_class=2,
            sources=(
                SourceConfig("campis", source_root / "campis", "professional"),
                SourceConfig("irish", source_root / "irish", "cellphone"),
            ),
            hash_duplicates=True,
            group_campis_by_minutes=0,
            max_images_per_source_class=0,
        ),
        model=ModelConfig((32, 32), (8,), 0.0, None, 0.0, False),
        training=TrainingConfig(2, 1, 0, 1e-3, 1e-5, 0, 0),
        augmentation=AugmentationConfig(False, 0.0, 0.0, 0.0, 0.0),
    )


def _populate_audit_dataset(config: AppConfig) -> None:
    paths: dict[tuple[str, str, int], Path] = {}
    for source_index, source in enumerate(config.data.sources):
        for class_index, class_name in enumerate(config.data.classes):
            for image_index in range(2):
                path = source.path / class_name / f"image_{image_index}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                base = source_index * 100 + class_index * 20 + image_index
                Image.new(
                    "RGB",
                    (12, 10),
                    color=((base + 3) % 256, (base * 5 + 7) % 256, (base * 9 + 11) % 256),
                ).save(path, format="PNG")
                paths[(source.name, class_name, image_index)] = path

    # Duplicado exacto entre fuentes, conservando la etiqueta healthy.
    shutil.copyfile(
        paths[("campis", "healthy", 0)],
        paths[("irish", "healthy", 1)],
    )
    corrupt = config.data.sources[0].path / "early_blight" / "invalid.png"
    corrupt.write_bytes(b"invalid image payload")


class AuditTests(unittest.TestCase):
    def test_audit_reports_distribution_duplicates_invalids_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _audit_config(root)
            _populate_audit_dataset(config)
            output_path = root / "reports" / "audit.json"

            summary, scan = audit_dataset(
                config,
                verify_images=True,
                hash_duplicates=True,
                output_path=output_path,
            )

            self.assertEqual(len(scan.records), 12)
            self.assertEqual(len(scan.invalid_images), 1)
            self.assertEqual(summary["total_valid_images"], 12)
            self.assertEqual(summary["by_source"], {"campis": 6, "irish": 6})
            self.assertEqual(summary["by_class"], {name: 4 for name in CLASSES})
            self.assertEqual(summary["by_camera_type"], {"cellphone": 6, "professional": 6})
            self.assertEqual(summary["source_percentages"], {"campis": 50.0, "irish": 50.0})
            self.assertEqual(
                summary["most_common_dimensions"][0],
                {"width": 12, "height": 10, "count": 12},
            )
            self.assertEqual(len(summary["invalid_images"]), 1)
            self.assertEqual(summary["exact_duplicate_group_count"], 1)
            self.assertEqual(summary["exact_duplicate_file_count"], 2)
            self.assertEqual(summary["conflicting_label_hashes"], {})
            self.assertTrue(summary["hashes_calculated"])

            persisted = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted, summary)
            formatted = format_summary(summary)
            self.assertIn("campis: 6 (50.00 %)", formatted)
            self.assertIn("irish: 6 (50.00 %)", formatted)
            self.assertIn("Grupos duplicados exactos: 1", formatted)

    def test_summary_surfaces_identical_hash_with_conflicting_labels(self) -> None:
        records = (
            ImageRecord(
                relative_path="a.png",
                source="campis",
                camera_type="professional",
                class_name="early_blight",
                class_index=0,
                sha256="same-digest",
                group_id="group-a",
                width=12,
                height=10,
                size_bytes=10,
            ),
            ImageRecord(
                relative_path="b.png",
                source="irish",
                camera_type="cellphone",
                class_name="late_blight",
                class_index=2,
                sha256="same-digest",
                group_id="group-b",
                width=12,
                height=10,
                size_bytes=10,
            ),
        )

        summary = summarize_scan(ScanResult(records=records, invalid_images=()))

        self.assertEqual(
            summary["conflicting_label_hashes"],
            {"same-digest": ["early_blight", "late_blight"]},
        )
        self.assertEqual(summary["exact_duplicate_group_count"], 1)
        self.assertIn("ERROR", format_summary(summary))

    def test_fast_audit_summary_marks_hashes_as_not_calculated(self) -> None:
        record = ImageRecord(
            relative_path="leaf.png",
            source="campis",
            camera_type="professional",
            class_name="healthy",
            class_index=1,
            group_id="path:leaf.png",
            width=12,
            height=10,
            size_bytes=10,
        )

        summary = summarize_scan(ScanResult(records=(record,), invalid_images=()))

        self.assertFalse(summary["hashes_calculated"])
        self.assertIn("no calculados", format_summary(summary))


class TensorFlowDecodingTests(unittest.TestCase):
    """El decodificador de TensorFlow es el arbitro final: es el que entrena."""

    def _config(self, root: Path):
        from campis.config import (
            AppConfig,
            AugmentationConfig,
            DataConfig,
            ModelConfig,
            PathsConfig,
            SourceConfig,
            TrainingConfig,
        )

        classes = ("early_blight", "healthy", "late_blight")
        source = SourceConfig(
            name="campis", path=root / "datasets" / "campis", camera_type="professional"
        )
        for class_name in classes:
            (source.path / class_name).mkdir(parents=True, exist_ok=True)
        return AppConfig(
            paths=PathsConfig(
                project_root=root,
                artifacts_dir=root / "artifacts",
                manifest_path=root / "artifacts" / "manifest.csv",
            ),
            data=DataConfig(
                classes=classes,
                extensions=(".jpg",),
                seed=11,
                ratios={"train": 0.5, "val": 0.25, "test": 0.25},
                verify_images=False,
                balance_strategy="none",
                samples_per_source_class=2,
                sources=(source,),
                hash_duplicates=False,
            ),
            model=ModelConfig(
                image_size=(32, 32),
                dense_units=(8,),
                dropout=0.0,
                weights=None,
                fine_tune_fraction=0.0,
                tflite_optimize=False,
            ),
            training=TrainingConfig(
                batch_size=2, head_epochs=1, fine_tune_epochs=0,
                head_lr=1e-3, fine_tune_lr=1e-5, patience=0, num_workers=2,
            ),
            augmentation=AugmentationConfig(
                flip=False, rotation=0.0, translation=0.0, zoom=0.0, contrast=0.0
            ),
        )

    def test_truncated_jpeg_is_caught_even_when_pillow_accepts_it(self) -> None:
        from PIL import Image

        from campis.audit import audit_dataset
        from campis.data import scan_dataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            victim_name = "leaf_0.jpg"
            for class_name in config.data.classes:
                for index in range(2):
                    path = config.data.sources[0].path / class_name / f"leaf_{index}.jpg"
                    Image.new(
                        "RGB", (64, 48), color=(index * 60 + 20, 110, 70)
                    ).save(path, format="JPEG", quality=95)

            victim = config.data.sources[0].path / "healthy" / victim_name
            intact = victim.read_bytes()
            victim.write_bytes(intact[: int(len(intact) * 0.97)])

            # Sin verificacion de Pillow, el archivo entra al catalogo.
            scan = scan_dataset(config, verify_images=False, progress_interval=0)
            self.assertEqual(scan.invalid_images, ())

            summary, _ = audit_dataset(config, verify_images=False, decode_with_tensorflow=True)

            self.assertTrue(summary["tensorflow_decode_checked"])
            failures = summary["tensorflow_decode_failures"]
            self.assertEqual(len(failures), 1)
            self.assertTrue(failures[0]["relative_path"].endswith("healthy/leaf_0.jpg"))
            self.assertEqual(failures[0]["source"], "campis")

    def test_healthy_corpus_reports_no_decode_failures(self) -> None:
        from PIL import Image

        from campis.audit import audit_dataset, format_summary

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            for class_name in config.data.classes:
                for index in range(2):
                    path = config.data.sources[0].path / class_name / f"leaf_{index}.jpg"
                    Image.new("RGB", (48, 32), color=(index * 50, 90, 140)).save(
                        path, format="JPEG"
                    )

            summary, _ = audit_dataset(config, decode_with_tensorflow=True)

            self.assertEqual(summary["tensorflow_decode_failures"], [])
            self.assertIn("Ilegibles para TensorFlow: 0", format_summary(summary))

    def test_summary_omits_the_line_when_the_check_did_not_run(self) -> None:
        from PIL import Image

        from campis.audit import audit_dataset, format_summary

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            for class_name in config.data.classes:
                for index in range(2):
                    path = config.data.sources[0].path / class_name / f"leaf_{index}.jpg"
                    Image.new("RGB", (48, 32), color=(index * 50, 90, 140)).save(
                        path, format="JPEG"
                    )

            summary, _ = audit_dataset(config)

            self.assertFalse(summary["tensorflow_decode_checked"])
            self.assertNotIn("Ilegibles para TensorFlow", format_summary(summary))



if __name__ == "__main__":
    unittest.main()
