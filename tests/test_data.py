from __future__ import annotations

import shutil
import tempfile
import unittest
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # Permite ``unittest discover -s tests``.
    import _bootstrap  # type: ignore[no-redef]  # noqa: F401

import numpy as np
from PIL import Image

from campis.config import (
    AppConfig,
    AugmentationConfig,
    DataConfig,
    ModelConfig,
    PathsConfig,
    SourceConfig,
    TrainingConfig,
)
from campis.data import (
    ImageRecord,
    apply_exif_orientation,
    assign_splits,
    balance_records,
    build_balanced_tf_dataset,
    build_tf_dataset,
    center_crop_square,
    read_manifest_file,
    scan_dataset,
    validate_manifest,
    write_manifest,
)


CLASSES = ("early_blight", "healthy", "late_blight")
SOURCES = (
    ("campis", "professional"),
    ("irish", "cellphone"),
)


def _synthetic_config(root: Path) -> AppConfig:
    dataset_root = root / "synthetic_datasets"
    sources = tuple(
        SourceConfig(
            name=name,
            path=dataset_root / name,
            camera_type=camera_type,
        )
        for name, camera_type in SOURCES
    )
    return AppConfig(
        paths=PathsConfig(
            project_root=root,
            artifacts_dir=root / "artifacts",
            manifest_path=root / "artifacts" / "manifest.csv",
        ),
        data=DataConfig(
            classes=CLASSES,
            extensions=(".png",),
            seed=37,
            ratios={"train": 0.50, "val": 0.25, "test": 0.25},
            verify_images=True,
            balance_strategy="source_class",
            samples_per_source_class=4,
            sources=sources,
            hash_duplicates=True,
            group_campis_by_minutes=5,
            max_images_per_source_class=0,
        ),
        model=ModelConfig(
            image_size=(18, 20),
            dense_units=(8,),
            dropout=0.0,
            weights=None,
            fine_tune_fraction=0.0,
            tflite_optimize=False,
        ),
        training=TrainingConfig(
            batch_size=4,
            head_epochs=1,
            fine_tune_epochs=0,
            head_lr=1e-3,
            fine_tune_lr=1e-5,
            patience=0,
            num_workers=0,
        ),
        augmentation=AugmentationConfig(
            flip=False,
            rotation=0.0,
            translation=0.0,
            zoom=0.0,
            contrast=0.0,
        ),
    )


def _save_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 10), color=color).save(path, format="PNG")


def _populate_dataset(config: AppConfig) -> None:
    campis_times = (
        "090000",
        "090100",  # Misma ventana de captura de cinco minutos.
        "091000",
        "092000",
        "093000",
        "094000",
        "095000",
    )
    for source_index, source in enumerate(config.data.sources):
        for class_index, class_name in enumerate(config.data.classes):
            class_dir = source.path / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            created: list[Path] = []
            for image_index in range(7):
                if source.name == "campis":
                    filename = (
                        f"20260816_{campis_times[image_index]}_"
                        f"{class_index}_{image_index}.png"
                    )
                else:
                    filename = f"irish_{class_index}_{image_index}.png"
                base = source_index * 100 + class_index * 20 + image_index
                path = class_dir / filename
                _save_image(
                    path,
                    (
                        (base + 11) % 256,
                        (base * 3 + 17) % 256,
                        (base * 7 + 23) % 256,
                    ),
                )
                created.append(path)

            if source.name == "irish":
                # Un duplicado exacto y con la misma etiqueta debe permanecer unido.
                shutil.copyfile(created[5], created[6])

    corrupt = config.data.sources[0].path / CLASSES[0] / "corrupt.png"
    corrupt.write_bytes(b"this is not a PNG image")


class DataTests(unittest.TestCase):
    def test_exif_orientation_six_rotates_clockwise(self) -> None:
        import tensorflow as tf

        image = np.repeat(
            np.arange(2 * 3, dtype=np.float32).reshape(2, 3, 1), 3, axis=2
        )
        actual = apply_exif_orientation(tf.constant(image), 6).numpy()
        expected = np.rot90(image, k=3)
        np.testing.assert_array_equal(actual, expected)

    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary_directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary_directory.name)
        cls.config = _synthetic_config(cls.root)
        _populate_dataset(cls.config)
        cls.scan = scan_dataset(cls.config, progress_interval=0)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()

    def _assigned(self) -> list[ImageRecord]:
        return assign_splits(
            self.scan.records,
            self.config.data.ratios,
            seed=self.config.data.seed,
        )

    def test_scan_catalogs_two_sources_three_classes_and_invalid_image(self) -> None:
        self.assertEqual(len(self.scan.records), 42)
        self.assertEqual(len(self.scan.invalid_images), 1)
        self.assertTrue(self.scan.invalid_images[0].relative_path.endswith("corrupt.png"))
        self.assertEqual(
            Counter(record.source for record in self.scan.records),
            {"campis": 21, "irish": 21},
        )
        self.assertEqual(
            Counter(record.class_name for record in self.scan.records),
            {class_name: 14 for class_name in CLASSES},
        )
        self.assertTrue(all(record.width == 12 for record in self.scan.records))
        self.assertTrue(all(record.height == 10 for record in self.scan.records))
        self.assertTrue(all(len(record.sha256) == 64 for record in self.scan.records))
        self.assertTrue(all(record.group_id for record in self.scan.records))

    def test_split_is_deterministic_exhaustive_and_disjoint(self) -> None:
        first = self._assigned()
        second = self._assigned()
        first_mapping = {record.relative_path: record.split for record in first}
        second_mapping = {record.relative_path: record.split for record in second}

        self.assertEqual(first_mapping, second_mapping)
        self.assertEqual(
            set(first_mapping),
            {record.relative_path for record in self.scan.records},
        )
        paths_by_split = {
            split: {
                record.relative_path for record in first if record.split == split
            }
            for split in ("train", "val", "test")
        }
        self.assertTrue(all(paths_by_split.values()))
        self.assertTrue(paths_by_split["train"].isdisjoint(paths_by_split["val"]))
        self.assertTrue(paths_by_split["train"].isdisjoint(paths_by_split["test"]))
        self.assertTrue(paths_by_split["val"].isdisjoint(paths_by_split["test"]))
        self.assertEqual(set().union(*paths_by_split.values()), set(first_mapping))
        validate_manifest(first)

    def test_capture_groups_and_duplicate_hashes_never_cross_splits(self) -> None:
        assigned = self._assigned()
        splits_by_group: dict[str, set[str]] = defaultdict(set)
        paths_by_group: dict[str, set[str]] = defaultdict(set)
        splits_by_hash: dict[str, set[str]] = defaultdict(set)
        paths_by_hash: dict[str, set[str]] = defaultdict(set)
        for record in assigned:
            splits_by_group[record.group_id].add(record.split)
            paths_by_group[record.group_id].add(record.relative_path)
            splits_by_hash[record.sha256].add(record.split)
            paths_by_hash[record.sha256].add(record.relative_path)

        self.assertTrue(all(len(splits) == 1 for splits in splits_by_group.values()))
        self.assertTrue(all(len(splits) == 1 for splits in splits_by_hash.values()))
        self.assertTrue(
            any(
                group.startswith("capture:campis:") and len(paths) > 1
                for group, paths in paths_by_group.items()
            )
        )
        self.assertTrue(any(len(paths) > 1 for paths in paths_by_hash.values()))

    def test_identical_content_with_conflicting_labels_is_rejected(self) -> None:
        first = self.scan.records[0]
        other_class = next(
            record for record in self.scan.records if record.class_name != first.class_name
        )
        conflicting = replace(
            other_class,
            sha256=first.sha256,
            group_id="independent-conflicting-group",
        )

        with self.assertRaisesRegex(ValueError, "etiquetas distintas"):
            assign_splits(
                [first, conflicting],
                self.config.data.ratios,
                seed=self.config.data.seed,
            )

    def test_manifest_csv_round_trip_preserves_records(self) -> None:
        assigned = self._assigned()
        manifest_path = self.root / "roundtrip" / "manifest.csv"

        returned = write_manifest(assigned, manifest_path)
        restored = read_manifest_file(manifest_path)

        self.assertEqual(returned, manifest_path.resolve())
        self.assertEqual(restored, assigned)

    def test_virtual_balance_is_deterministic_without_creating_images(self) -> None:
        training_records = [record for record in self._assigned() if record.split == "train"]
        before = {
            path.relative_to(self.root)
            for path in self.root.rglob("*.png")
        }

        first = balance_records(
            training_records,
            strategy="source_class",
            target_per_group=4,
            seed=91,
        )
        second = balance_records(
            training_records,
            strategy="source_class",
            target_per_group=4,
            seed=91,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), len(SOURCES) * len(CLASSES) * 4)
        self.assertEqual(
            Counter((record.source, record.class_name) for record in first),
            {
                (source, class_name): 4
                for source, _camera in SOURCES
                for class_name in CLASSES
            },
        )
        self.assertLessEqual(
            {record.relative_path for record in first},
            {record.relative_path for record in training_records},
        )
        after = {path.relative_to(self.root) for path in self.root.rglob("*.png")}
        self.assertEqual(after, before)

    def test_tf_dataset_has_expected_shape_dtype_order_and_range(self) -> None:
        import tensorflow as tf

        records = list(self.scan.records[:4])
        dataset = build_tf_dataset(
            records,
            self.config.paths.project_root,
            image_size=(18, 20),
            batch_size=4,
            training=False,
            seed=self.config.data.seed,
        )

        images, labels = next(iter(dataset))

        self.assertEqual(tuple(images.shape), (4, 18, 20, 3))
        self.assertEqual(tuple(labels.shape), (4,))
        self.assertEqual(images.dtype, tf.float32)
        self.assertEqual(labels.dtype, tf.int32)
        self.assertGreaterEqual(float(tf.reduce_min(images).numpy()), 0.0)
        self.assertLessEqual(float(tf.reduce_max(images).numpy()), 255.0)
        np.testing.assert_array_equal(
            labels.numpy(),
            np.asarray([record.class_index for record in records], dtype=np.int32),
        )


class CaptureGroupingTests(unittest.TestCase):
    """Cobertura de las heuristicas de sesion que evitan fugas entre splits."""

    def _config(self, root: Path, gap: int) -> AppConfig:
        config = _synthetic_config(root)
        return replace(config, data=replace(config.data, group_sequence_gap=gap))

    def _populate(self, config: AppConfig, names: dict[str, list[str]]) -> None:
        for source_index, source in enumerate(config.data.sources):
            for class_index, class_name in enumerate(config.data.classes):
                for index, stem in enumerate(names[class_name]):
                    # Colores unicos: sin duplicados exactos, el fallback de
                    # agrupacion por SHA-256 no puede unir archivos distintos.
                    _save_image(
                        source.path / class_name / f"{stem}.png",
                        (index * 7 + 1, class_index * 40 + 3, source_index * 90 + 5),
                    )

    def _campis_groups(self, config: AppConfig) -> dict[str, str]:
        return {
            record.relative_path.rsplit("/", 1)[-1]: record.group_id
            for record in scan_dataset(config, progress_interval=0).records
            if record.source == "campis"
        }

    def test_camera_counter_runs_share_one_group_across_classes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory), gap=8)
            # IMG_0051/0052/0054 son una rafaga; IMG_0400 abre otra sesion.
            self._populate(
                config,
                {
                    "early_blight": ["IMG_0051", "IMG_0052"],
                    "healthy": ["IMG_0054", "IMG_0400"],
                    "late_blight": ["IMG_0401", "IMG_0900"],
                },
            )
            groups = self._campis_groups(config)

            self.assertEqual(groups["IMG_0051.png"], groups["IMG_0052.png"])
            # La rafaga cruza la frontera de clase: el contador es de la camara.
            self.assertEqual(groups["IMG_0051.png"], groups["IMG_0054.png"])
            self.assertNotEqual(groups["IMG_0051.png"], groups["IMG_0400.png"])
            self.assertEqual(groups["IMG_0400.png"], groups["IMG_0401.png"])
            self.assertNotEqual(groups["IMG_0400.png"], groups["IMG_0900.png"])

    def test_counter_grouping_is_disabled_with_gap_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory), gap=0)
            self._populate(
                config,
                {
                    "early_blight": ["IMG_0051", "IMG_0052"],
                    "healthy": ["IMG_0054", "IMG_0400"],
                    "late_blight": ["IMG_0401", "IMG_0900"],
                },
            )
            groups = self._campis_groups(config)
            self.assertEqual(len(set(groups.values())), len(groups))

    def test_epoch_millisecond_names_share_a_capture_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory), gap=8)
            self._populate(
                config,
                {
                    "early_blight": ["1692336758850", "1692336758935"],
                    "healthy": ["1692340000000", "IMG_0400"],
                    "late_blight": ["IMG_0401", "IMG_0900"],
                },
            )
            groups = self._campis_groups(config)

            self.assertEqual(
                groups["1692336758850.png"], groups["1692336758935.png"]
            )
            self.assertTrue(groups["1692336758850.png"].startswith("capture:campis:"))
            self.assertNotEqual(
                groups["1692336758850.png"], groups["1692340000000.png"]
            )


class PreprocessingTests(unittest.TestCase):
    def test_center_crop_square_keeps_the_middle_of_a_wide_image(self) -> None:
        import tensorflow as tf

        image = np.repeat(
            np.arange(4 * 10, dtype=np.float32).reshape(4, 10, 1), 3, axis=2
        )
        cropped = center_crop_square(tf.constant(image)).numpy()

        self.assertEqual(cropped.shape, (4, 4, 3))
        np.testing.assert_array_equal(cropped, image[:, 3:7, :])

    def test_center_crop_leaves_a_square_image_untouched(self) -> None:
        import tensorflow as tf

        image = np.repeat(
            np.arange(5 * 5, dtype=np.float32).reshape(5, 5, 1), 3, axis=2
        )
        np.testing.assert_array_equal(
            center_crop_square(tf.constant(image)).numpy(), image
        )


class BalancedDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary_directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary_directory.name)
        cls.config = _synthetic_config(cls.root)
        _populate_dataset(cls.config)
        cls.records = list(scan_dataset(cls.config, progress_interval=0).records)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()

    def _unbalanced(self) -> list[ImageRecord]:
        """Deja una fuente/clase minoritaria frente al resto."""

        counts: Counter = Counter()
        kept: list[ImageRecord] = []
        for record in self.records:
            key = (record.source, record.class_name)
            limit = 2 if key == ("campis", "healthy") else 7
            if counts[key] < limit:
                counts[key] += 1
                kept.append(record)
        return kept

    def _build(self, records: list[ImageRecord], target: int) -> tuple[Any, int]:
        return build_balanced_tf_dataset(
            records,
            self.config.paths.project_root,
            image_size=(18, 20),
            batch_size=4,
            strategy="source_class",
            target_per_group=target,
            seed=self.config.data.seed,
        )

    def test_each_epoch_is_balanced_across_sources_and_classes(self) -> None:
        dataset, steps = self._build(self._unbalanced(), target=4)

        # Seis grupos fuente/clase x 4 referencias = 24 imagenes por epoca.
        self.assertEqual(steps, 6)
        for _ in range(2):
            labels = [
                int(value)
                for _, batch in dataset.as_numpy_iterator()
                for value in batch
            ]
            self.assertEqual(len(labels), 24)
            # Cada clase aporta 8: dos fuentes x 4 referencias, pese a que
            # campis/healthy solo tiene 2 imagenes reales disponibles.
            self.assertEqual(Counter(labels), {0: 8, 1: 8, 2: 8})

    def test_consecutive_epochs_draw_different_references(self) -> None:
        dataset, _ = self._build(self._unbalanced(), target=4)

        def epoch_signature() -> list[float]:
            return sorted(
                float(image.sum())
                for images, _ in dataset.as_numpy_iterator()
                for image in images
            )

        # Con 7 referencias disponibles y 4 por epoca, la fuente mayoritaria
        # tiene que mostrar imagenes distintas en la vuelta siguiente.
        self.assertNotEqual(epoch_signature(), epoch_signature())

    def test_balanced_dataset_rejects_an_invalid_target(self) -> None:
        with self.assertRaises(ValueError):
            self._build(list(self.records), target=0)



class TruncatedImageTests(unittest.TestCase):
    """Un JPEG cortado debe morir en la auditoria, no dentro de tf.data."""

    def test_scan_reports_a_truncated_jpeg_as_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _synthetic_config(root)
            config = replace(
                config, data=replace(config.data, extensions=(".jpg",))
            )

            for source in config.data.sources:
                for class_index, class_name in enumerate(config.data.classes):
                    class_dir = source.path / class_name
                    class_dir.mkdir(parents=True, exist_ok=True)
                    for index in range(2):
                        path = class_dir / f"leaf_{class_index}_{index}.jpg"
                        Image.new(
                            "RGB", (64, 48), color=(index * 40 + 10, class_index * 50 + 5, 90)
                        ).save(path, format="JPEG", quality=95)

            healthy = config.data.sources[0].path / "healthy"
            victim = healthy / "leaf_1_0.jpg"
            intact = victim.read_bytes()
            # Cortar solo la cola del scan comprimido: las cabeceras y los
            # marcadores siguen intactos, asi que verify() lo aprueba. Es el
            # mismo dano que trae un archivo copiado a medias.
            victim.write_bytes(intact[: int(len(intact) * 0.97)])

            with Image.open(image_or_path := victim) as image:
                image.verify()  # no levanta nada: esa es justamente la trampa
            with self.assertRaises(OSError):
                with Image.open(image_or_path) as image:
                    image.load()

            scan = scan_dataset(config, progress_interval=0)

            # El mismo nombre existe en la otra fuente y esta sano: hay que
            # comparar rutas completas, no nombres de archivo.
            victim_relative = victim.resolve().relative_to(root.resolve()).as_posix()
            self.assertEqual(
                [item.relative_path for item in scan.invalid_images],
                [victim_relative],
            )
            self.assertNotIn(
                victim_relative,
                [record.relative_path for record in scan.records],
            )
            # Su gemelo intacto en la otra fuente sí entra al catálogo.
            self.assertEqual(
                sum(
                    1
                    for record in scan.records
                    if record.relative_path.endswith("healthy/leaf_1_0.jpg")
                ),
                1,
            )



if __name__ == "__main__":
    unittest.main()
