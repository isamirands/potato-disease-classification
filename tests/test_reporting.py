from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # Permite ``unittest discover -s tests``.
    import _bootstrap  # type: ignore[no-redef]  # noqa: F401

from PIL import Image

from campis.reporting import (
    build_report,
    plot_class_metrics,
    plot_confidence_distribution,
    plot_dataset_distribution,
    plot_sample_predictions,
    plot_top_errors,
    plot_training_curves,
)


CLASSES = ("early_blight", "healthy", "late_blight")


def _write_csv(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _history_rows() -> list[dict[str, object]]:
    """Dos fases, con las metricas por dominio solo en la segunda."""

    rows = []
    for epoch in (1, 2):
        rows.append(
            {
                "phase": "head",
                "epoch": epoch,
                "loss": 1.1 - 0.1 * epoch,
                "accuracy": 0.4 + 0.1 * epoch,
                "val_loss": 1.0 - 0.05 * epoch,
                "val_accuracy": 0.45 + 0.08 * epoch,
                "val_campis_loss": "",
                "val_campis_accuracy": "",
                "val_irish_loss": "",
                "val_irish_accuracy": "",
            }
        )
    for epoch in (1, 2):
        rows.append(
            {
                "phase": "fine_tune",
                "epoch": epoch,
                "loss": 0.7 - 0.05 * epoch,
                "accuracy": 0.7 + 0.05 * epoch,
                "val_loss": 0.8 - 0.05 * epoch,
                "val_accuracy": 0.7 + 0.04 * epoch,
                "val_campis_loss": 0.95 - 0.05 * epoch,
                "val_campis_accuracy": 0.6 + 0.03 * epoch,
                "val_irish_loss": 0.7 - 0.04 * epoch,
                "val_irish_accuracy": 0.8 + 0.02 * epoch,
            }
        )
    return rows


def _class_report() -> dict[str, object]:
    report = {
        name: {
            "precision": 0.8,
            "recall": 0.75,
            "f1-score": 0.77,
            "support": 40,
        }
        for name in CLASSES
    }
    report["accuracy"] = 0.78
    report["macro avg"] = {
        "precision": 0.8, "recall": 0.75, "f1-score": 0.77, "support": 120
    }
    report["weighted avg"] = {
        "precision": 0.8, "recall": 0.75, "f1-score": 0.77, "support": 120
    }
    return report


def _metrics() -> dict[str, object]:
    section = {
        "samples": 120,
        "accuracy": 0.78,
        "macro_f1": 0.77,
        "weighted_f1": 0.77,
        "classification_report": _class_report(),
        "confusion_matrix": [[30, 5, 5], [4, 32, 4], [6, 4, 30]],
    }
    return {
        "global": dict(section),
        "by_source": {"campis": dict(section), "irish": dict(section)},
    }


def _prediction_rows(root: Path) -> list[dict[str, object]]:
    rows = []
    for index in range(8):
        source = "campis" if index % 2 == 0 else "irish"
        relative = f"datasets/{source}/{CLASSES[index % 3]}/img{index}.png"
        image_path = root / relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (24, 18), color=(index * 20 % 256, 120, 90)).save(image_path)
        correct = index % 3 != 0
        rows.append(
            {
                "relative_path": relative,
                "source": source,
                "camera_type": "professional" if source == "campis" else "cellphone",
                "true_class": CLASSES[index % 3],
                "predicted_class": CLASSES[index % 3] if correct else CLASSES[(index + 1) % 3],
                "confidence": f"{0.55 + index * 0.05:.4f}",
                "correct": correct,
            }
        )
    return rows


PREDICTION_FIELDS = (
    "relative_path", "source", "camera_type", "true_class",
    "predicted_class", "confidence", "correct",
)


class ReportingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()

        history_rows = _history_rows()
        _write_csv(self.run_dir / "history.csv", list(history_rows[0]), history_rows)

        evaluation = self.run_dir / "evaluation"
        evaluation.mkdir()
        (evaluation / "metrics.json").write_text(
            json.dumps(_metrics()), encoding="utf-8"
        )
        self.predictions = _prediction_rows(self.root)
        _write_csv(evaluation / "predictions.csv", PREDICTION_FIELDS, self.predictions)

        _write_csv(
            self.run_dir / "manifest.csv",
            ("relative_path", "source", "class_name", "split"),
            [
                {
                    "relative_path": row["relative_path"],
                    "source": row["source"],
                    "class_name": row["true_class"],
                    "split": ("train", "val", "test")[index % 3],
                }
                for index, row in enumerate(self.predictions)
            ],
        )

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _assert_png(self, path) -> None:
        self.assertIsNotNone(path)
        path = Path(path)
        self.assertTrue(path.is_file(), f"No se creo {path}")
        self.assertGreater(path.stat().st_size, 1000)
        with Image.open(path) as image:
            self.assertEqual(image.format, "PNG")

    def test_training_curves_handle_two_phases_and_missing_domain_columns(self) -> None:
        output = self.run_dir / "curves.png"
        self._assert_png(plot_training_curves(self.run_dir / "history.csv", output))

    def test_training_curves_reject_an_empty_history(self) -> None:
        empty = self.run_dir / "empty.csv"
        _write_csv(empty, ("phase", "epoch"), [])
        with self.assertRaises(ValueError):
            plot_training_curves(empty, self.run_dir / "nope.png")

    def test_class_metrics_draw_one_panel_per_scope(self) -> None:
        output = self.run_dir / "classes.png"
        self._assert_png(
            plot_class_metrics(self.run_dir / "evaluation" / "metrics.json", output)
        )

    def test_confidence_distribution_separates_hits_from_errors(self) -> None:
        output = self.run_dir / "confidence.png"
        self._assert_png(
            plot_confidence_distribution(
                self.run_dir / "evaluation" / "predictions.csv", output
            )
        )

    def test_sample_grid_reads_the_referenced_images(self) -> None:
        output = self.run_dir / "samples.png"
        self._assert_png(
            plot_sample_predictions(
                self.run_dir / "evaluation" / "predictions.csv",
                self.root,
                (16, 16),
                output,
                per_source=2,
            )
        )

    def test_top_errors_only_uses_wrong_predictions(self) -> None:
        output = self.run_dir / "errors.png"
        self._assert_png(
            plot_top_errors(
                self.run_dir / "evaluation" / "predictions.csv",
                self.root,
                (16, 16),
                output,
                limit=3,
            )
        )

    def test_top_errors_returns_none_when_everything_is_correct(self) -> None:
        perfect = self.run_dir / "perfect.csv"
        rows = [dict(row, correct=True, predicted_class=row["true_class"]) for row in self.predictions]
        _write_csv(perfect, PREDICTION_FIELDS, rows)

        self.assertIsNone(
            plot_top_errors(perfect, self.root, (16, 16), self.run_dir / "none.png")
        )

    def test_dataset_distribution_stacks_the_three_splits(self) -> None:
        output = self.run_dir / "corpus.png"
        self._assert_png(
            plot_dataset_distribution(self.run_dir / "manifest.csv", output)
        )

    def test_build_report_creates_every_figure(self) -> None:
        created = build_report(self.run_dir, self.root, (16, 16), seed=7)

        names = {Path(path).name for path in created}
        self.assertEqual(
            names,
            {
                "training_curves.png",
                "class_metrics.png",
                "confidence_distribution.png",
                "sample_predictions.png",
                "top_errors.png",
                "dataset_distribution.png",
            },
        )
        for path in created:
            self._assert_png(path)

    def test_build_report_skips_figures_whose_input_is_missing(self) -> None:
        (self.run_dir / "history.csv").unlink()
        (self.run_dir / "manifest.csv").unlink()

        created = build_report(self.run_dir, self.root, (16, 16), seed=7)

        names = {Path(path).name for path in created}
        self.assertNotIn("training_curves.png", names)
        self.assertNotIn("dataset_distribution.png", names)
        # Lo que sí tiene sus artefactos se sigue dibujando.
        self.assertIn("class_metrics.png", names)
        self.assertIn("sample_predictions.png", names)

    def test_build_report_tolerates_an_unreadable_image(self) -> None:
        broken = self.root / self.predictions[0]["relative_path"]
        broken.write_bytes(b"esto no es un PNG")

        created = build_report(self.run_dir, self.root, (16, 16), seed=7)

        self.assertIn(
            "sample_predictions.png", {Path(path).name for path in created}
        )


if __name__ == "__main__":
    unittest.main()
