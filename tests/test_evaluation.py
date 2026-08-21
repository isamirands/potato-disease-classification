from __future__ import annotations

import csv
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # Permite ``unittest discover -s tests``.
    import _bootstrap  # type: ignore[no-redef]  # noqa: F401

import numpy as np

from campis.evaluation import compute_metrics, evaluate_model, save_evaluation_artifacts


CLASS_NAMES = ("early_blight", "healthy", "late_blight")
PROBABILITIES = np.asarray(
    [
        [0.90, 0.05, 0.05],
        [0.05, 0.10, 0.85],
        [0.05, 0.10, 0.85],
        [0.80, 0.10, 0.10],
        [0.10, 0.80, 0.10],
        [0.10, 0.80, 0.10],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class _Record:
    relative_path: str
    source: str
    camera_type: str
    class_index: int


def _records() -> list[_Record]:
    return [
        _Record("campis/a.jpg", "campis", "professional", 0),
        _Record("campis/b.jpg", "campis", "professional", 1),
        _Record("campis/c.jpg", "campis", "professional", 2),
        _Record("irish/a.jpg", "irish", "cellphone", 0),
        _Record("irish/b.jpg", "irish", "cellphone", 1),
        _Record("irish/c.jpg", "irish", "cellphone", 2),
    ]


class EvaluationTests(unittest.TestCase):
    def test_global_and_per_source_metrics(self) -> None:
        records = _records()

        metrics = compute_metrics(
            [record.class_index for record in records],
            PROBABILITIES,
            [record.source for record in records],
            CLASS_NAMES,
        )

        self.assertEqual(metrics["global"]["samples"], 6)
        self.assertAlmostEqual(metrics["global"]["accuracy"], 4 / 6)
        np.testing.assert_array_equal(
            metrics["global"]["confusion_matrix"],
            np.asarray([[2, 0, 0], [0, 1, 1], [0, 1, 1]]),
        )
        self.assertEqual(list(metrics["by_source"]), ["campis", "irish"])
        self.assertEqual(metrics["by_source"]["campis"]["samples"], 3)
        self.assertEqual(metrics["by_source"]["irish"]["samples"], 3)
        self.assertAlmostEqual(metrics["by_source"]["campis"]["accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["by_source"]["irish"]["accuracy"], 2 / 3)

    def test_evaluate_model_aligns_predictions_with_records(self) -> None:
        model = MagicMock()
        model.predict.return_value = PROBABILITIES
        records = _records()

        metrics, returned_probabilities = evaluate_model(
            model, "ordered-dataset", records, CLASS_NAMES, verbose=0
        )

        model.predict.assert_called_once_with("ordered-dataset", verbose=0)
        np.testing.assert_array_equal(returned_probabilities, PROBABILITIES)
        self.assertEqual(metrics["global"]["samples"], len(records))

    def test_evaluation_artifacts_are_complete_and_serializable(self) -> None:
        records = _records()
        metrics = compute_metrics(
            [record.class_index for record in records],
            PROBABILITIES,
            [record.source for record in records],
            CLASS_NAMES,
        )

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "evaluation"

            save_evaluation_artifacts(
                metrics, PROBABILITIES, records, CLASS_NAMES, output_dir
            )

            metrics_json = json.loads(
                (output_dir / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics_json["global"]["samples"], 6)
            self.assertEqual(
                metrics_json["global"]["confusion_matrix"],
                [[2, 0, 0], [0, 1, 1], [0, 1, 1]],
            )

            with (output_dir / "predictions.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 6)
            self.assertEqual(rows[0]["predicted_class"], "early_blight")
            self.assertEqual(rows[1]["correct"], "False")

            for filename in (
                "confusion_global.png",
                "confusion_source_01_campis.png",
                "confusion_source_02_irish.png",
            ):
                artifact = output_dir / filename
                self.assertTrue(artifact.is_file(), filename)
                self.assertGreater(artifact.stat().st_size, 0, filename)


if __name__ == "__main__":
    unittest.main()
