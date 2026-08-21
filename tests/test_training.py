from __future__ import annotations

import csv
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # Permite ``unittest discover -s tests``.
    import _bootstrap  # type: ignore[no-redef]  # noqa: F401

from campis.training import (
    build_domain_validation_callback,
    _save_combined_history,
    compute_class_weights,
    train_two_phases,
)


@dataclass(frozen=True)
class _Record:
    class_index: int


class _SavedModel:
    def __init__(self) -> None:
        self.saved_paths: list[Path] = []

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.write_text("synthetic model", encoding="utf-8")
        self.saved_paths.append(destination)


class TrainingTests(unittest.TestCase):
    def test_class_weights_preserve_non_contiguous_real_indices(self) -> None:
        records = [
            _Record(2),
            _Record(2),
            _Record(2),
            _Record(7),
            _Record(7),
            _Record(11),
        ]

        weights = compute_class_weights(records)

        self.assertEqual(set(weights), {2, 7, 11})
        self.assertAlmostEqual(weights[2], 2 / 3)
        self.assertAlmostEqual(weights[7], 1.0)
        self.assertAlmostEqual(weights[11], 2.0)

    def test_combined_history_writes_phases_and_missing_metrics(self) -> None:
        histories = {
            "head": {"loss": [1.0, 0.8], "accuracy": [0.4, 0.6]},
            "fine_tune": {"loss": [0.7], "val_loss": [0.75]},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "history.csv"

            _save_combined_history(histories, output)

            with output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["phase"] for row in rows], ["head", "head", "fine_tune"])
            self.assertEqual([row["epoch"] for row in rows], ["1", "2", "1"])
            self.assertEqual(rows[0]["val_loss"], "")
            self.assertEqual(rows[2]["accuracy"], "")

    def test_two_phase_training_selects_best_checkpoint_and_saves_artifacts(self) -> None:
        model = MagicMock(name="training_model")
        model.fit.side_effect = [
            SimpleNamespace(history={"loss": [1.0], "val_loss": [0.8]}),
            SimpleNamespace(history={"loss": [0.7], "val_loss": [0.6]}),
        ]
        model.get_layer.return_value = SimpleNamespace(
            layers=[SimpleNamespace(trainable=True)]
        )
        loaded_model = _SavedModel()
        load_model = MagicMock(return_value=loaded_model)
        fake_tensorflow = types.ModuleType("tensorflow")
        fake_tensorflow.keras = SimpleNamespace(models=SimpleNamespace(load_model=load_model))
        class_weights = {2: 0.75, 7: 1.5}

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / "best_head.keras").write_text("head", encoding="utf-8")
            (output_dir / "best_fine_tune.keras").write_text("fine", encoding="utf-8")

            with (
                patch.dict(sys.modules, {"tensorflow": fake_tensorflow}),
                patch("campis.modeling.compile_model") as compile_model,
                patch("campis.modeling.enable_fine_tuning", return_value=5) as enable,
                patch(
                    "campis.training._callbacks",
                    side_effect=lambda _out, phase, _patience, _monitor="val_loss": [phase],
                ),
            ):
                final_model, histories, best_phase = train_two_phases(
                    model,
                    train_dataset="train",
                    validation_dataset="validation",
                    output_dir=output_dir,
                    head_epochs=2,
                    fine_tune_epochs=1,
                    head_learning_rate=1e-3,
                    fine_tune_learning_rate=1e-5,
                    fine_tune_fraction=0.25,
                    patience=2,
                    class_weights=class_weights,
                    verbose=0,
                )

            self.assertIs(final_model, loaded_model)
            self.assertEqual(best_phase, "fine_tune")
            self.assertEqual(set(histories), {"head", "fine_tune"})
            self.assertEqual(compile_model.call_args_list, [call(model, 1e-3), call(model, 1e-5)])
            enable.assert_called_once_with(model, 0.25)
            self.assertEqual(model.fit.call_count, 2)
            self.assertEqual(model.fit.call_args_list[0].kwargs["class_weight"], class_weights)
            self.assertEqual(model.fit.call_args_list[1].kwargs["class_weight"], class_weights)
            load_model.assert_called_once_with(output_dir / "best_fine_tune.keras")
            self.assertEqual(loaded_model.saved_paths, [output_dir / "model.keras"])
            self.assertTrue((output_dir / "model.keras").is_file())
            self.assertTrue((output_dir / "history.csv").is_file())


class DomainValidationCallbackTests(unittest.TestCase):
    """La seleccion de modelo no puede quedar en manos de la fuente mayoritaria."""

    def _callback(self, probabilities, sources, labels):
        import numpy as np

        callback = build_domain_validation_callback(
            dataset=object(), sources=sources, labels=labels
        )
        callback.set_model(
            SimpleNamespace(
                predict=lambda _dataset, verbose=0: np.asarray(probabilities)
            )
        )
        return callback

    def test_reports_each_domain_and_averages_them_evenly(self) -> None:
        # Nueve muestras de "irish" perfectas y una de "campis" equivocada:
        # la metrica global la absorbe la mayoria, la balanceada no.
        probabilities = [[0.99, 0.005, 0.005]] * 9 + [[0.99, 0.005, 0.005]]
        sources = ["irish"] * 9 + ["campis"]
        labels = [0] * 9 + [1]

        callback = self._callback(probabilities, sources, labels)
        logs: dict[str, float] = {}
        callback.on_epoch_end(0, logs)

        self.assertAlmostEqual(logs["val_irish_accuracy"], 1.0)
        self.assertAlmostEqual(logs["val_campis_accuracy"], 0.0)
        self.assertAlmostEqual(logs["val_accuracy"], 0.9)
        # El promedio entre dominios trata a cada camara por igual.
        self.assertAlmostEqual(logs["val_balanced_accuracy"], 0.5)
        self.assertGreater(logs["val_balanced_loss"], logs["val_loss"])
        self.assertAlmostEqual(
            logs["val_balanced_loss"],
            (logs["val_irish_loss"] + logs["val_campis_loss"]) / 2,
        )

    def test_rejects_predictions_that_do_not_match_the_manifest(self) -> None:
        callback = self._callback([[1.0, 0.0, 0.0]], ["irish", "campis"], [0, 1])

        with self.assertRaises(RuntimeError):
            callback.on_epoch_end(0, {})

    def test_rejects_misaligned_sources_and_labels(self) -> None:
        with self.assertRaises(ValueError):
            build_domain_validation_callback(
                dataset=object(), sources=["irish", "campis"], labels=[0]
            )



if __name__ == "__main__":
    unittest.main()
