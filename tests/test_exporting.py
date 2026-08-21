from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

try:
    from ._bootstrap import PROJECT_ROOT  # noqa: F401
except ImportError:
    from _bootstrap import PROJECT_ROOT  # noqa: F401

from campis.exporting import (
    create_tflite_interpreter,
    export_model,
    predict_tflite,
    verify_tflite_parity,
)


class ExportingTests(unittest.TestCase):
    def test_keras_tflite_export_and_parity(self) -> None:
        import tensorflow as tf

        inputs = tf.keras.Input((8, 8, 3), dtype=tf.float32)
        x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            keras_path = root / "model.keras"
            model.save(keras_path)
            artifacts = export_model(
                keras_path,
                root / "model.tflite",
                ("early_blight", "healthy", "late_blight"),
            )
            self.assertTrue(artifacts.tflite_path.is_file())
            self.assertTrue(artifacts.labels_path.is_file())
            self.assertTrue(artifacts.metadata_path.is_file())

            images = np.random.default_rng(42).uniform(
                0, 255, size=(2, 8, 8, 3)
            ).astype(np.float32)
            parity = verify_tflite_parity(model, artifacts.tflite_path, images)
            self.assertTrue(parity.passed)

            interpreter = create_tflite_interpreter(artifacts.tflite_path)
            probabilities = predict_tflite(interpreter, images)
            self.assertEqual(probabilities.shape, (2, 3))

    def test_export_rejects_same_labels_and_metadata_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "different"):
            export_model(
                "missing.keras",
                "output.tflite",
                ("a", "b"),
                labels_path="bundle.json",
                metadata_path="bundle.json",
            )


class ParityCriteriaTests(unittest.TestCase):
    """El veredicto debe seguir la decision que se despliega, no un valor suelto."""

    def _report(self, keras_probabilities, tflite_probabilities, **kwargs):
        """Evalua los criterios sobre probabilidades dadas, sin modelos reales."""

        import numpy as np

        from campis import exporting

        keras_array = np.asarray(keras_probabilities, dtype=np.float32)
        tflite_array = np.asarray(tflite_probabilities, dtype=np.float32)
        images = np.zeros((len(keras_array), 4, 4, 3), dtype=np.float32)

        options = {
            "absolute_tolerance": 0.05,
            "mean_absolute_tolerance": 0.01,
        }
        options.update(kwargs)

        with (
            patch.object(exporting, "predict_keras", return_value=keras_array),
            patch.object(exporting, "predict_tflite", return_value=tflite_array),
        ):
            return exporting.verify_tflite_parity(
                object(), "modelo.tflite", images, **options
            )

    def test_one_skewed_outlier_does_not_fail_a_sound_conversion(self) -> None:
        # 63 muestras identicas y una que la cuantizacion volvio mas confiada,
        # sin cambiar la clase: es el patron real de int8.
        keras = [[0.90, 0.07, 0.03]] * 63 + [[0.78, 0.22, 0.00]]
        tflite = [[0.90, 0.07, 0.03]] * 63 + [[0.97, 0.03, 0.00]]

        report = self._report(keras, tflite)

        self.assertTrue(report.passed)
        self.assertEqual(report.class_flips, 0)
        # El maximo es grande, pero el percentil 99 se mantiene bajo.
        self.assertGreater(report.max_absolute_error, 0.15)
        self.assertLess(report.percentile_absolute_error, 0.05)

    def test_a_flip_on_a_confident_prediction_fails(self) -> None:
        # Margen amplio en Keras (0,80 contra 0,15) y TFLite elige otra clase:
        # eso no lo explica el ruido de cuantizacion.
        keras = [[0.90, 0.07, 0.03]] * 63 + [[0.80, 0.15, 0.05]]
        tflite = [[0.90, 0.07, 0.03]] * 63 + [[0.10, 0.85, 0.05]]

        report = self._report(keras, tflite)

        self.assertFalse(report.passed)
        self.assertEqual(report.unexplained_class_flips, 1)
        self.assertAlmostEqual(report.largest_unexplained_margin, 0.65, places=4)
        self.assertLess(report.class_agreement_rate, 1.0)

    def test_a_flip_on_a_tie_is_accepted(self) -> None:
        # Empate tecnico: 0,455 contra 0,445. Cualquiera de las dos clases es
        # defendible y el orden depende de centesimas.
        keras = [[0.90, 0.07, 0.03]] * 63 + [[0.455, 0.445, 0.10]]
        tflite = [[0.90, 0.07, 0.03]] * 63 + [[0.445, 0.455, 0.10]]

        report = self._report(keras, tflite)

        self.assertTrue(report.passed)
        self.assertEqual(report.class_flips, 1)
        self.assertEqual(report.unexplained_class_flips, 0)
        self.assertFalse(report.predicted_classes_match)

    def test_a_broadly_wrong_conversion_fails_on_the_mean(self) -> None:
        # Todas las salidas desplazadas: una conversion rota, no un caso limite.
        keras = [[0.90, 0.07, 0.03]] * 64
        tflite = [[0.70, 0.20, 0.10]] * 64

        report = self._report(keras, tflite)

        self.assertFalse(report.passed)
        self.assertFalse(report.probabilities_close)
        self.assertGreater(report.mean_absolute_error, 0.01)
        # La clase no cambio: sin el criterio sobre la media pasaria inadvertida.
        self.assertEqual(report.class_flips, 0)

    def test_relative_error_ignores_probabilities_near_zero(self) -> None:
        # 0,0001 contra 0,0009 es un error relativo de 8, y no significa nada.
        keras = [[0.90, 0.0999, 0.0001]] * 64
        tflite = [[0.90, 0.0991, 0.0009]] * 64

        report = self._report(keras, tflite, relative_floor=0.01)

        self.assertTrue(report.passed)
        self.assertLess(report.max_relative_error, 0.05)

    def test_an_invalid_percentile_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._report([[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], error_percentile=0.0)
        with self.assertRaises(ValueError):
            self._report([[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], error_percentile=101.0)



if __name__ == "__main__":
    unittest.main()

