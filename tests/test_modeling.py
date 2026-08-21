from __future__ import annotations

import os
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # Permite ``unittest discover -s tests``.
    import _bootstrap  # type: ignore[no-redef]  # noqa: F401

import numpy as np
import tensorflow as tf

from campis.modeling import BACKBONE_NAME, build_model, compile_model, enable_fine_tuning


class ModelingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = build_model(
            image_size=(96, 96),
            num_classes=3,
            dense_units=(16,),
            dropout=0.0,
            weights=None,
            seed=17,
            augmentation=False,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.model = None
        tf.keras.backend.clear_session()

    def test_model_output_for_96px_three_class_input_without_augmentation(self) -> None:
        batch = np.zeros((2, 96, 96, 3), dtype=np.float32)

        probabilities = self.model(batch, training=False).numpy()

        self.assertEqual(self.model.input_shape, (None, 96, 96, 3))
        self.assertEqual(self.model.output_shape, (None, 3))
        self.assertEqual(probabilities.shape, (2, 3))
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2), atol=1e-6)
        with self.assertRaises(ValueError):
            self.model.get_layer("online_augmentation")

    def test_compile_uses_sparse_multiclass_contract(self) -> None:
        returned = compile_model(self.model, learning_rate=1e-3)

        self.assertIs(returned, self.model)
        self.assertIsInstance(self.model.loss, tf.keras.losses.SparseCategoricalCrossentropy)
        self.assertAlmostEqual(float(self.model.optimizer.learning_rate.numpy()), 1e-3)

    def test_fine_tuning_enables_tail_but_keeps_batch_normalization_frozen(self) -> None:
        backbone = self.model.get_layer(BACKBONE_NAME)
        self.assertFalse(backbone.trainable)

        enabled_count = enable_fine_tuning(self.model, fraction=0.30)

        batch_norm_layers = [
            layer
            for layer in backbone.layers
            if isinstance(layer, tf.keras.layers.BatchNormalization)
        ]
        enabled_non_bn = [
            layer
            for layer in backbone.layers
            if not isinstance(layer, tf.keras.layers.BatchNormalization) and layer.trainable
        ]
        frozen_non_bn = [
            layer
            for layer in backbone.layers
            if not isinstance(layer, tf.keras.layers.BatchNormalization) and not layer.trainable
        ]

        self.assertGreater(enabled_count, 0)
        self.assertEqual(enabled_count, len(enabled_non_bn))
        self.assertTrue(batch_norm_layers)
        self.assertTrue(all(not layer.trainable for layer in batch_norm_layers))
        self.assertTrue(frozen_non_bn)


if __name__ == "__main__":
    unittest.main()
