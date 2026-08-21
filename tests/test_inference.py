from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from ._bootstrap import PROJECT_ROOT  # noqa: F401
except ImportError:
    from _bootstrap import PROJECT_ROOT  # noqa: F401

from campis.inference import discover_inputs, load_image, load_labels, predict_files


class InferenceTests(unittest.TestCase):
    def test_discovery_labels_and_tensorflow_resize(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "leaf.jpg"
            Image.fromarray(np.full((20, 10, 3), 127, dtype=np.uint8)).save(image_path)
            (root / "ignore.txt").write_text("x", encoding="utf-8")
            labels_path = root / "labels.json"
            labels_path.write_text(
                json.dumps({"labels": ["early_blight", "healthy", "late_blight"]}),
                encoding="utf-8",
            )

            self.assertEqual(discover_inputs(root, (".jpg",)), [image_path])
            self.assertEqual(load_labels(labels_path)[1], "healthy")
            image = load_image(image_path, (8, 12))
            self.assertEqual(image.shape, (8, 12, 3))
            self.assertEqual(image.dtype, np.float32)
            self.assertGreaterEqual(float(image.min()), 0.0)
            self.assertLessEqual(float(image.max()), 255.0)

    def test_predict_rejects_invalid_batch_size_before_runtime(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size"):
            predict_files(
                "missing.keras",
                ["leaf.jpg"],
                ("a", "b"),
                (32, 32),
                runtime="keras",
                batch_size=0,
            )


if __name__ == "__main__":
    unittest.main()
