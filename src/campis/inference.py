"""Carga de imágenes y predicción reutilizable para Keras y TFLite."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .exporting import create_tflite_interpreter, predict_keras, predict_tflite
from .data import apply_exif_orientation, center_crop_square


@dataclass(frozen=True, slots=True)
class Prediction:
    path: str
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_labels(path: str | Path) -> tuple[str, ...]:
    """Lee el `labels.json` generado junto al modelo."""

    labels_path = Path(path)
    if not labels_path.is_file():
        raise FileNotFoundError(f"No existe el archivo de etiquetas: {labels_path}")
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    labels = payload.get("labels") if isinstance(payload, dict) else None
    if not isinstance(labels, list) or len(labels) < 2:
        raise ValueError(f"Archivo de etiquetas inválido: {labels_path}")
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError(f"Archivo de etiquetas inválido: {labels_path}")
    return tuple(labels)


def discover_inputs(
    input_path: str | Path,
    extensions: Iterable[str],
) -> list[Path]:
    """Devuelve una imagen o todas las imágenes directas de una carpeta."""

    path = Path(input_path)
    normalized_extensions = {
        extension.casefold() if extension.startswith(".") else f".{extension.casefold()}"
        for extension in extensions
    }
    if path.is_file():
        if path.suffix.casefold() not in normalized_extensions:
            raise ValueError(f"Extensión de imagen no soportada: {path.suffix}")
        return [path]
    if path.is_dir():
        files = sorted(
            candidate
            for candidate in path.iterdir()
            if candidate.is_file()
            and candidate.suffix.casefold() in normalized_extensions
        )
        if not files:
            raise ValueError(f"No se encontraron imágenes compatibles en: {path}")
        return files
    raise FileNotFoundError(f"No existe la entrada de inferencia: {path}")


def load_image(
    path: str | Path,
    image_size: Sequence[int],
    *,
    crop_to_square: bool = True,
) -> np.ndarray:
    """Carga RGB float32 en rango 0..255 y forma `(alto, ancho, 3)`.

    ``crop_to_square`` debe coincidir con el valor usado al entrenar; el
    bundle exportado lo registra en ``model.metadata.json``.
    """

    if len(image_size) != 2:
        raise ValueError("image_size debe contener [alto, ancho].")
    height, width = (int(value) for value in image_size)
    if height <= 0 or width <= 0:
        raise ValueError("Las dimensiones de image_size deben ser positivas.")

    image_path = Path(path)
    try:
        import tensorflow as tf
        from PIL import Image

        with Image.open(image_path) as metadata:
            raw_orientation = metadata.getexif().get(274, 1)
        try:
            orientation = int(raw_orientation)
        except (TypeError, ValueError):
            orientation = 1
        if not 1 <= orientation <= 8:
            orientation = 1

        encoded = tf.io.read_file(str(image_path))
        image = tf.io.decode_image(encoded, channels=3, expand_animations=False)
        image.set_shape((None, None, 3))
        image = apply_exif_orientation(image, orientation)
        if crop_to_square:
            image = center_crop_square(image)
        image = tf.image.resize(
            image, (height, width), method="bilinear", antialias=True
        )
        array = tf.clip_by_value(tf.cast(image, tf.float32), 0.0, 255.0).numpy()
    except Exception as error:
        raise ValueError(f"No se pudo leer la imagen {image_path}: {error}") from error
    return array


def predict_files(
    model: Any,
    files: Sequence[str | Path],
    labels: Sequence[str],
    image_size: Sequence[int],
    *,
    runtime: str,
    batch_size: int = 32,
    crop_to_square: bool = True,
) -> list[Prediction]:
    """Predice una lista de archivos con el runtime seleccionado."""

    if runtime not in {"keras", "tflite"}:
        raise ValueError("runtime debe ser 'keras' o 'tflite'.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size debe ser un entero mayor que cero.")
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("labels debe contener nombres únicos en orden de salida.")
    if not files:
        raise ValueError("No hay archivos para predecir.")

    resolved_files = [Path(path) for path in files]
    runtime_model = model
    if isinstance(model, (str, Path)):
        if runtime == "keras":
            import tensorflow as tf

            runtime_model = tf.keras.models.load_model(model, compile=False)
        else:
            runtime_model = create_tflite_interpreter(model)
    results: list[Prediction] = []
    for start in range(0, len(resolved_files), batch_size):
        batch_files = resolved_files[start : start + batch_size]
        images = np.stack(
            [
                load_image(path, image_size, crop_to_square=crop_to_square)
                for path in batch_files
            ]
        )
        probabilities = (
            predict_keras(runtime_model, images, batch_size=batch_size)
            if runtime == "keras"
            else predict_tflite(runtime_model, images)
        )
        if probabilities.shape[1] != len(labels):
            raise ValueError(
                f"El modelo devuelve {probabilities.shape[1]} clases y labels contiene "
                f"{len(labels)}."
            )
        for image_path, row in zip(batch_files, probabilities):
            predicted_index = int(np.argmax(row))
            results.append(
                Prediction(
                    path=str(image_path),
                    predicted_class=labels[predicted_index],
                    confidence=float(row[predicted_index]),
                    probabilities={
                        label: float(row[index]) for index, label in enumerate(labels)
                    },
                )
            )
    return results


__all__ = [
    "Prediction",
    "discover_inputs",
    "load_image",
    "load_labels",
    "predict_files",
]
