"""Keras/TFLite export, metadata, inference, and parity utilities.

Every prediction helper in this module accepts an RGB image or batch represented
as ``float32`` values in ``[0, 255]``.  No helper divides by 255: preprocessing
is part of the MobileNetV3 model built by :mod:`campis.modeling`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Sequence


PathLike = str | os.PathLike[str]


@dataclass(frozen=True, slots=True)
class ExportArtifacts:
    """Paths produced by :func:`export_model`."""

    tflite_path: Path
    labels_path: Path
    metadata_path: Path


@dataclass(frozen=True, slots=True)
class ParityReport:
    """Numerical and classification agreement between Keras and TFLite."""

    passed: bool
    probabilities_close: bool
    predicted_classes_match: bool
    sample_count: int
    output_count: int
    max_absolute_error: float
    mean_absolute_error: float
    max_relative_error: float
    absolute_tolerance: float
    relative_tolerance: float
    mean_absolute_tolerance: float = 0.0
    percentile_absolute_error: float = 0.0
    error_percentile: float = 99.0
    relative_floor: float = 0.0
    class_flips: int = 0
    unexplained_class_flips: int = 0
    class_agreement_rate: float = 1.0
    largest_unexplained_margin: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the report."""

        return asdict(self)


def _require_tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "TensorFlow is required for Keras export and TFLite inference. "
            "Install the project's training dependencies before calling this function."
        ) from exc
    return tf


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - NumPy is a core dependency
        raise ImportError(
            "NumPy is required for model prediction and parity verification."
        ) from exc
    return np


def _existing_model_path(value: PathLike, suffix: str, description: str) -> Path:
    path = Path(value)
    if path.suffix.lower() != suffix:
        raise ValueError(f"{description} must use the {suffix!r} extension: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist or is not a file: {path}")
    return path


def _output_path(value: PathLike, suffix: str, description: str) -> Path:
    path = Path(value)
    if path.suffix.lower() != suffix:
        raise ValueError(f"{description} must use the {suffix!r} extension: {path}")
    return path


def _normalise_labels(labels: Sequence[str]) -> tuple[str, ...]:
    if isinstance(labels, (str, bytes)):
        raise TypeError("labels must be a sequence of class-name strings, not one string.")
    try:
        normalised = tuple(labels)
    except TypeError as exc:
        raise TypeError("labels must be a sequence of non-empty strings.") from exc

    if len(normalised) < 2:
        raise ValueError("At least two labels are required for multiclass classification.")
    if any(not isinstance(label, str) or not label.strip() for label in normalised):
        raise ValueError("Every label must be a non-empty string.")
    if len(set(normalised)) != len(normalised):
        raise ValueError("labels must be unique and in model-output order.")
    return normalised


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        encoded = (
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TypeError("Metadata contains a value that cannot be encoded as JSON.") from exc
    _atomic_write_bytes(path, encoded)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_image_batch(images: Any) -> Any:
    """Validate and return a contiguous NHWC float32 batch in [0, 255]."""

    np = _require_numpy()
    try:
        batch = np.asarray(images)
    except Exception as exc:
        raise TypeError("images must be convertible to a numeric NumPy array.") from exc

    if batch.ndim == 3:
        batch = batch[np.newaxis, ...]
    if batch.ndim != 4:
        raise ValueError(
            "images must have shape (height, width, 3) or "
            f"(batch, height, width, 3); received {batch.shape}."
        )
    if batch.shape[0] < 1:
        raise ValueError("images must contain at least one sample.")
    if batch.shape[1] < 1 or batch.shape[2] < 1:
        raise ValueError("images must have positive height and width.")
    if batch.shape[-1] != 3:
        raise ValueError(f"images must contain exactly 3 RGB channels; received {batch.shape}.")
    if not np.issubdtype(batch.dtype, np.number):
        raise TypeError(f"images must contain numeric values; received dtype {batch.dtype}.")

    batch = np.ascontiguousarray(batch, dtype=np.float32)
    if not np.isfinite(batch).all():
        raise ValueError("images contain NaN or infinite values.")
    minimum = float(batch.min())
    maximum = float(batch.max())
    if minimum < 0.0 or maximum > 255.0:
        raise ValueError(
            "images must use the model input range [0, 255]; "
            f"observed range [{minimum:.6g}, {maximum:.6g}]."
        )
    return batch


def _validate_positive_integer(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer or None.")
    return int(value)


def convert_keras_to_tflite(
    keras_model_path: PathLike,
    tflite_model_path: PathLike,
    *,
    optimize: bool = False,
) -> Path:
    """Convert a ``.keras`` model to TFLite without external preprocessing.

    ``optimize=True`` enables TensorFlow's default (normally dynamic-range)
    optimisation.  It does not add a rescaling step and therefore preserves the
    public ``float32`` ``[0, 255]`` input contract of the saved Keras model.
    """

    if not isinstance(optimize, bool):
        raise TypeError("optimize must be a boolean.")
    keras_path = _existing_model_path(keras_model_path, ".keras", "Keras model")
    tflite_path = _output_path(tflite_model_path, ".tflite", "TFLite output")
    if keras_path.absolute() == tflite_path.absolute():
        raise ValueError("Keras input and TFLite output paths must be different.")

    tf = _require_tensorflow()
    try:
        model = tf.keras.models.load_model(str(keras_path), compile=False)
    except Exception as exc:
        raise RuntimeError(f"Could not load Keras model: {keras_path}") from exc

    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        if optimize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_bytes = converter.convert()
    except Exception as exc:
        raise RuntimeError(f"TFLite conversion failed for: {keras_path}") from exc

    if not isinstance(tflite_bytes, bytes) or tflite_bytes[4:8] != b"TFL3":
        raise RuntimeError("TensorFlow returned an invalid TFLite FlatBuffer.")
    _atomic_write_bytes(tflite_path, tflite_bytes)
    return tflite_path


def _create_interpreter(tflite_model_path: PathLike, num_threads: int | None = None) -> Any:
    model_path = _existing_model_path(tflite_model_path, ".tflite", "TFLite model")
    num_threads = _validate_positive_integer(num_threads, "num_threads")
    tf = _require_tensorflow()
    kwargs: dict[str, Any] = {"model_path": str(model_path)}
    if num_threads is not None:
        kwargs["num_threads"] = num_threads
    try:
        interpreter = tf.lite.Interpreter(**kwargs)
        interpreter.allocate_tensors()
    except Exception as exc:
        raise RuntimeError(f"Could not initialise TFLite model: {model_path}") from exc
    return interpreter


def create_tflite_interpreter(
    tflite_model_path: PathLike, num_threads: int | None = None
) -> Any:
    """Carga y prepara una vez un intérprete reutilizable para varios batches."""

    return _create_interpreter(tflite_model_path, num_threads=num_threads)


def _single_io_details(interpreter: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(
            "CAMPIS classifiers must have exactly one input and one output; "
            f"the TFLite model has {len(inputs)} input(s) and {len(outputs)} output(s)."
        )
    return inputs[0], outputs[0]


def _shape_list(details: Mapping[str, Any], *, signature: bool = True) -> list[int]:
    key = "shape_signature" if signature and "shape_signature" in details else "shape"
    value = details[key]
    return [int(dimension) for dimension in value]


def _quantization_metadata(details: Mapping[str, Any]) -> dict[str, Any] | None:
    parameters = details.get("quantization_parameters") or {}
    scales = parameters.get("scales")
    zero_points = parameters.get("zero_points")
    scale_values = [] if scales is None else [float(value) for value in scales]
    zero_values = [] if zero_points is None else [int(value) for value in zero_points]
    if not scale_values:
        legacy_scale, legacy_zero = details.get("quantization", (0.0, 0))
        if float(legacy_scale) == 0.0:
            return None
        scale_values = [float(legacy_scale)]
        zero_values = [int(legacy_zero)]
    return {
        "scales": scale_values,
        "zero_points": zero_values,
        "quantized_dimension": int(parameters.get("quantized_dimension", 0)),
    }


def _serialise_tensor_details(details: Mapping[str, Any]) -> dict[str, Any]:
    np = _require_numpy()
    dtype = np.dtype(details["dtype"])
    return {
        "name": str(details.get("name", "")),
        "dtype": dtype.name,
        "shape": _shape_list(details, signature=False),
        "shape_signature": _shape_list(details, signature=True),
        "quantization": _quantization_metadata(details),
    }


def save_labels_json(output_path: PathLike, labels: Sequence[str]) -> Path:
    """Save class names in their exact model-output order."""

    normalised_labels = _normalise_labels(labels)
    path = _output_path(output_path, ".json", "Labels JSON output")
    _atomic_write_json(
        path,
        {
            "schema_version": 1,
            "labels": list(normalised_labels),
        },
    )
    return path


def save_metadata_json(
    output_path: PathLike,
    labels: Sequence[str],
    tflite_model_path: PathLike,
    *,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Inspect a TFLite file and save its deployment contract as JSON."""

    normalised_labels = _normalise_labels(labels)
    metadata_path = _output_path(output_path, ".json", "Metadata JSON output")
    model_path = _existing_model_path(tflite_model_path, ".tflite", "TFLite model")
    if extra is not None and not isinstance(extra, Mapping):
        raise TypeError("extra metadata must be a mapping or None.")

    interpreter = _create_interpreter(model_path)
    input_details, output_details = _single_io_details(interpreter)
    input_shape = _shape_list(input_details, signature=True)
    output_shape = _shape_list(output_details, signature=True)

    if len(input_shape) != 4 or input_shape[-1] not in (-1, 3):
        raise ValueError(
            "Expected a rank-4 NHWC RGB TFLite input; "
            f"received shape signature {input_shape}."
        )
    if len(output_shape) != 2:
        raise ValueError(
            "Expected a rank-2 classifier output; "
            f"received shape signature {output_shape}."
        )
    if output_shape[-1] > 0 and output_shape[-1] != len(normalised_labels):
        raise ValueError(
            f"The TFLite model produces {output_shape[-1]} classes but "
            f"{len(normalised_labels)} labels were supplied."
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "model": {
            "filename": model_path.name,
            "format": "tflite",
            "sha256": _sha256_file(model_path),
        },
        "input": {
            **_serialise_tensor_details(input_details),
            "color_space": "RGB",
            "value_range": [0.0, 255.0],
            "preprocessing": "embedded_mobilenet_v3",
        },
        "output": {
            **_serialise_tensor_details(output_details),
            "semantics": "class_probabilities",
        },
        "labels": list(normalised_labels),
        "num_classes": len(normalised_labels),
    }
    if extra:
        payload["extra"] = dict(extra)
    _atomic_write_json(metadata_path, payload)
    return metadata_path


def export_model(
    keras_model_path: PathLike,
    tflite_model_path: PathLike,
    labels: Sequence[str],
    *,
    labels_path: PathLike | None = None,
    metadata_path: PathLike | None = None,
    optimize: bool = False,
    extra_metadata: Mapping[str, Any] | None = None,
) -> ExportArtifacts:
    """Convert a Keras model and write adjacent labels and metadata files."""

    normalised_labels = _normalise_labels(labels)
    requested_tflite_path = _output_path(
        tflite_model_path, ".tflite", "TFLite output"
    )
    default_stem = requested_tflite_path.with_suffix("")
    resolved_labels_path = (
        Path(labels_path)
        if labels_path is not None
        else default_stem.with_name(f"{default_stem.name}.labels.json")
    )
    resolved_metadata_path = (
        Path(metadata_path)
        if metadata_path is not None
        else default_stem.with_name(f"{default_stem.name}.metadata.json")
    )
    if resolved_labels_path.absolute() == resolved_metadata_path.absolute():
        raise ValueError("labels_path and metadata_path must be different files.")

    tflite_path = convert_keras_to_tflite(
        keras_model_path,
        requested_tflite_path,
        optimize=optimize,
    )
    labels_json = save_labels_json(resolved_labels_path, normalised_labels)
    metadata_json = save_metadata_json(
        resolved_metadata_path,
        normalised_labels,
        tflite_path,
        extra=extra_metadata,
    )
    return ExportArtifacts(
        tflite_path=tflite_path,
        labels_path=labels_json,
        metadata_path=metadata_json,
    )


def _load_keras_model(model_or_path: Any) -> Any:
    if isinstance(model_or_path, (str, os.PathLike)):
        model_path = _existing_model_path(model_or_path, ".keras", "Keras model")
        tf = _require_tensorflow()
        try:
            return tf.keras.models.load_model(str(model_path), compile=False)
        except Exception as exc:
            raise RuntimeError(f"Could not load Keras model: {model_path}") from exc
    if model_or_path is None or not callable(model_or_path):
        raise TypeError("model_or_path must be a Keras model or a path to a .keras file.")
    return model_or_path


def _validate_keras_input_shape(model: Any, batch: Any) -> None:
    input_shape = getattr(model, "input_shape", None)
    if input_shape is None:
        return
    if isinstance(input_shape, list):
        raise ValueError("CAMPIS classifiers must have exactly one Keras input.")
    expected = tuple(input_shape)
    if len(expected) != 4:
        raise ValueError(f"Expected a rank-4 Keras input, received {expected}.")
    for expected_dimension, actual_dimension in zip(expected[1:], batch.shape[1:]):
        if expected_dimension is not None and int(expected_dimension) != int(actual_dimension):
            raise ValueError(
                f"Image batch shape {batch.shape} does not match model input {expected}."
            )


def predict_keras(model_or_path: Any, images: Any, *, batch_size: int | None = None) -> Any:
    """Return Keras probabilities for RGB data in ``float32 [0, 255]``."""

    batch_size = _validate_positive_integer(batch_size, "batch_size")
    batch = _normalise_image_batch(images)
    model = _load_keras_model(model_or_path)
    _validate_keras_input_shape(model, batch)
    np = _require_numpy()

    try:
        predictions = model.predict(batch, batch_size=batch_size, verbose=0)
    except Exception as exc:
        raise RuntimeError("Keras prediction failed for the supplied image batch.") from exc
    probabilities = np.asarray(predictions, dtype=np.float32)
    if probabilities.ndim != 2 or probabilities.shape[0] != batch.shape[0]:
        raise ValueError(
            "Expected Keras output shape (batch, classes); "
            f"received {probabilities.shape}."
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("Keras predictions contain NaN or infinite values.")
    return probabilities


def _quantization_arrays(details: Mapping[str, Any], np: Any) -> tuple[Any, Any, int]:
    parameters = details.get("quantization_parameters") or {}
    scales = np.asarray(parameters.get("scales", []), dtype=np.float32)
    zero_points = np.asarray(parameters.get("zero_points", []), dtype=np.float32)
    axis = int(parameters.get("quantized_dimension", 0))
    if scales.size == 0:
        legacy_scale, legacy_zero = details.get("quantization", (0.0, 0))
        if float(legacy_scale) == 0.0:
            raise ValueError(
                f"Tensor {details.get('name', '<unnamed>')!r} uses an integer dtype "
                "but has no valid quantization scale."
            )
        scales = np.asarray([legacy_scale], dtype=np.float32)
        zero_points = np.asarray([legacy_zero], dtype=np.float32)
    if not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise ValueError("Invalid TFLite quantization parameters: scales must be positive.")
    if zero_points.size not in (1, scales.size):
        raise ValueError("Invalid TFLite quantization parameters: scales/zero-points differ.")
    return scales, zero_points, axis


def _broadcast_quantization(
    values: Any,
    scales: Any,
    zero_points: Any,
    axis: int,
) -> tuple[Any, Any]:
    if scales.size == 1:
        return scales.reshape(()), zero_points.reshape(())
    if axis < 0 or axis >= values.ndim or values.shape[axis] != scales.size:
        raise ValueError(
            "Per-axis TFLite quantization parameters do not match the tensor shape."
        )
    shape = [1] * values.ndim
    shape[axis] = scales.size
    broadcast_zero_points = (
        zero_points.reshape(shape) if zero_points.size > 1 else zero_points.reshape(())
    )
    return scales.reshape(shape), broadcast_zero_points


def _encode_tflite_input(values: Any, details: Mapping[str, Any], np: Any) -> Any:
    dtype = np.dtype(details["dtype"])
    if np.issubdtype(dtype, np.floating):
        return values.astype(dtype, copy=False)
    if not np.issubdtype(dtype, np.integer):
        raise TypeError(f"Unsupported TFLite input dtype: {dtype}.")

    scales, zero_points, axis = _quantization_arrays(details, np)
    scales, zero_points = _broadcast_quantization(values, scales, zero_points, axis)
    quantized = np.rint(values / scales + zero_points)
    limits = np.iinfo(dtype)
    return np.clip(quantized, limits.min, limits.max).astype(dtype)


def _decode_tflite_output(values: Any, details: Mapping[str, Any], np: Any) -> Any:
    dtype = np.dtype(details["dtype"])
    if np.issubdtype(dtype, np.floating):
        return values.astype(np.float32, copy=False)
    if not np.issubdtype(dtype, np.integer):
        raise TypeError(f"Unsupported TFLite output dtype: {dtype}.")

    scales, zero_points, axis = _quantization_arrays(details, np)
    scales, zero_points = _broadcast_quantization(values, scales, zero_points, axis)
    return (values.astype(np.float32) - zero_points) * scales


def _prepare_single_sample_interpreter(
    interpreter: Any,
    sample_shape: Sequence[int],
) -> tuple[Any, Any]:
    input_details, output_details = _single_io_details(interpreter)
    signature = _shape_list(input_details, signature=True)
    target = [int(dimension) for dimension in sample_shape]
    if len(signature) != len(target):
        raise ValueError(
            f"TFLite input rank {len(signature)} does not match image rank {len(target)}."
        )
    for expected, actual in zip(signature, target):
        if expected != -1 and expected != actual:
            raise ValueError(
                f"TFLite input shape signature {signature} does not accept {target}."
            )

    allocated = _shape_list(input_details, signature=False)
    if allocated != target:
        try:
            interpreter.resize_tensor_input(input_details["index"], target, strict=True)
            interpreter.allocate_tensors()
        except Exception as exc:
            raise ValueError(
                f"Could not resize TFLite input from {allocated} to {target}."
            ) from exc
        input_details, output_details = _single_io_details(interpreter)
    return input_details, output_details


def predict_tflite(
    tflite_model_path: PathLike | Any,
    images: Any,
    *,
    num_threads: int | None = None,
) -> Any:
    """Return TFLite probabilities without applying an extra normalization."""

    batch = _normalise_image_batch(images)
    if isinstance(tflite_model_path, (str, os.PathLike)):
        interpreter = _create_interpreter(tflite_model_path, num_threads=num_threads)
    elif all(
        hasattr(tflite_model_path, attribute)
        for attribute in ("get_input_details", "get_output_details", "invoke")
    ):
        if num_threads is not None:
            raise ValueError(
                "num_threads solo puede configurarse al crear el intérprete desde una ruta."
            )
        interpreter = tflite_model_path
    else:
        raise TypeError(
            "tflite_model_path debe ser una ruta .tflite o un intérprete preparado."
        )
    np = _require_numpy()
    outputs: list[Any] = []

    # One sample at a time also supports exported models whose allocated batch
    # dimension is fixed at one, without changing their spatial signature.
    for sample in batch:
        sample_batch = sample[np.newaxis, ...]
        input_details, output_details = _prepare_single_sample_interpreter(
            interpreter,
            sample_batch.shape,
        )
        encoded = _encode_tflite_input(sample_batch, input_details, np)
        try:
            interpreter.set_tensor(input_details["index"], encoded)
            interpreter.invoke()
            raw_output = interpreter.get_tensor(output_details["index"])
        except Exception as exc:
            raise RuntimeError("TFLite inference failed for an image sample.") from exc
        output = np.asarray(_decode_tflite_output(raw_output, output_details, np))
        if output.ndim == 1:
            output = output[np.newaxis, ...]
        if output.ndim != 2 or output.shape[0] != 1:
            raise ValueError(
                "Expected TFLite output shape (1, classes); "
                f"received {output.shape}."
            )
        outputs.append(output.astype(np.float32, copy=False))

    probabilities = np.concatenate(outputs, axis=0)
    if not np.isfinite(probabilities).all():
        raise ValueError("TFLite predictions contain NaN or infinite values.")
    return probabilities


def _normalise_tolerance(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a non-negative real number.")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite, non-negative value.")
    return value


def verify_tflite_parity(
    keras_model_or_path: Any,
    tflite_model_path: PathLike,
    images: Any,
    *,
    absolute_tolerance: float = 1e-5,
    relative_tolerance: float = 1e-4,
    mean_absolute_tolerance: float | None = None,
    error_percentile: float = 99.0,
    relative_floor: float = 0.01,
    num_threads: int | None = None,
    raise_on_failure: bool = False,
) -> ParityReport:
    """Compare Keras and TFLite probabilities on the same image batch.

    The outputs are softmax probabilities bounded in ``[0, 1]``, so absolute
    error is the meaningful measure and drives the verdict. Relative error is
    reported for diagnosis only, and only over entries whose reference
    probability reaches ``relative_floor``: on a probability near zero, a
    deviation of 0.001 yields a relative error near 1.0 that says nothing about
    the conversion.

    A quantised model may flip the predicted class when the top two
    probabilities are closer together than the quantisation noise itself. Such
    a flip is expected and is not counted against the export; a flip on a
    confident prediction is a real defect and fails the check.

    The deviation is judged at ``error_percentile``, not at the maximum. The
    quantisation error distribution is extremely skewed — most outputs match to
    five decimals while a single uncertain sample can move by tenths — and the
    maximum of a skewed sample grows with the sample size, so a maximum-based
    threshold silently tightens as more images are compared. The percentile is
    stable. The maximum is still reported for diagnosis.

    Three conditions must hold to pass: the deviation at ``error_percentile``
    stays within ``absolute_tolerance``, the mean deviation stays within
    ``mean_absolute_tolerance`` (a fifth of the absolute one by default), and
    no class flip happens on a prediction whose margin exceeded the tolerance.
    """

    absolute_tolerance = _normalise_tolerance(absolute_tolerance, "absolute_tolerance")
    relative_tolerance = _normalise_tolerance(relative_tolerance, "relative_tolerance")
    if mean_absolute_tolerance is None:
        mean_absolute_tolerance = absolute_tolerance / 5.0
    mean_absolute_tolerance = _normalise_tolerance(
        mean_absolute_tolerance, "mean_absolute_tolerance"
    )
    relative_floor = _normalise_tolerance(relative_floor, "relative_floor")
    if (
        isinstance(error_percentile, bool)
        or not isinstance(error_percentile, (int, float))
        or not 0.0 < float(error_percentile) <= 100.0
    ):
        raise ValueError("error_percentile debe estar en el intervalo (0, 100].")
    error_percentile = float(error_percentile)
    if not isinstance(raise_on_failure, bool):
        raise TypeError("raise_on_failure must be a boolean.")

    # Normalise once so both runtimes receive byte-for-byte equivalent floats.
    batch = _normalise_image_batch(images)
    keras_probabilities = predict_keras(keras_model_or_path, batch)
    tflite_probabilities = predict_tflite(
        tflite_model_path,
        batch,
        num_threads=num_threads,
    )
    if keras_probabilities.shape != tflite_probabilities.shape:
        raise ValueError(
            "Keras and TFLite output shapes differ: "
            f"{keras_probabilities.shape} versus {tflite_probabilities.shape}."
        )

    np = _require_numpy()
    absolute_error = np.abs(keras_probabilities - tflite_probabilities)
    max_absolute_error = float(absolute_error.max(initial=0.0))
    mean_absolute_error = float(absolute_error.mean())
    percentile_absolute_error = (
        float(np.percentile(absolute_error, error_percentile))
        if absolute_error.size
        else 0.0
    )

    # El error relativo solo se mide donde la referencia es lo bastante grande
    # para que dividir por ella signifique algo.
    significant = np.abs(keras_probabilities) >= relative_floor
    if bool(significant.any()):
        max_relative_error = float(
            (absolute_error[significant] / np.abs(keras_probabilities[significant])).max()
        )
    else:
        max_relative_error = 0.0

    probabilities_close = bool(
        percentile_absolute_error <= absolute_tolerance
        and mean_absolute_error <= mean_absolute_tolerance
    )

    keras_classes = np.argmax(keras_probabilities, axis=1)
    tflite_classes = np.argmax(tflite_probabilities, axis=1)
    flipped = keras_classes != tflite_classes
    class_flips = int(flipped.sum())
    sample_count = int(keras_probabilities.shape[0])
    predicted_classes_match = class_flips == 0

    # Margen entre las dos clases mas probables segun Keras: si es menor que la
    # tolerancia, el ruido de cuantizacion basta para explicar el cambio.
    if keras_probabilities.shape[1] >= 2:
        ordered = np.sort(keras_probabilities, axis=1)
        margins = ordered[:, -1] - ordered[:, -2]
    else:
        margins = np.full(sample_count, np.inf, dtype=np.float64)

    unexplained = np.logical_and(flipped, margins > absolute_tolerance)
    unexplained_class_flips = int(unexplained.sum())
    largest_unexplained_margin = (
        float(margins[unexplained].max()) if unexplained_class_flips else 0.0
    )

    report = ParityReport(
        passed=probabilities_close and unexplained_class_flips == 0,
        probabilities_close=probabilities_close,
        predicted_classes_match=predicted_classes_match,
        sample_count=sample_count,
        output_count=int(keras_probabilities.shape[1]),
        max_absolute_error=max_absolute_error,
        mean_absolute_error=mean_absolute_error,
        max_relative_error=max_relative_error,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        mean_absolute_tolerance=mean_absolute_tolerance,
        percentile_absolute_error=percentile_absolute_error,
        error_percentile=error_percentile,
        relative_floor=relative_floor,
        class_flips=class_flips,
        unexplained_class_flips=unexplained_class_flips,
        class_agreement_rate=(
            float((sample_count - class_flips) / sample_count) if sample_count else 1.0
        ),
        largest_unexplained_margin=largest_unexplained_margin,
    )
    if raise_on_failure and not report.passed:
        raise RuntimeError(
            "Keras/TFLite parity verification failed: "
            f"p{error_percentile:g}_abs_error={report.percentile_absolute_error:.6g} "
            f"(tolerancia {absolute_tolerance:.6g}), "
            f"max_abs_error={report.max_absolute_error:.6g}, "
            f"mean_abs_error={report.mean_absolute_error:.6g} "
            f"(tolerancia {mean_absolute_tolerance:.6g}), "
            f"cambios de clase no explicados={report.unexplained_class_flips} "
            f"de {sample_count}."
        )
    return report


__all__ = [
    "ExportArtifacts",
    "ParityReport",
    "convert_keras_to_tflite",
    "create_tflite_interpreter",
    "export_model",
    "predict_keras",
    "predict_tflite",
    "save_labels_json",
    "save_metadata_json",
    "verify_tflite_parity",
]
