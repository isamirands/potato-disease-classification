"""Model construction and fine-tuning helpers for CAMPIS.

The public contract of models built here is deliberately simple: callers pass
``float32`` RGB tensors whose values are in the ``[0, 255]`` range.  The
MobileNetV3 preprocessing layer remains embedded in the backbone, so training,
Keras inference, and TFLite inference all use the same preprocessing path.

TensorFlow is imported lazily so configuration, manifest, and audit commands
can import the CAMPIS package on machines that do not have the training stack
installed.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any, Sequence


BACKBONE_NAME = "mobilenet_v3_small_backbone"


def _require_tensorflow() -> Any:
    """Import TensorFlow or raise an actionable dependency error."""

    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "TensorFlow is required for model construction and training. "
            "Install the project's training dependencies before calling this function."
        ) from exc
    return tf


def _normalise_image_size(image_size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(image_size, bool):
        raise TypeError("image_size must be an integer or a (height, width) pair.")

    if isinstance(image_size, Integral):
        height = width = int(image_size)
    else:
        try:
            values = tuple(image_size)
        except TypeError as exc:
            raise TypeError(
                "image_size must be an integer or a (height, width) pair."
            ) from exc
        if len(values) != 2:
            raise ValueError(
                f"image_size must contain exactly two values; received {values!r}."
            )
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in values):
            raise TypeError("image_size height and width must be integers.")
        height, width = (int(value) for value in values)

    if height < 32 or width < 32:
        raise ValueError(
            "MobileNetV3Small requires image dimensions of at least 32 pixels; "
            f"received {(height, width)}."
        )
    return height, width


def _normalise_dense_units(dense_units: Sequence[int]) -> tuple[int, ...]:
    try:
        units = tuple(dense_units)
    except TypeError as exc:
        raise TypeError("dense_units must be a sequence of positive integers.") from exc

    if any(isinstance(unit, bool) or not isinstance(unit, Integral) for unit in units):
        raise TypeError("Every value in dense_units must be an integer.")
    if any(int(unit) <= 0 for unit in units):
        raise ValueError("Every value in dense_units must be greater than zero.")
    return tuple(int(unit) for unit in units)


def _default_augmentation(tf: Any, seed: int) -> Any:
    """Return an online augmentation policy based on iteration 5."""

    layers = tf.keras.layers
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=seed),
            layers.RandomRotation(
                factor=15.0 / 360.0,
                fill_mode="nearest",
                seed=seed + 1,
            ),
            layers.RandomTranslation(
                height_factor=0.1,
                width_factor=0.1,
                fill_mode="nearest",
                seed=seed + 2,
            ),
            layers.RandomZoom(
                height_factor=(-0.1, 0.1),
                width_factor=(-0.1, 0.1),
                fill_mode="nearest",
                seed=seed + 3,
            ),
            layers.RandomBrightness(
                factor=0.2,
                value_range=(0.0, 255.0),
                seed=seed + 4,
            ),
        ],
        name="online_augmentation",
    )


def build_model(
    image_size: int | Sequence[int],
    num_classes: int,
    dense_units: Sequence[int] = (256, 128),
    dropout: float = 0.3,
    weights: str | None = "imagenet",
    seed: int = 42,
    augmentation: Any = None,
) -> Any:
    """Build a MobileNetV3Small classifier.

    Parameters
    ----------
    image_size:
        Input height and width, or one integer for a square input.
    num_classes:
        Number of mutually exclusive labels.  Targets are integer class IDs.
    dense_units:
        Width of each fully connected classifier layer.  An empty sequence is
        valid and connects pooled backbone features directly to the output.
    dropout:
        Dropout rate applied after each dense layer.
    weights:
        MobileNetV3Small weights accepted by Keras, normally ``"imagenet"`` or
        ``None``.
    seed:
        Seed used by TensorFlow, augmentation, initializers, and dropout.
    augmentation:
        A custom Keras augmentation layer/model.  ``None`` creates the default
        iteration-5 online augmentation.  Pass ``False`` to disable it.

    Returns
    -------
    keras.Model
        An uncompiled model accepting ``float32`` RGB batches in ``[0, 255]``.
        MobileNetV3 preprocessing is included in the saved model.
    """

    height, width = _normalise_image_size(image_size)
    units = _normalise_dense_units(dense_units)

    if isinstance(num_classes, bool) or not isinstance(num_classes, Integral):
        raise TypeError("num_classes must be an integer.")
    if int(num_classes) < 2:
        raise ValueError("num_classes must be at least 2 for multiclass training.")
    num_classes = int(num_classes)

    if isinstance(dropout, bool) or not isinstance(dropout, Real):
        raise TypeError("dropout must be a real number in the interval [0, 1).")
    dropout = float(dropout)
    if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be a finite value in the interval [0, 1).")

    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer.")
    seed = int(seed)

    tf = _require_tensorflow()
    tf.keras.utils.set_random_seed(seed)

    inputs = tf.keras.Input(
        shape=(height, width, 3),
        dtype=tf.float32,
        name="image",
    )
    x = inputs

    augmentation_layer = _default_augmentation(tf, seed) if augmentation is None else augmentation
    if augmentation_layer is not False:
        if not callable(augmentation_layer):
            raise TypeError(
                "augmentation must be a callable Keras layer/model, None, or False."
            )
        x = augmentation_layer(x)

    try:
        backbone = tf.keras.applications.MobileNetV3Small(
            input_shape=(height, width, 3),
            include_top=False,
            weights=weights,
            include_preprocessing=True,
            name=BACKBONE_NAME,
        )
    except Exception as exc:
        raise ValueError(
            "Could not construct MobileNetV3Small. Check image_size, weights, "
            "and whether ImageNet weights are available."
        ) from exc

    backbone.trainable = False
    # Keeping training=False here also prevents BatchNormalization moving
    # statistics from changing during the later fine-tuning phase. Convolution
    # weights remain differentiable when selected by enable_fine_tuning().
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)

    for index, unit_count in enumerate(units):
        x = tf.keras.layers.Dense(
            unit_count,
            activation="relu",
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + 10 + index),
            name=f"classifier_dense_{index + 1}",
        )(x)
        if dropout > 0.0:
            x = tf.keras.layers.Dropout(
                dropout,
                seed=seed + 100 + index,
                name=f"classifier_dropout_{index + 1}",
            )(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        dtype=tf.float32,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + 1_000),
        name="probabilities",
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="campis_mobilenet_v3_small")


def compile_model(model: Any, learning_rate: float) -> Any:
    """Compile a model for sparse integer labels and return it for chaining."""

    if isinstance(learning_rate, bool) or not isinstance(learning_rate, Real):
        raise TypeError("learning_rate must be a positive real number.")
    learning_rate = float(learning_rate)
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be a finite value greater than zero.")
    if model is None or not hasattr(model, "compile"):
        raise TypeError("model must be a Keras model with a compile() method.")

    tf = _require_tensorflow()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def enable_fine_tuning(model: Any, fraction: float) -> int:
    """Unfreeze the final ``fraction`` of backbone layers.

    BatchNormalization layers always remain frozen.  Call :func:`compile_model`
    again after this function so Keras refreshes the set of trainable variables.
    A fraction of ``0`` restores a fully frozen backbone; ``1`` selects every
    non-BatchNormalization backbone layer. The returned integer is the number
    of backbone layers actually enabled.
    """

    if isinstance(fraction, bool) or not isinstance(fraction, Real):
        raise TypeError("fraction must be a real number in the interval [0, 1].")
    fraction = float(fraction)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be a finite value in the interval [0, 1].")
    if model is None or not hasattr(model, "get_layer"):
        raise TypeError("model must be a Keras model created by build_model().")

    tf = _require_tensorflow()
    try:
        backbone = model.get_layer(BACKBONE_NAME)
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            f"Model does not contain the expected backbone layer {BACKBONE_NAME!r}. "
            "Use a model created by build_model()."
        ) from exc

    backbone_layers = tuple(backbone.layers)
    if not backbone_layers:
        raise ValueError("The MobileNetV3Small backbone contains no layers to fine-tune.")

    if fraction == 0.0:
        backbone.trainable = False
        return 0

    backbone.trainable = True
    selected_count = max(1, math.ceil(len(backbone_layers) * fraction))
    first_trainable = len(backbone_layers) - selected_count

    trainable_count = 0
    for index, layer in enumerate(backbone_layers):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = index >= first_trainable
            trainable_count += int(layer.trainable)

    return trainable_count


__all__ = [
    "BACKBONE_NAME",
    "build_model",
    "compile_model",
    "enable_fine_tuning",
]
