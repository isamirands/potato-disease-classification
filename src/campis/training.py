"""Orquestación de entrenamiento en dos fases y persistencia del historial."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


def compute_class_weights(records: Sequence[Any]) -> dict[int, float]:
    """Calcula pesos balanceados conservando los índices reales de clase."""

    if not records:
        raise ValueError("No hay registros de entrenamiento.")

    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight

    labels = np.asarray([int(record.class_index) for record in records], dtype=np.int64)
    unique_labels = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced", classes=unique_labels, y=labels
    )
    return {
        int(label): float(weight)
        for label, weight in zip(unique_labels.tolist(), weights.tolist())
    }


def build_domain_validation_callback(
    dataset: Any, sources: Sequence[str], labels: Sequence[int]
) -> Any:
    """Evalúa la validación una vez por época y reporta cada dominio aparte.

    Validación conserva la distribución natural del corpus, donde una fuente
    puede representar el 96 % de las imágenes. Si el entrenamiento está
    balanceado por dominio pero ``val_loss`` no lo está, el checkpoint elegido
    y el early stopping quedan gobernados por la fuente mayoritaria. Este
    callback publica además ``val_balanced_loss``: el promedio simple entre
    dominios, que trata a cada cámara como igualmente importante.

    Sustituye a ``validation_data`` en ``fit`` en lugar de sumarse a él, así
    que la validación sigue costando una sola pasada por época.
    """

    import numpy as np
    import tensorflow as tf

    source_array = np.asarray(list(sources), dtype=str)
    truth = np.asarray(list(labels), dtype=np.int64)
    if len(truth) != len(source_array):
        raise ValueError("sources y labels deben tener la misma longitud.")
    unique_sources = sorted(set(source_array.tolist()))

    class DomainValidation(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
            if logs is None:
                return
            probabilities = np.asarray(
                self.model.predict(dataset, verbose=0), dtype=np.float64
            )
            if len(probabilities) != len(source_array):
                raise RuntimeError(
                    "La validación por dominio no coincide con el manifest: "
                    f"{len(probabilities)} != {len(source_array)}."
                )
            confidences = probabilities[np.arange(len(truth)), truth]
            losses = -np.log(np.clip(confidences, 1e-7, 1.0))
            correct = probabilities.argmax(axis=1) == truth

            logs["val_loss"] = float(losses.mean())
            logs["val_accuracy"] = float(correct.mean())
            per_source_losses = []
            per_source_accuracies = []
            for source in unique_sources:
                mask = source_array == source
                source_loss = float(losses[mask].mean())
                source_accuracy = float(correct[mask].mean())
                logs[f"val_{source}_loss"] = source_loss
                logs[f"val_{source}_accuracy"] = source_accuracy
                per_source_losses.append(source_loss)
                per_source_accuracies.append(source_accuracy)
            logs["val_balanced_loss"] = float(sum(per_source_losses) / len(per_source_losses))
            logs["val_balanced_accuracy"] = float(
                sum(per_source_accuracies) / len(per_source_accuracies)
            )

    return DomainValidation()


def _callbacks(
    output_dir: Path, phase: str, patience: int, monitor: str = "val_loss"
) -> list[Any]:
    import tensorflow as tf

    checkpoint_path = output_dir / f"best_{phase}.keras"
    # Las metricas por dominio son nombres propios: Keras no puede inferir si
    # se maximizan o minimizan, asi que la direccion va explicita.
    mode = "max" if "accuracy" in monitor else "min"
    return [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(output_dir / f"history_{phase}.csv"),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode=mode,
            factor=0.5,
            patience=max(1, patience // 2),
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


def _best_validation_loss(history: Any, monitor: str = "val_loss") -> float:
    """Mejor valor del monitor; las metricas de acierto se comparan invertidas."""

    values = [float(value) for value in history.history.get(monitor, [])]
    if "accuracy" in monitor:
        return -max(values, default=-math.inf)
    return min(values, default=math.inf)


def train_two_phases(
    model: Any,
    train_dataset: Any,
    validation_dataset: Any,
    output_dir: str | Path,
    *,
    head_epochs: int,
    fine_tune_epochs: int,
    head_learning_rate: float,
    fine_tune_learning_rate: float,
    fine_tune_fraction: float,
    patience: int,
    class_weights: Mapping[int, float] | None = None,
    verbose: int = 1,
    validation_callback: Any = None,
    monitor: str = "val_loss",
    steps_per_epoch: int | None = None,
) -> tuple[Any, dict[str, dict[str, list[float]]], str]:
    """Entrena el clasificador y, opcionalmente, el último tramo del backbone.

    El mejor checkpoint de la fase inicial se conserva. Si el fine-tuning no
    mejora ``monitor``, el modelo final vuelve automáticamente a ese checkpoint.

    Con ``validation_callback`` la validación la calcula ese callback en una
    sola pasada y ``fit`` no recibe ``validation_data``; el callback se ubica
    antes que los demás para que checkpoint, early stopping y scheduler ya vean
    sus métricas al cerrar la época.
    """

    if head_epochs <= 0:
        raise ValueError("head_epochs debe ser mayor que cero.")
    if fine_tune_epochs < 0:
        raise ValueError("fine_tune_epochs no puede ser negativo.")

    import tensorflow as tf

    from .modeling import compile_model, enable_fine_tuning

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def phase_callbacks(phase: str) -> list[Any]:
        callbacks = _callbacks(output_dir, phase, patience, monitor)
        return [validation_callback, *callbacks] if validation_callback else callbacks

    fit_validation = None if validation_callback else validation_dataset

    compile_model(model, head_learning_rate)
    head_history = model.fit(
        train_dataset,
        validation_data=fit_validation,
        epochs=head_epochs,
        steps_per_epoch=steps_per_epoch,
        class_weight=dict(class_weights) if class_weights else None,
        callbacks=phase_callbacks("head"),
        shuffle=False,
        verbose=verbose,
    )
    histories: dict[str, dict[str, list[float]]] = {
        "head": {
            name: [float(value) for value in values]
            for name, values in head_history.history.items()
        }
    }
    best_phase = "head"
    best_loss = _best_validation_loss(head_history, monitor)

    if fine_tune_epochs > 0 and fine_tune_fraction > 0:
        enable_fine_tuning(model, fine_tune_fraction)
        backbone = model.get_layer("mobilenet_v3_small_backbone")
        trainable_layers = sum(1 for layer in backbone.layers if layer.trainable)
        if trainable_layers == 0:
            raise RuntimeError("No se habilitó ninguna capa para fine-tuning.")
        compile_model(model, fine_tune_learning_rate)
        fine_history = model.fit(
            train_dataset,
            validation_data=fit_validation,
            epochs=fine_tune_epochs,
            steps_per_epoch=steps_per_epoch,
            class_weight=dict(class_weights) if class_weights else None,
            callbacks=phase_callbacks("fine_tune"),
            shuffle=False,
            verbose=verbose,
        )
        histories["fine_tune"] = {
            name: [float(value) for value in values]
            for name, values in fine_history.history.items()
        }
        fine_loss = _best_validation_loss(fine_history, monitor)
        if fine_loss < best_loss:
            best_phase = "fine_tune"
            best_loss = fine_loss

    best_checkpoint = output_dir / f"best_{best_phase}.keras"
    if not best_checkpoint.is_file():
        raise RuntimeError(f"No se creó el checkpoint esperado: {best_checkpoint}")
    model = tf.keras.models.load_model(best_checkpoint)
    model.save(output_dir / "model.keras")

    _save_combined_history(histories, output_dir / "history.csv")
    return model, histories, best_phase


def _save_combined_history(
    histories: Mapping[str, Mapping[str, Sequence[float]]], output_path: Path
) -> None:
    metric_names = sorted(
        {
            metric
            for history in histories.values()
            for metric in history.keys()
        }
    )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["phase", "epoch", *metric_names]
        )
        writer.writeheader()
        for phase, history in histories.items():
            length = max((len(values) for values in history.values()), default=0)
            for epoch in range(length):
                row: dict[str, Any] = {"phase": phase, "epoch": epoch + 1}
                for metric in metric_names:
                    values = history.get(metric, [])
                    row[metric] = values[epoch] if epoch < len(values) else ""
                writer.writerow(row)
