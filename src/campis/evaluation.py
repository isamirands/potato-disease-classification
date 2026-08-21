"""Métricas y artefactos de evaluación globales y por dominio."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _json_ready(value: Any) -> Any:
    """Convierte escalares/arrays de NumPy en valores serializables."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _metrics_for_subset(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, Any]:
    from sklearn.metrics import classification_report, confusion_matrix

    labels = list(range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "samples": int(len(y_true)),
        "accuracy": float(report.get("accuracy", 0.0)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": matrix,
    }


def compute_metrics(
    y_true: Sequence[int] | np.ndarray,
    probabilities: np.ndarray,
    sources: Sequence[str],
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Calcula las mismas métricas para el conjunto completo y cada fuente."""

    truth = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    source_array = np.asarray(sources, dtype=str)

    if len(class_names) < 2 or any(
        not isinstance(name, str) or not name.strip() for name in class_names
    ):
        raise ValueError("class_names debe contener al menos dos nombres no vacíos.")
    if len(set(class_names)) != len(class_names):
        raise ValueError("class_names no puede contener duplicados.")
    if truth.ndim != 1 or source_array.ndim != 1:
        raise ValueError("Etiquetas y fuentes deben ser vectores de una dimensión.")

    if probabilities.ndim != 2 or probabilities.shape[1] != len(class_names):
        raise ValueError(
            "Las probabilidades deben tener forma "
            f"(n, {len(class_names)}); se recibió {probabilities.shape}."
        )
    if len(truth) != len(probabilities) or len(truth) != len(source_array):
        raise ValueError("Etiquetas, probabilidades y fuentes deben tener igual longitud.")
    if len(truth) == 0:
        raise ValueError("No hay muestras para evaluar.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Las probabilidades contienen NaN o infinito.")
    if np.any(truth < 0) or np.any(truth >= len(class_names)):
        raise ValueError("Las etiquetas contienen índices fuera del rango de clases.")

    predictions = probabilities.argmax(axis=1)
    result: dict[str, Any] = {
        "global": _metrics_for_subset(truth, predictions, class_names),
        "by_source": {},
    }
    for source in sorted(set(source_array.tolist())):
        mask = source_array == source
        result["by_source"][source] = _metrics_for_subset(
            truth[mask], predictions[mask], class_names
        )
    return result


def evaluate_model(
    model: Any,
    dataset: Any,
    records: Sequence[Any],
    class_names: Sequence[str],
    *,
    verbose: int = 1,
) -> tuple[dict[str, Any], np.ndarray]:
    """Predice un dataset ordenado y devuelve métricas alineadas al manifest."""

    if not records:
        raise ValueError("No hay registros para evaluar.")

    probabilities = np.asarray(model.predict(dataset, verbose=verbose))
    if len(probabilities) != len(records):
        raise RuntimeError(
            "El número de predicciones no coincide con el manifest: "
            f"{len(probabilities)} != {len(records)}."
        )

    labels = [int(record.class_index) for record in records]
    sources = [str(record.source) for record in records]
    metrics = compute_metrics(labels, probabilities, sources, class_names)
    return metrics, probabilities


def _save_confusion_plot(
    matrix: np.ndarray,
    class_names: Sequence[str],
    title: str,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    ticks = np.arange(len(class_names))
    axis.set(
        xticks=ticks,
        yticks=ticks,
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicción",
        ylabel="Etiqueta real",
        title=title,
    )
    axis.tick_params(axis="x", labelrotation=25)
    threshold = float(matrix.max()) / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(int(matrix[row, column])),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_evaluation_artifacts(
    metrics: dict[str, Any],
    probabilities: np.ndarray,
    records: Sequence[Any],
    class_names: Sequence[str],
    output_dir: str | Path,
) -> None:
    """Guarda JSON, matrices, predicciones y errores sin mostrar ventanas."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(class_names):
        raise ValueError(
            "Las probabilidades deben tener forma "
            f"(n, {len(class_names)}); se recibió {probabilities.shape}."
        )
    if len(probabilities) != len(records):
        raise ValueError(
            "El número de probabilidades no coincide con los registros: "
            f"{len(probabilities)} != {len(records)}."
        )

    serializable = _json_ready(metrics)
    metrics_path = output_dir / "metrics.json"
    temporary_path = metrics_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_path.replace(metrics_path)

    all_sections = [("global", metrics["global"])] + [
        (f"source_{index:02d}_{source}", section)
        for index, (source, section) in enumerate(
            sorted(metrics["by_source"].items()), start=1
        )
    ]
    for name, section in all_sections:
        safe_name = "".join(char if char.isalnum() else "_" for char in name.lower())
        matrix = np.asarray(section["confusion_matrix"], dtype=np.int64)
        _save_confusion_plot(
            matrix,
            class_names,
            f"Matriz de confusión — {name}",
            output_dir / f"confusion_{safe_name}.png",
        )

    predictions = probabilities.argmax(axis=1)
    predictions_path = output_dir / "predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "relative_path",
            "source",
            "camera_type",
            "true_class",
            "predicted_class",
            "confidence",
            "correct",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record, predicted, probability in zip(records, predictions, probabilities):
            true_index = int(record.class_index)
            predicted_index = int(predicted)
            writer.writerow(
                {
                    "relative_path": record.relative_path,
                    "source": record.source,
                    "camera_type": record.camera_type,
                    "true_class": class_names[true_index],
                    "predicted_class": class_names[predicted_index],
                    "confidence": f"{float(probability[predicted_index]):.8f}",
                    "correct": true_index == predicted_index,
                }
            )
