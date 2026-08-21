"""Figuras de diagnóstico a partir de los artefactos que deja una ejecución.

Todo se dibuja leyendo archivos ya escritos —``history.csv``, ``metrics.json``,
``predictions.csv`` y el manifest— y nunca desde el modelo en memoria. Así el
reporte se puede regenerar sobre una corrida antigua, o repetir tras corregir
un gráfico, sin volver a entrenar.

Matplotlib se importa con el backend ``Agg`` dentro de cada función: importar
este módulo no debe abrir ventanas ni exigir un entorno gráfico.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


LOGGER = logging.getLogger("campis.reporting")

# Cada dominio conserva el mismo color en todas las figuras: leer una serie
# como "la cámara profesional" no debería exigir consultar la leyenda.
SOURCE_COLORS = {
    "campis": "#B4761E",
    "irish": "#2A6674",
}
_FALLBACK_COLORS = ("#3F6C51", "#6B4E7D", "#8C4A3F", "#40607F")
_CORRECT_COLOR = "#3F7D55"
_ERROR_COLOR = "#B0463C"


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _style(axis: Any) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(True, axis="y", alpha=0.25, linewidth=0.7)
    axis.set_axisbelow(True)


def _source_color(source: str, index: int = 0) -> str:
    return SOURCE_COLORS.get(
        source.casefold(), _FALLBACK_COLORS[index % len(_FALLBACK_COLORS)]
    )


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _series(rows: Sequence[Mapping[str, str]], column: str) -> list[float | None]:
    return [_number(row.get(column)) for row in rows]


def _has_values(values: Iterable[float | None]) -> bool:
    return any(value is not None for value in values)


def _plot_series(
    axis: Any,
    x_values: Sequence[float],
    y_values: Sequence[float | None],
    **kwargs: Any,
) -> None:
    """Dibuja saltando los huecos: una fase sin la métrica no inventa una línea."""

    points = [
        (x, y) for x, y in zip(x_values, y_values) if y is not None
    ]
    if not points:
        return
    axis.plot([x for x, _ in points], [y for _, y in points], **kwargs)


# --------------------------------------------------------------------------
# 1. Curvas de entrenamiento
# --------------------------------------------------------------------------


def plot_training_curves(history_path: str | Path, output_path: str | Path) -> Path:
    """Pérdida y acierto por época, con las dos fases en un eje continuo.

    Las épocas del CSV reinician en 1 al empezar el fine-tuning; aquí se
    encadenan y la frontera se marca con una línea vertical, porque el salto
    de learning rate hace que las dos fases no sean comparables entre sí.
    """

    rows = _read_csv(history_path)
    if not rows:
        raise ValueError(f"El historial está vacío: {history_path}")

    plt = _pyplot()
    from matplotlib.ticker import MaxNLocator

    steps = list(range(1, len(rows) + 1))
    phases = [row.get("phase", "") for row in rows]
    boundaries = [
        index + 1
        for index in range(1, len(phases))
        if phases[index] != phases[index - 1]
    ]

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    panels = (
        (axes[0], "loss", "val_loss", "Pérdida", "Entropía cruzada"),
        (axes[1], "accuracy", "val_accuracy", "Acierto", "Proporción correcta"),
    )
    for axis, train_column, val_column, title, ylabel in panels:
        _plot_series(
            axis, steps, _series(rows, train_column),
            color="#4A4A4A", linewidth=1.8, label="entrenamiento",
        )
        _plot_series(
            axis, steps, _series(rows, val_column),
            color="#1F4F7A", linewidth=1.8, label="validación (global)",
        )
        for index, source in enumerate(sorted(SOURCE_COLORS)):
            values = _series(rows, f"val_{source}_{train_column}")
            if _has_values(values):
                _plot_series(
                    axis, steps, values,
                    color=_source_color(source, index), linewidth=1.5,
                    linestyle="--", label=f"validación · {source}",
                )
        for boundary in boundaries:
            axis.axvline(boundary - 0.5, color="#999999", linewidth=1, linestyle=":")
            axis.text(
                boundary - 0.4,
                axis.get_ylim()[1],
                " fine-tuning",
                fontsize=8,
                color="#666666",
                va="top",
            )
        axis.set(title=title, xlabel="Época acumulada", ylabel=ylabel)
        # Las épocas son enteras: un tick en 2,5 no corresponde a nada.
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.legend(frameon=False, fontsize=8.5)
        _style(axis)

    figure.suptitle(
        "Curvas de entrenamiento — la brecha entre dominios es la señal a vigilar",
        fontsize=11,
    )
    figure.tight_layout()
    output_path = Path(output_path)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


# --------------------------------------------------------------------------
# 2. Precision / recall / F1 por clase
# --------------------------------------------------------------------------


def plot_class_metrics(metrics_path: str | Path, output_path: str | Path) -> Path:
    """Barras de precision, recall y F1 por clase, un panel por dominio."""

    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    sections: list[tuple[str, dict[str, Any]]] = [("global", metrics["global"])]
    sections.extend(sorted(metrics.get("by_source", {}).items()))

    plt = _pyplot()
    figure, axes = plt.subplots(
        1, len(sections), figsize=(5.2 * len(sections), 4.6), squeeze=False
    )

    metric_names = ("precision", "recall", "f1-score")
    metric_labels = ("Precisión", "Recall", "F1")
    shades = ("#6E8FA8", "#3F6C51", "#B4761E")

    for column, (name, section) in enumerate(sections):
        axis = axes[0][column]
        report = section["classification_report"]
        classes = [
            key
            for key in report
            if isinstance(report[key], dict) and not key.endswith("avg")
        ]
        positions = range(len(classes))
        width = 0.26
        for index, (metric, label, shade) in enumerate(
            zip(metric_names, metric_labels, shades)
        ):
            values = [float(report[name_][metric]) for name_ in classes]
            axis.bar(
                [position + (index - 1) * width for position in positions],
                values,
                width=width,
                label=label,
                color=shade,
            )
        support = [int(report[name_]["support"]) for name_ in classes]
        axis.set_xticks(list(positions))
        axis.set_xticklabels(
            [
                f"{name_.replace('_', ' ')}\nn={count:,}".replace(",", ".")
                for name_, count in zip(classes, support)
            ],
            fontsize=8.5,
        )
        axis.set_ylim(0, 1.05)
        axis.set_title(
            f"{name} · {section['samples']:,} imágenes".replace(",", "."),
            fontsize=10,
        )
        if column == 0:
            axis.set_ylabel("Puntaje")
            axis.legend(frameon=False, fontsize=8.5, loc="lower right")
        _style(axis)

    figure.suptitle("Precision, recall y F1 por clase", fontsize=11)
    figure.tight_layout()
    output_path = Path(output_path)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


# --------------------------------------------------------------------------
# 3. Distribución de confianza
# --------------------------------------------------------------------------


def plot_confidence_distribution(
    predictions_path: str | Path, output_path: str | Path
) -> Path:
    """Compara la confianza de los aciertos con la de los errores.

    Un modelo útil en campo se equivoca con poca confianza. Si los errores se
    acumulan cerca de 1,0 no existe ningún umbral de rechazo que los filtre, y
    conviene saberlo antes de desplegar.
    """

    rows = _read_csv(predictions_path)
    correct = [
        _number(row["confidence"])
        for row in rows
        if row.get("correct", "").strip().casefold() == "true"
    ]
    wrong = [
        _number(row["confidence"])
        for row in rows
        if row.get("correct", "").strip().casefold() != "true"
    ]
    correct = [value for value in correct if value is not None]
    wrong = [value for value in wrong if value is not None]

    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(8.5, 4.6))
    bins = [index / 40 for index in range(41)]
    if correct:
        axis.hist(
            correct, bins=bins, color=_CORRECT_COLOR, alpha=0.75,
            label=f"aciertos ({len(correct):,})".replace(",", "."),
        )
    if wrong:
        axis.hist(
            wrong, bins=bins, color=_ERROR_COLOR, alpha=0.8,
            label=f"errores ({len(wrong):,})".replace(",", "."),
        )
    axis.set(
        title="Confianza de la predicción, separando aciertos de errores",
        xlabel="Confianza de la clase predicha",
        ylabel="Imágenes",
        yscale="log",
    )
    axis.legend(frameon=False, fontsize=9)
    _style(axis)
    figure.tight_layout()
    output_path = Path(output_path)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


# --------------------------------------------------------------------------
# 4 y 5. Grillas de imágenes
# --------------------------------------------------------------------------


def _load_grid_image(
    relative_path: str,
    project_root: str | Path,
    image_size: Sequence[int],
    crop_to_square: bool,
) -> Any:
    from .inference import load_image

    path = Path(relative_path)
    if not path.is_absolute():
        path = Path(project_root) / path
    array = load_image(path, image_size, crop_to_square=crop_to_square)
    return array.astype("uint8")


def _image_grid(
    rows: Sequence[Mapping[str, str]],
    project_root: str | Path,
    image_size: Sequence[int],
    output_path: str | Path,
    *,
    crop_to_square: bool,
    title: str,
    columns: int = 6,
) -> Path | None:
    if not rows:
        LOGGER.warning("Sin imágenes para la grilla: %s", output_path)
        return None

    plt = _pyplot()
    count = len(rows)
    grid_rows = math.ceil(count / columns)
    figure, axes = plt.subplots(
        grid_rows, columns, figsize=(2.05 * columns, 2.9 * grid_rows), squeeze=False
    )

    for index in range(grid_rows * columns):
        axis = axes[index // columns][index % columns]
        axis.set_xticks([])
        axis.set_yticks([])
        if index >= count:
            axis.axis("off")
            continue
        row = rows[index]
        try:
            axis.imshow(
                _load_grid_image(
                    row["relative_path"], project_root, image_size, crop_to_square
                )
            )
        except Exception as error:  # una imagen ilegible no invalida la figura
            LOGGER.warning("No se pudo dibujar %s: %s", row["relative_path"], error)
            axis.axis("off")
            continue
        is_correct = row.get("correct", "").strip().casefold() == "true"
        color = _CORRECT_COLOR if is_correct else _ERROR_COLOR
        for spine in axis.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.2)
        confidence = _number(row["confidence"]) or 0.0
        caption = (
            f"{row['predicted_class'].replace('_', ' ')} · {confidence:.0%}"
            if is_correct
            else f"{row['predicted_class'].replace('_', ' ')} · {confidence:.0%}\n"
            f"real: {row['true_class'].replace('_', ' ')}"
        )
        axis.set_xlabel(caption, fontsize=7.5, color=color, labelpad=3)
        axis.set_title(row.get("source", ""), fontsize=7, color="#666666", pad=2)

    figure.suptitle(title, fontsize=11)
    # h_pad separa el pie de una fila del titulo de la siguiente: sin holgura
    # extra, "real: ..." se encima con el nombre de la fuente de abajo.
    figure.tight_layout(h_pad=2.6)
    output_path = Path(output_path)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_sample_predictions(
    predictions_path: str | Path,
    project_root: str | Path,
    image_size: Sequence[int],
    output_path: str | Path,
    *,
    crop_to_square: bool = True,
    seed: int = 42,
    per_source: int = 6,
) -> Path | None:
    """Muestra imágenes de test tal como las ve el modelo, con su predicción.

    Se dibuja el tensor ya preprocesado —recortado y redimensionado—, no el
    archivo original: así la figura también sirve para verificar que el recorte
    cuadrado no está cortando la hoja.
    """

    rows = _read_csv(predictions_path)
    by_source: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        by_source.setdefault(row.get("source", ""), []).append(row)

    selected: list[Mapping[str, str]] = []
    for source in sorted(by_source):
        candidates = sorted(by_source[source], key=lambda row: row["relative_path"])
        rng = random.Random(f"{seed}:{source}")
        selected.extend(rng.sample(candidates, min(per_source, len(candidates))))

    return _image_grid(
        selected,
        project_root,
        image_size,
        output_path,
        crop_to_square=crop_to_square,
        title="Muestras del test · borde verde acierta, rojo se equivoca",
    )


def plot_top_errors(
    predictions_path: str | Path,
    project_root: str | Path,
    image_size: Sequence[int],
    output_path: str | Path,
    *,
    crop_to_square: bool = True,
    limit: int = 12,
) -> Path | None:
    """Los errores más confiados: el diagnóstico más informativo del test.

    Equivocarse con 99 % de confianza suele delatar etiquetas dudosas o un
    patrón que el modelo aprendió mal, no ruido estadístico.
    """

    rows = [
        row
        for row in _read_csv(predictions_path)
        if row.get("correct", "").strip().casefold() != "true"
    ]
    rows.sort(key=lambda row: -(_number(row["confidence"]) or 0.0))
    return _image_grid(
        rows[:limit],
        project_root,
        image_size,
        output_path,
        crop_to_square=crop_to_square,
        title="Errores más confiados del test",
    )


# --------------------------------------------------------------------------
# 6. Composición del corpus
# --------------------------------------------------------------------------


def plot_dataset_distribution(
    manifest_path: str | Path, output_path: str | Path
) -> Path:
    """Cuántas imágenes aporta cada fuente y clase, y cómo se repartieron."""

    rows = _read_csv(manifest_path)
    if not rows:
        raise ValueError(f"El manifest está vacío: {manifest_path}")

    splits = ("train", "val", "test")
    counts: dict[tuple[str, str], dict[str, int]] = {}
    for row in rows:
        key = (row["source"], row["class_name"])
        bucket = counts.setdefault(key, {split: 0 for split in splits})
        if row["split"] in bucket:
            bucket[row["split"]] += 1

    keys = sorted(counts)
    labels = [
        f"{source}\n{class_name.replace('_', ' ')}" for source, class_name in keys
    ]
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(1.35 * len(keys) + 3, 4.8))

    shades = {"train": "#3F6C51", "val": "#6E8FA8", "test": "#B4761E"}
    bottoms = [0.0] * len(keys)
    for split in splits:
        values = [counts[key][split] for key in keys]
        axis.bar(labels, values, bottom=bottoms, label=split, color=shades[split])
        bottoms = [base + value for base, value in zip(bottoms, values)]

    for index, total in enumerate(bottoms):
        axis.text(
            index, total, f"{int(total):,}".replace(",", "."),
            ha="center", va="bottom", fontsize=8.5,
        )

    axis.set(
        title="Composición del corpus por fuente, clase y partición",
        ylabel="Imágenes",
    )
    axis.tick_params(axis="x", labelsize=8.5)
    axis.legend(frameon=False, fontsize=9)
    _style(axis)
    figure.tight_layout()
    output_path = Path(output_path)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


# --------------------------------------------------------------------------
# Orquestación
# --------------------------------------------------------------------------


def build_report(
    run_dir: str | Path,
    project_root: str | Path,
    image_size: Sequence[int],
    *,
    crop_to_square: bool = True,
    seed: int = 42,
) -> list[Path]:
    """Genera todas las figuras que los artefactos presentes permitan.

    Una figura cuya entrada falta se omite con una advertencia en lugar de
    interrumpir el reporte: una corrida antigua sin métricas por dominio debe
    seguir produciendo todo lo demás.
    """

    run_dir = Path(run_dir)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir = run_dir / "evaluation"

    tasks: list[tuple[str, Path, Any]] = [
        (
            "curvas de entrenamiento",
            run_dir / "history.csv",
            lambda source: plot_training_curves(
                source, figures_dir / "training_curves.png"
            ),
        ),
        (
            "métricas por clase",
            evaluation_dir / "metrics.json",
            lambda source: plot_class_metrics(
                source, figures_dir / "class_metrics.png"
            ),
        ),
        (
            "distribución de confianza",
            evaluation_dir / "predictions.csv",
            lambda source: plot_confidence_distribution(
                source, figures_dir / "confidence_distribution.png"
            ),
        ),
        (
            "muestras del test",
            evaluation_dir / "predictions.csv",
            lambda source: plot_sample_predictions(
                source,
                project_root,
                image_size,
                figures_dir / "sample_predictions.png",
                crop_to_square=crop_to_square,
                seed=seed,
            ),
        ),
        (
            "errores más confiados",
            evaluation_dir / "predictions.csv",
            lambda source: plot_top_errors(
                source,
                project_root,
                image_size,
                figures_dir / "top_errors.png",
                crop_to_square=crop_to_square,
            ),
        ),
        (
            "composición del corpus",
            run_dir / "manifest.csv",
            lambda source: plot_dataset_distribution(
                source, figures_dir / "dataset_distribution.png"
            ),
        ),
    ]

    created: list[Path] = []
    for label, source, draw in tasks:
        if not source.is_file():
            LOGGER.warning("Se omite '%s': falta %s", label, source)
            continue
        result = draw(source)
        if result is not None:
            created.append(Path(result))
            LOGGER.info("Figura generada: %s", result)

    return created


__all__ = [
    "build_report",
    "plot_class_metrics",
    "plot_confidence_distribution",
    "plot_dataset_distribution",
    "plot_sample_predictions",
    "plot_top_errors",
    "plot_training_curves",
]
