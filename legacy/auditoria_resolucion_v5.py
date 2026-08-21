"""Auditoría de resolución y metadatos EXIF de dataset_v5.

Clasifica de manera inferida imágenes tomadas con celular o cámara dedicada a
partir de Make/Model EXIF. Los valores DPI/PPI del JPEG se informan solo como
metadatos: no son una medida fiable de calidad ni de origen de la imagen.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError


DATASET_DIR = Path(r"G:\Mi unidad\CAMPIS\datasets\dataset_v5")
OUTPUT_DIR = Path(__file__).resolve().parent / "graficos_densidad_v5"
CLASS_NAMES = ("early_blight", "healthy", "late_blight")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PROGRESS_INTERVAL = 100

# Indicadores habituales en metadatos EXIF de smartphones. Se revisan antes
# que las cámaras dedicadas porque marcas como Sony también fabrican teléfonos.
PHONE_HINTS = (
    "iphone", "ipad", "samsung", "galaxy", "pixel", "google", "xiaomi",
    "redmi", "poco", "huawei", "honor", "oneplus", "oppo", "realme",
    "motorola", "moto", "vivo", "asus", "nothing", "xperia",
)
DEDICATED_CAMERA_HINTS = (
    "canon", "nikon", "fujifilm", "leica", "olympus", "panasonic",
    "lumix", "pentax", "hasselblad", "gopro", "eos", "ilce", "alpha",
    "dsc-", "coolpix", "powershot",
)


def clean_exif_value(value: object | None) -> str:
    if value is None:
        return ""
    return str(value).replace("\x00", "").strip()


def classify_source(make: str, model: str) -> str:
    description = f"{make} {model}".lower()
    if any(hint in description for hint in PHONE_HINTS):
        return "Celular (EXIF)"
    if any(hint in description for hint in DEDICATED_CAMERA_HINTS):
        return "Cámara dedicada (EXIF)"
    return "Sin metadatos / no clasificable"


def image_paths(class_dir: Path) -> list[Path]:
    return sorted(
        path for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def format_dpi(dpi: object | None) -> str:
    if not dpi:
        return ""
    try:
        horizontal, vertical = dpi
        return f"{float(horizontal):.2f} × {float(vertical):.2f}"
    except (TypeError, ValueError):
        return str(dpi)


def percent(value: int, total: int) -> str:
    return f"{100 * value / total:.1f}%" if total else "0.0%"


def audit() -> list[dict[str, object]]:
    missing = [name for name in CLASS_NAMES if not (DATASET_DIR / name).is_dir()]
    if missing:
        raise FileNotFoundError(f"No se encontraron las carpetas de clase: {missing}")

    paths_by_class = {name: image_paths(DATASET_DIR / name) for name in CLASS_NAMES}
    total = sum(len(paths) for paths in paths_by_class.values())
    if not total:
        raise ValueError("No se encontraron imágenes compatibles.")

    records: list[dict[str, object]] = []
    unreadable: list[tuple[Path, str]] = []
    processed = 0
    print(f"Analizando encabezados y EXIF de {total} imágenes...")

    for class_name, paths in paths_by_class.items():
        print(f"Clase '{class_name}': {len(paths)} imágenes", flush=True)
        for path in paths:
            try:
                # PIL obtiene tamaño y EXIF desde el encabezado; no decodifica los píxeles completos.
                with Image.open(path) as image:
                    width, height = image.size
                    exif = image.getexif()
                    make = clean_exif_value(exif.get(271))
                    model = clean_exif_value(exif.get(272))
                    dpi = format_dpi(image.info.get("dpi"))

                records.append({
                    "class_name": class_name,
                    "path": str(path.relative_to(DATASET_DIR)),
                    "width": width,
                    "height": height,
                    "megapixels": round((width * height) / 1_000_000, 4),
                    "make": make,
                    "model": model,
                    "dpi_reported": dpi,
                    "source_type": classify_source(make, model),
                })
            except (UnidentifiedImageError, OSError, ValueError) as error:
                unreadable.append((path, str(error)))

            processed += 1
            if processed % PROGRESS_INTERVAL == 0 or processed == total:
                print(f"  Progreso: {processed}/{total} ({percent(processed, total)})", flush=True)

    print(f"Imágenes procesadas correctamente: {len(records)}")
    print(f"Imágenes no legibles: {len(unreadable)}")
    return records


def save_csv(records: list[dict[str, object]]) -> Path:
    csv_path = OUTPUT_DIR / "detalle_resolucion_y_origen.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    return csv_path


def save_source_count_chart(records: list[dict[str, object]]) -> None:
    counts = Counter(record["source_type"] for record in records)
    labels, values = zip(*counts.most_common())
    figure, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(labels, values, color=("#4C78A8", "#F58518", "#9E9E9E"))
    axis.bar_label(bars, labels=[f"{value} ({percent(value, len(records))})" for value in values])
    axis.set_title("Origen inferido mediante EXIF")
    axis.set_ylabel("Número de imágenes")
    axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "01_origen_inferido_exif.jpg", dpi=180)
    plt.close(figure)


def save_megapixel_chart(records: list[dict[str, object]]) -> None:
    groups = ["Celular (EXIF)", "Cámara dedicada (EXIF)", "Sin metadatos / no clasificable"]
    data = [[float(record["megapixels"]) for record in records if record["source_type"] == group] for group in groups]
    available = [(group, values) for group, values in zip(groups, data) if values]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.boxplot([values for _, values in available], tick_labels=[group for group, _ in available], showfliers=False)
    axis.set_title("Megapíxeles por origen inferido")
    axis.set_ylabel("Megapíxeles por imagen")
    axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "02_megapixeles_por_origen.jpg", dpi=180)
    plt.close(figure)


def save_resolution_scatter(records: list[dict[str, object]]) -> None:
    colors = {
        "Celular (EXIF)": "#4C78A8",
        "Cámara dedicada (EXIF)": "#F58518",
        "Sin metadatos / no clasificable": "#9E9E9E",
    }
    figure, axis = plt.subplots(figsize=(8, 6))
    for source_type, color in colors.items():
        group = [record for record in records if record["source_type"] == source_type]
        if group:
            axis.scatter(
                [record["width"] for record in group],
                [record["height"] for record in group],
                label=source_type,
                color=color,
                alpha=0.55,
                s=18,
            )
    axis.set_title("Resolución: ancho frente a alto")
    axis.set_xlabel("Ancho (px)")
    axis.set_ylabel("Alto (px)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "03_resolucion_por_origen.jpg", dpi=180)
    plt.close(figure)


def save_model_chart(records: list[dict[str, object]]) -> None:
    models = Counter(
        f"{record['make']} {record['model']}".strip()
        for record in records
        if record["make"] or record["model"]
    )
    if not models:
        return
    labels, values = zip(*models.most_common(12))
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.barh(labels[::-1], values[::-1], color="#54A24B")
    axis.set_title("Modelos detectados en EXIF")
    axis.set_xlabel("Número de imágenes")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "04_modelos_exif.jpg", dpi=180)
    plt.close(figure)


def write_report(records: list[dict[str, object]]) -> Path:
    counts = Counter(record["source_type"] for record in records)
    megapixels = defaultdict(list)
    for record in records:
        megapixels[record["source_type"]].append(float(record["megapixels"]))

    report_path = OUTPUT_DIR / "informe_resolucion_y_origen.txt"
    with report_path.open("w", encoding="utf-8") as file:
        file.write("ANÁLISIS DE RESOLUCIÓN Y ORIGEN INFERIDO\n")
        file.write("=" * 44 + "\n\n")
        file.write(f"Imágenes analizadas: {len(records)}\n\n")
        for source_type, count in counts.most_common():
            values = np.asarray(megapixels[source_type])
            file.write(f"{source_type}: {count} ({percent(count, len(records))})\n")
            file.write(
                f"  Megapíxeles: mediana {np.median(values):.2f}; "
                f"rango {values.min():.2f}–{values.max():.2f}\n"
            )
        file.write("\nNota: la clasificación depende de Make/Model EXIF. "
                   "Sin EXIF no es posible asegurar el origen solo por resolución.\n")
        file.write("El valor DPI/PPI guardado en JPEG no se usa para clasificar, "
                   "pues normalmente no representa densidad física real.\n")
    return report_path


def main() -> None:
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"No se encontró el dataset en: {DATASET_DIR}")
    OUTPUT_DIR.mkdir(exist_ok=True)
    records = audit()
    if not records:
        raise ValueError("No se pudo analizar ninguna imagen.")
    csv_path = save_csv(records)
    save_source_count_chart(records)
    save_megapixel_chart(records)
    save_resolution_scatter(records)
    save_model_chart(records)
    report_path = write_report(records)
    print(f"\nGráficos y reporte guardados en: {OUTPUT_DIR}")
    print(f"Detalle por imagen: {csv_path.name}")
    print(f"Informe: {report_path.name}")


if __name__ == "__main__":
    main()
