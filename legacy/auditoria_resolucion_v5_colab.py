"""Auditoría de resolución de dataset_v5 para ejecutar en Google Colab.

Mide dimensiones en píxeles y megapíxeles. No calcula PPI físico real, pues
ese valor depende del tamaño físico de captura/impresión y suele no estar en
los JPEG. Los datos se procesan en el entorno temporal de Colab, no en el PC.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError

from google.colab import drive


DATASET_DIR = Path("/content/drive/MyDrive/CAMPIS/datasets/dataset_v5")
CLASS_NAMES = ("early_blight", "healthy", "late_blight")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PROGRESS_INTERVAL = 100


def image_paths_for_class(class_dir: Path) -> list[Path]:
    return sorted(
        path for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def describe(values: list[float]) -> str:
    data = np.asarray(values, dtype=float)
    return (
        f"mín={data.min():.2f}, p25={np.percentile(data, 25):.2f}, "
        f"mediana={np.median(data):.2f}, p75={np.percentile(data, 75):.2f}, "
        f"máx={data.max():.2f}"
    )


def audit_resolution() -> tuple[dict[str, list[float]], dict[str, Counter]]:
    class_files = {
        class_name: image_paths_for_class(DATASET_DIR / class_name)
        for class_name in CLASS_NAMES
    }
    missing = [name for name in CLASS_NAMES if not (DATASET_DIR / name).is_dir()]
    if missing:
        raise FileNotFoundError(f"No se encontraron las clases: {missing}")

    total = sum(len(paths) for paths in class_files.values())
    if total == 0:
        raise ValueError("No se encontraron imágenes compatibles.")

    megapixels = defaultdict(list)
    resolutions = defaultdict(Counter)
    invalid = []
    below_model_size = []
    processed = 0

    print(f"Analizando resolución de {total} imágenes en Drive...")
    for class_name, paths in class_files.items():
        print(f"Clase '{class_name}': {len(paths)} imágenes", flush=True)
        for path in paths:
            try:
                # PIL lee el encabezado JPEG para obtener size; no decodifica la imagen completa.
                with Image.open(path) as image:
                    width, height = image.size
                megapixels[class_name].append((width * height) / 1_000_000)
                resolutions[class_name][(width, height)] += 1
                if width < 224 or height < 224:
                    below_model_size.append(path)
            except (UnidentifiedImageError, OSError, ValueError) as error:
                invalid.append((path, str(error)))

            processed += 1
            if processed % PROGRESS_INTERVAL == 0 or processed == total:
                print(f"  Progreso: {processed}/{total} ({processed / total:.1%})", flush=True)

    print("\n=== DENSIDAD DE PÍXELES / RESOLUCIÓN ===")
    for class_name in CLASS_NAMES:
        values = megapixels[class_name]
        print(f"\n{class_name}: {len(values)} imágenes")
        if values:
            print(f"  Megapíxeles: {describe(values)}")
            print(f"  Resoluciones frecuentes: {resolutions[class_name].most_common(8)}")
    print(f"\nImágenes menores de 224×224: {len(below_model_size)}")
    print(f"Imágenes no legibles: {len(invalid)}")

    return megapixels, resolutions


def plot_megapixels(megapixels: dict[str, list[float]]) -> None:
    data = [megapixels[class_name] for class_name in CLASS_NAMES]
    plt.figure(figsize=(10, 5))
    plt.boxplot(data, tick_labels=CLASS_NAMES, showfliers=False)
    plt.ylabel("Megapíxeles por imagen")
    plt.title("Distribución de resolución por clase")
    plt.tight_layout()
    plt.show()


def main() -> None:
    drive.mount("/content/drive", force_remount=False)
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"No se encontró el dataset en: {DATASET_DIR}")
    megapixels, _ = audit_resolution()
    plot_megapixels(megapixels)


if __name__ == "__main__":
    main()
