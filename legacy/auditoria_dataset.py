"""Auditoría de solo lectura para el dataset de imágenes en Google Drive.

No entrena modelos, no crea splits y no modifica archivos. Google Drive para
escritorio puede transmitir los archivos a su caché temporal para poder leerlos,
pero este script no los copia ni los marca como disponibles sin conexión.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError


DATASET_DIR = Path(r"G:\Mi unidad\CAMPIS\datasets\dataset_v4")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ("train_augmented","test","val","train")
SAMPLES_PER_CLASS = 5
# True evita abrir imágenes y reduce al mínimo las descargas/caché de Drive.
# En este modo no se valida integridad, formato, dimensiones ni color.
METADATA_ONLY = True
# Activarlo lee cada archivo completo y puede tardar más si Drive está en modo streaming.
CHECK_EXACT_DUPLICATES = False
PROGRESS_INTERVAL = 100


def get_class_dirs(dataset_dir: Path) -> list[Path]:
    """Obtiene únicamente las tres clases solicitadas."""
    missing_classes = [name for name in CLASS_NAMES if not (dataset_dir / name).is_dir()]
    if missing_classes:
        raise FileNotFoundError(
            "No se encontraron estas carpetas de clases: "
            f"{missing_classes}"
        )
    return [dataset_dir / name for name in CLASS_NAMES]


def format_size(size_in_bytes: int) -> str:
    """Convierte bytes a una unidad legible."""
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(size_in_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size_in_bytes} B"


def audit_dataset(dataset_dir: Path, class_dirs: list[Path]) -> dict[str, list[Path]]:
    valid_files: dict[str, list[Path]] = defaultdict(list)
    counts = Counter()
    image_sizes = Counter()
    formats = Counter()
    modes = Counter()
    dimensions = Counter()
    invalid_files: list[tuple[Path, str]] = []
    ignored_files: list[Path] = []
    too_small: list[Path] = []
    hashes: dict[str, list[Path]] = defaultdict(list)
    # rglob permite auditar imágenes ubicadas en cualquier subcarpeta de cada
    # carpeta principal (por ejemplo train/early_blight/*.jpg).
    files_by_class = {
        class_dir.name: sorted(path for path in class_dir.rglob("*") if path.is_file())
        for class_dir in class_dirs
    }
    total_files = sum(len(files) for files in files_by_class.values())
    processed_files = 0

    print(f"\nIniciando revisión de {total_files} archivos...")

    for class_dir in class_dirs:
        class_files = files_by_class[class_dir.name]
        print(f"Revisando clase '{class_dir.name}' ({len(class_files)} archivos)...", flush=True)
        for path in class_files:
            processed_files += 1
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                ignored_files.append(path)
            else:
                try:
                    image_sizes[class_dir.name] += path.stat().st_size
                    if METADATA_ONLY:
                        valid_files[class_dir.name].append(path)
                        counts[class_dir.name] += 1
                    else:
                        with Image.open(path) as image:
                            image.verify()
                        with Image.open(path) as image:
                            width, height = image.size
                            formats[(image.format or path.suffix).upper()] += 1
                            modes[image.mode] += 1
                            dimensions[(width, height)] += 1
                            if width < 224 or height < 224:
                                too_small.append(path)
                        valid_files[class_dir.name].append(path)
                        counts[class_dir.name] += 1

                    if CHECK_EXACT_DUPLICATES and not METADATA_ONLY:
                        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                        hashes[file_hash].append(path)
                except (UnidentifiedImageError, OSError, ValueError) as error:
                    invalid_files.append((path, str(error)))

            if processed_files % PROGRESS_INTERVAL == 0 or processed_files == total_files:
                percentage = 100 * processed_files / total_files if total_files else 100
                print(
                    f"  Progreso: {processed_files}/{total_files} ({percentage:.1f}%)",
                    flush=True,
                )

    total = sum(counts.values())
    if total == 0:
        raise ValueError(
            "No se encontraron imágenes compatibles. Extensiones permitidas: "
            f"{sorted(SUPPORTED_EXTENSIONS)}"
        )

    duplicate_groups = [group for group in hashes.values() if len(group) > 1]
    class_counts = {class_dir.name: counts[class_dir.name] for class_dir in class_dirs}
    class_image_sizes = {class_dir.name: image_sizes[class_dir.name] for class_dir in class_dirs}
    smallest = min(class_counts.values())
    largest = max(class_counts.values())

    print(f"\n=== AUDITORÍA DE {dataset_dir.name.upper()} ===")
    print(f"Ruta: {dataset_dir}")
    count_label = "Total de archivos de imagen" if METADATA_ONLY else "Total de imágenes válidas"
    print(f"{count_label}: {total}")
    for class_name, count in class_counts.items():
        print(
            f"  {class_name}: {count} ({count / total:.2%}) | "
            f"Peso de imágenes: {format_size(class_image_sizes[class_name])}"
        )
    print(f"Peso total de imágenes: {format_size(sum(class_image_sizes.values()))}")
    print(f"Relación clase mayor/menor: {largest / smallest:.2f}" if smallest else "Hay una clase vacía.")
    print(f"Archivos ignorados por extensión: {len(ignored_files)}")
    if METADATA_ONLY:
        print("Modo: solo metadatos; no se abrieron imágenes ni se mostraron muestras.")
        print("Integridad, tamaños en píxeles, formatos y color: no analizados.")
    else:
        print(f"Imágenes ilegibles: {len(invalid_files)}")
        print(f"Imágenes menores de 224 px: {len(too_small)}")
        print("Formatos:", dict(formats))
        print("Modos de color:", dict(modes))
        print("Dimensiones más frecuentes:", dimensions.most_common(10))
    if CHECK_EXACT_DUPLICATES and not METADATA_ONLY:
        print(f"Grupos de duplicados exactos: {len(duplicate_groups)}")
    else:
        print("Duplicados exactos: no analizados en modo solo metadatos.")

    if invalid_files and not METADATA_ONLY:
        print("\nPrimeros archivos ilegibles:")
        for path, error in invalid_files[:20]:
            print(f"  {path.relative_to(dataset_dir)}: {error}")
    if too_small and not METADATA_ONLY:
        print("\nPrimeras imágenes menores de 224 px:")
        for path in too_small[:20]:
            print(f"  {path.relative_to(dataset_dir)}")
    if duplicate_groups:
        print("\nPrimeros duplicados exactos:")
        for group in duplicate_groups[:10]:
            print("  " + " | ".join(str(path.relative_to(dataset_dir)) for path in group))

    return valid_files


def plot_summary(valid_files: dict[str, list[Path]]) -> None:
    labels = list(valid_files)
    values = [len(valid_files[label]) for label in labels]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(labels, values)
    axis.set_title(
        "Archivos de imagen por carpeta" if METADATA_ONLY else "Imágenes válidas por carpeta"
    )
    axis.set_ylabel("Cantidad")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    plt.show()


def show_samples(valid_files: dict[str, list[Path]]) -> None:
    rows = len(valid_files)
    figure, axes = plt.subplots(rows, SAMPLES_PER_CLASS, figsize=(15, 3 * rows), squeeze=False)

    for row, (class_name, files) in enumerate(valid_files.items()):
        for column, path in enumerate(files[:SAMPLES_PER_CLASS]):
            with Image.open(path) as image:
                axes[row, column].imshow(image.convert("RGB"))
            axes[row, column].set_title(class_name)
        for axis in axes[row]:
            axis.axis("off")

    figure.suptitle("Muestras del dataset", y=1.01)
    figure.tight_layout()
    plt.show()


def main() -> None:
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"No se encontró el dataset en: {DATASET_DIR}")

    class_dirs = get_class_dirs(DATASET_DIR)
    print("Clases detectadas:", [path.name for path in class_dirs])
    valid_files = audit_dataset(DATASET_DIR, class_dirs)
    plot_summary(valid_files)
    if not METADATA_ONLY:
        show_samples(valid_files)


if __name__ == "__main__":
    main()
