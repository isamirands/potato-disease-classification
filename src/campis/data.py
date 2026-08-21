"""Catálogo, particionado virtual y entrada ``tf.data`` para ambos dominios.

Los datasets originales se tratan como datos crudos inmutables. El split vive
en un CSV y nunca se materializa copiando o moviendo imágenes.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageFile, UnidentifiedImageError

from .config import AppConfig
from .utils import atomic_write_csv, read_csv


LOGGER = logging.getLogger("campis.data")
# Si otra libreria lo activa, Pillow rellenaria los JPEG truncados con gris en
# vez de fallar, y la auditoria los daria por buenos.
ImageFile.LOAD_TRUNCATED_IMAGES = False
SPLITS = ("train", "val", "test")
MANIFEST_FIELDS = (
    "relative_path",
    "source",
    "camera_type",
    "class_name",
    "class_index",
    "split",
    "sha256",
    "group_id",
    "orientation",
    "width",
    "height",
    "size_bytes",
)
_TIMESTAMP_RE = re.compile(r"(?P<date>\d{8})[_-]?(?P<time>\d{6})")
# Contadores de cámara tipo ``IMG_0051``: prefijo no numérico y sufijo corto.
# Excluye deliberadamente nombres que son solo dígitos (marcas epoch).
_SEQUENCE_RE = re.compile(r"^(?P<prefix>\D+)(?P<number>\d{1,6})$")


@dataclass(frozen=True, slots=True)
class ImageRecord:
    relative_path: str
    source: str
    camera_type: str
    class_name: str
    class_index: int
    split: str = ""
    sha256: str = ""
    group_id: str = ""
    orientation: int = 1
    width: int = 0
    height: int = 0
    size_bytes: int = 0

    def resolve(self, project_root: str | Path) -> Path:
        path = Path(self.relative_path)
        return path if path.is_absolute() else Path(project_root) / path

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class InvalidImage:
    relative_path: str
    source: str
    class_name: str
    error: str


@dataclass(frozen=True, slots=True)
class ScanResult:
    records: tuple[ImageRecord, ...]
    invalid_images: tuple[InvalidImage, ...]


def _relative_path(path: Path, project_root: Path) -> str:
    try:
        relative = path.resolve().relative_to(project_root.resolve())
    except ValueError:
        relative = Path(os.path.relpath(path.resolve(), project_root.resolve()))
    return relative.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_capture_seconds(stem: str) -> tuple[str, int] | None:
    """Devuelve ``(esquema, segundos)`` para los nombres con marca temporal.

    Reconoce ``20250107_093439`` y también las marcas epoch de 10 dígitos
    (segundos) y 13 dígitos (milisegundos) que producen algunas cámaras de
    celular. Cada esquema mantiene su propia escala y se etiqueta por separado
    para que dos convenciones distintas nunca compartan una ventana.
    """

    if stem.isdigit():
        if len(stem) == 13:
            return "epoch_ms", int(stem) // 1000
        if len(stem) == 10:
            return "epoch_s", int(stem)
        return None

    match = _TIMESTAMP_RE.search(stem)
    if not match:
        return None
    try:
        captured = datetime.strptime(
            match.group("date") + match.group("time"), "%Y%m%d%H%M%S"
        )
    except (ValueError, OverflowError, OSError):
        return None
    # Evita que la zona horaria del equipo cambie los grupos.
    return "filename", (
        captured.toordinal() * 86_400
        + captured.hour * 3_600
        + captured.minute * 60
        + captured.second
    )


def _sequence_runs(
    paths: Iterable[Path],
    source: str,
    max_gap: int,
) -> dict[Path, str]:
    """Agrupa contadores de cámara correlativos en una sola sesión.

    Un contador como ``IMG_0051, IMG_0052, IMG_0054`` corresponde a disparos
    seguidos sobre la misma planta. El contador es global a la cámara, así que
    la detección recorre todas las clases de la fuente a la vez: una ráfaga que
    cambió de clase a mitad de sesión sigue siendo una sola sesión.
    """

    if max_gap <= 0:
        return {}

    by_prefix: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for path in paths:
        match = _SEQUENCE_RE.match(path.stem)
        if match:
            by_prefix[match.group("prefix").casefold()].append(
                (int(match.group("number")), path)
            )

    runs: dict[Path, str] = {}
    for prefix, entries in by_prefix.items():
        entries.sort(key=lambda item: (item[0], str(item[1])))
        run_start = entries[0][0]
        previous = run_start
        for number, path in entries:
            if number - previous > max_gap:
                run_start = number
            previous = number
            runs[path] = f"sequence:{source}:{prefix}:{run_start}"
    return runs


def _capture_group(
    source: str,
    class_name: str,
    image_path: Path,
    relative_path: str,
    digest: str,
    group_campis_by_minutes: int,
    sequence_group: str = "",
) -> str:
    if source.casefold() == "campis" and group_campis_by_minutes > 0:
        parsed = _parse_capture_seconds(image_path.stem)
        if parsed is not None:
            scheme, seconds = parsed
            window_seconds = group_campis_by_minutes * 60
            bucket = seconds - seconds % window_seconds
            return f"capture:{source}:{scheme}:{bucket}"
    if sequence_group:
        return sequence_group
    if digest:
        return f"sha256:{digest}"
    return f"path:{source}:{class_name}:{relative_path}"


def scan_dataset(
    config: AppConfig,
    *,
    verify_images: bool | None = None,
    hash_duplicates: bool | None = None,
    max_per_source_class: int | None = None,
    progress_interval: int = 1000,
) -> ScanResult:
    """Descubre imágenes válidas y metadatos sin modificar ninguna fuente."""

    verify = config.data.verify_images if verify_images is None else verify_images
    hash_files = (
        config.data.hash_duplicates if hash_duplicates is None else hash_duplicates
    )
    limit = (
        config.data.max_images_per_source_class
        if max_per_source_class is None
        else max_per_source_class
    )
    if limit < 0:
        raise ValueError("max_per_source_class no puede ser negativo.")

    records: list[ImageRecord] = []
    invalid: list[InvalidImage] = []
    processed = 0
    extensions = set(config.data.extensions)
    project_root = config.paths.project_root

    for source in config.data.sources:
        # Primera pasada: solo nombres. El contador de la cámara es global a la
        # fuente, así que las ráfagas se detectan sobre todas sus clases juntas.
        candidates_by_class: dict[str, list[Path]] = {}
        for class_name in config.data.classes:
            class_dir = source.path / class_name
            class_candidates = sorted(
                (
                    path
                    for path in class_dir.iterdir()
                    if path.is_file() and path.suffix.casefold() in extensions
                ),
                key=lambda path: path.name.casefold(),
            )
            if limit:
                class_candidates = class_candidates[:limit]
            if not class_candidates:
                raise ValueError(
                    f"No hay imágenes compatibles para {source.name}/{class_name}."
                )
            candidates_by_class[class_name] = class_candidates

        sequence_groups = (
            _sequence_runs(
                (path for paths in candidates_by_class.values() for path in paths),
                source.name,
                config.data.group_sequence_gap,
            )
            if source.name.casefold() == "campis"
            else {}
        )

        for class_index, class_name in enumerate(config.data.classes):
            candidates = candidates_by_class[class_name]

            for image_path in candidates:
                relative = _relative_path(image_path, project_root)
                width = height = 0
                orientation = 1
                try:
                    size_bytes = image_path.stat().st_size
                    if verify:
                        with Image.open(image_path) as image:
                            width, height = image.size
                            raw_orientation = image.getexif().get(274, 1)
                            try:
                                parsed_orientation = int(raw_orientation)
                            except (TypeError, ValueError):
                                parsed_orientation = 1
                            orientation = (
                                parsed_orientation
                                if 1 <= parsed_orientation <= 8
                                else 1
                            )
                        # PNG exige que verify() sea la primera operación de
                        # decodificación sobre una instancia recién abierta.
                        with Image.open(image_path) as image:
                            image.verify()
                        # verify() valida marcadores, no descomprime: un JPEG
                        # cortado a mitad del scan lo pasa y luego revienta
                        # dentro de tf.data, con el entrenamiento ya avanzado.
                        # load() descomprime de verdad y lo detecta aquí.
                        with Image.open(image_path) as image:
                            image.load()
                    digest = _sha256(image_path) if hash_files else ""
                except (UnidentifiedImageError, OSError, ValueError, RuntimeError) as error:
                    invalid.append(
                        InvalidImage(
                            relative_path=relative,
                            source=source.name,
                            class_name=class_name,
                            error=str(error),
                        )
                    )
                    continue

                records.append(
                    ImageRecord(
                        relative_path=relative,
                        source=source.name,
                        camera_type=source.camera_type,
                        class_name=class_name,
                        class_index=class_index,
                        sha256=digest,
                        group_id=_capture_group(
                            source.name,
                            class_name,
                            image_path,
                            relative,
                            digest,
                            config.data.group_campis_by_minutes,
                            sequence_groups.get(image_path, ""),
                        ),
                        orientation=orientation,
                        width=width,
                        height=height,
                        size_bytes=size_bytes,
                    )
                )
                processed += 1
                if progress_interval and processed % progress_interval == 0:
                    LOGGER.info("Imágenes catalogadas: %s", processed)

    if not records:
        raise ValueError("No se encontró ninguna imagen válida.")
    return ScanResult(tuple(records), tuple(invalid))


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _group_records(records: Sequence[ImageRecord]) -> list[list[int]]:
    """Une sesiones y hashes; un duplicado nunca puede cruzar splits."""

    disjoint = _DisjointSet(len(records))
    first_by_key: dict[str, int] = {}
    for index, record in enumerate(records):
        keys = [f"group:{record.group_id}"]
        if record.sha256:
            keys.append(f"sha256:{record.sha256}")
        for key in keys:
            if key in first_by_key:
                disjoint.union(index, first_by_key[key])
            else:
                first_by_key[key] = index

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        groups[disjoint.find(index)].append(index)
    return list(groups.values())


def _validate_hash_labels(records: Sequence[ImageRecord]) -> None:
    labels_by_hash: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.sha256:
            labels_by_hash[record.sha256].add(record.class_name)
    conflicts = {
        digest: sorted(labels)
        for digest, labels in labels_by_hash.items()
        if len(labels) > 1
    }
    if conflicts:
        digest, labels = next(iter(conflicts.items()))
        raise ValueError(
            "Se encontró contenido idéntico con etiquetas distintas: "
            f"sha256={digest}, clases={labels}."
        )


def assign_splits(
    records: Sequence[ImageRecord],
    ratios: Mapping[str, float],
    *,
    seed: int,
) -> list[ImageRecord]:
    """Asigna grupos completos minimizando el desvío por fuente y clase."""

    if not records:
        raise ValueError("No hay registros para dividir.")
    if set(ratios) != set(SPLITS):
        raise ValueError(f"ratios debe contener exactamente {SPLITS}.")
    if abs(sum(float(ratios[name]) for name in SPLITS) - 1.0) > 1e-9:
        raise ValueError("Los ratios deben sumar 1.0.")
    _validate_hash_labels(records)

    strata_totals = Counter((record.source, record.class_name) for record in records)
    targets = {
        (stratum, split): total * float(ratios[split])
        for stratum, total in strata_totals.items()
        for split in SPLITS
    }
    assigned: Counter[tuple[tuple[str, str], str]] = Counter()

    groups = _group_records(records)

    def group_tie(indices: Sequence[int]) -> str:
        identity = "|".join(sorted(records[index].group_id for index in indices))
        return hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()

    groups.sort(key=lambda indices: (-len(indices), group_tie(indices)))
    split_for_index: dict[int, str] = {}
    for indices in groups:
        group_counts = Counter(
            (records[index].source, records[index].class_name) for index in indices
        )
        candidate_scores: list[tuple[float, str, str]] = []
        identity = group_tie(indices)
        for split in SPLITS:
            score = 0.0
            for stratum, total in strata_totals.items():
                for candidate_split in SPLITS:
                    value = assigned[(stratum, candidate_split)]
                    if candidate_split == split:
                        value += group_counts[stratum]
                    target = targets[(stratum, candidate_split)]
                    score += ((value - target) ** 2) / max(target, 1.0)
            split_tie = hashlib.sha256(
                f"{identity}:{split}".encode()
            ).hexdigest()
            candidate_scores.append((score, split_tie, split))
        chosen = min(candidate_scores)[2]
        for stratum, count in group_counts.items():
            assigned[(stratum, chosen)] += count
        for index in indices:
            split_for_index[index] = chosen

    result = [
        replace(record, split=split_for_index[index])
        for index, record in enumerate(records)
    ]
    validate_manifest(result, require_all_splits=True)
    return sorted(
        result,
        key=lambda record: (
            SPLITS.index(record.split),
            record.source,
            record.class_index,
            record.relative_path.casefold(),
        ),
    )


def validate_manifest(
    records: Sequence[ImageRecord], *, require_all_splits: bool = True
) -> None:
    if not records:
        raise ValueError("El manifest está vacío.")
    paths = [record.relative_path for record in records]
    if len(paths) != len(set(paths)):
        raise ValueError("El manifest contiene rutas repetidas.")

    class_indices: dict[str, set[int]] = defaultdict(set)
    splits_by_group: dict[str, set[str]] = defaultdict(set)
    splits_by_hash: dict[str, set[str]] = defaultdict(set)
    strata: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        if record.split not in SPLITS:
            raise ValueError(f"Split inválido para {record.relative_path}: {record.split!r}")
        if not 1 <= record.orientation <= 8:
            raise ValueError(
                f"Orientación EXIF inválida para {record.relative_path}: "
                f"{record.orientation}"
            )
        class_indices[record.class_name].add(record.class_index)
        splits_by_group[record.group_id].add(record.split)
        if record.sha256:
            splits_by_hash[record.sha256].add(record.split)
        strata[(record.source, record.class_name)].add(record.split)

    if any(len(indices) != 1 for indices in class_indices.values()):
        raise ValueError("Una clase aparece asociada a varios índices.")
    if len({next(iter(indices)) for indices in class_indices.values()}) != len(
        class_indices
    ):
        raise ValueError("Dos clases comparten el mismo índice.")
    if any(len(splits) > 1 for splits in splits_by_group.values()):
        raise ValueError("Un group_id aparece en más de un split.")
    if any(len(splits) > 1 for splits in splits_by_hash.values()):
        raise ValueError("Un SHA-256 duplicado aparece en más de un split.")
    _validate_hash_labels(records)

    if require_all_splits:
        incomplete = [stratum for stratum, splits in strata.items() if splits != set(SPLITS)]
        if incomplete:
            raise ValueError(
                "No fue posible representar train/val/test en: "
                + ", ".join(f"{source}/{class_name}" for source, class_name in incomplete)
                + ". Hay muy pocos grupos independientes."
            )


def write_manifest(records: Sequence[ImageRecord], path: str | Path) -> Path:
    validate_manifest(records)
    return atomic_write_csv(
        path, (record.to_row() for record in records), fieldnames=MANIFEST_FIELDS
    )


def read_manifest_file(path: str | Path) -> list[ImageRecord]:
    rows = read_csv(path)
    # `orientation` se añadió en v1.0; manifests previos se interpretan como 1.
    required_fields = set(MANIFEST_FIELDS) - {"orientation"}
    missing = required_fields - (set(rows[0]) if rows else set())
    if missing:
        raise ValueError(f"Faltan columnas en el manifest: {sorted(missing)}")
    try:
        records = [
            ImageRecord(
                relative_path=row["relative_path"],
                source=row["source"],
                camera_type=row["camera_type"],
                class_name=row["class_name"],
                class_index=int(row["class_index"]),
                split=row["split"],
                sha256=row["sha256"],
                group_id=row["group_id"],
                orientation=int(row.get("orientation") or 1),
                width=int(row["width"] or 0),
                height=int(row["height"] or 0),
                size_bytes=int(row["size_bytes"] or 0),
            )
            for row in rows
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Manifest inválido en {path}: {error}") from error
    validate_manifest(records)
    return records


def prepare_manifest(
    config: AppConfig,
    *,
    verify_images: bool | None = None,
    hash_duplicates: bool | None = None,
) -> tuple[list[ImageRecord], tuple[InvalidImage, ...]]:
    scan = scan_dataset(
        config,
        verify_images=verify_images,
        hash_duplicates=hash_duplicates,
    )
    if scan.invalid_images:
        first = scan.invalid_images[0]
        raise ValueError(
            f"Se detectaron {len(scan.invalid_images)} imágenes inválidas. "
            f"Primera: {first.relative_path}: {first.error}"
        )
    records = assign_splits(scan.records, config.data.ratios, seed=config.data.seed)
    write_manifest(records, config.paths.manifest_path)
    return records, scan.invalid_images


def records_for_split(
    records: Iterable[ImageRecord], split: str
) -> list[ImageRecord]:
    if split not in SPLITS:
        raise ValueError(f"Split desconocido: {split}")
    return sorted(
        (record for record in records if record.split == split),
        key=lambda record: (record.source, record.class_index, record.relative_path),
    )


def balance_records(
    records: Sequence[ImageRecord],
    *,
    strategy: str,
    target_per_group: int,
    seed: int,
) -> list[ImageRecord]:
    """Submuestrea o repite referencias; nunca crea nuevas imágenes."""

    if not records:
        raise ValueError("No hay registros para balancear.")
    if target_per_group <= 0:
        raise ValueError("target_per_group debe ser mayor que cero.")
    if strategy == "none":
        return list(records)
    key_functions = {
        "class": lambda record: (record.class_name,),
        "source": lambda record: (record.source,),
        "source_class": lambda record: (record.source, record.class_name),
    }
    if strategy not in key_functions:
        raise ValueError(f"Estrategia de balance desconocida: {strategy}")

    grouped: dict[tuple[str, ...], list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[key_functions[strategy](record)].append(record)

    balanced: list[ImageRecord] = []
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda record: record.relative_path)
        rng = random.Random(f"{seed}:{'|'.join(key)}")
        rng.shuffle(group)
        if len(group) >= target_per_group:
            balanced.extend(group[:target_per_group])
        else:
            repeats, remainder = divmod(target_per_group, len(group))
            balanced.extend(group * repeats)
            balanced.extend(group[:remainder])
    random.Random(seed).shuffle(balanced)
    return balanced


def center_crop_square(image: Any) -> Any:
    """Recorta el cuadrado central para igualar la geometría de los dominios.

    CAMPIS entrega fotos 1:1 e Irish 4:3; redimensionar directo a un lienzo
    cuadrado deformaría solo a Irish y dejaría la relación de aspecto como una
    pista espuria del dominio de captura.
    """

    import tensorflow as tf

    shape = tf.shape(image)
    side = tf.minimum(shape[0], shape[1])
    offset_height = (shape[0] - side) // 2
    offset_width = (shape[1] - side) // 2
    cropped = tf.image.crop_to_bounding_box(
        image, offset_height, offset_width, side, side
    )
    cropped.set_shape((None, None, 3))
    return cropped


def build_tf_dataset(
    records: Sequence[ImageRecord],
    project_root: str | Path,
    image_size: Sequence[int],
    batch_size: int,
    *,
    training: bool,
    seed: int,
    num_workers: int = 0,
    crop_to_square: bool = True,
) -> Any:
    """Crea batches `(RGB float32 0..255, class_index)` reproducibles."""

    if not records:
        raise ValueError("No hay registros para crear tf.data.Dataset.")
    if len(image_size) != 2 or any(int(value) <= 0 for value in image_size):
        raise ValueError("image_size debe contener [alto, ancho] positivos.")
    if batch_size <= 0:
        raise ValueError("batch_size debe ser mayor que cero.")
    if num_workers < 0:
        raise ValueError("num_workers no puede ser negativo.")

    try:
        import tensorflow as tf
    except ImportError as error:  # pragma: no cover
        raise ImportError("TensorFlow es necesario para construir el dataset.") from error

    # Conserva el orden del manifest: evaluación alinea predicciones y records.
    ordered = list(records)
    paths = [str(record.resolve(project_root)) for record in ordered]
    labels = [record.class_index for record in ordered]
    orientations = [record.orientation for record in ordered]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels, orientations))
    if training:
        dataset = dataset.shuffle(
            buffer_size=min(len(ordered), 10_000),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    height, width = (int(value) for value in image_size)

    def decode(path: Any, label: Any, orientation: Any) -> tuple[Any, Any]:
        encoded = tf.io.read_file(path)
        image = tf.io.decode_image(encoded, channels=3, expand_animations=False)
        image.set_shape((None, None, 3))
        image = apply_exif_orientation(image, orientation)
        if crop_to_square:
            image = center_crop_square(image)
        image = tf.image.resize(
            image, (height, width), method="bilinear", antialias=True
        )
        image = tf.clip_by_value(tf.cast(image, tf.float32), 0.0, 255.0)
        return image, tf.cast(label, tf.int32)

    options = tf.data.Options()
    options.experimental_deterministic = True
    if num_workers:
        options.threading.private_threadpool_size = num_workers
    dataset = dataset.with_options(options)
    parallel_calls = num_workers if num_workers else tf.data.AUTOTUNE
    dataset = dataset.map(decode, num_parallel_calls=parallel_calls)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset.prefetch(tf.data.AUTOTUNE)


def balance_groups(
    records: Sequence[ImageRecord], strategy: str
) -> dict[tuple[str, ...], list[ImageRecord]]:
    """Agrupa registros según la estrategia de balance, sin muestrear nada."""

    key_functions = {
        "none": lambda record: ("all",),
        "class": lambda record: (record.class_name,),
        "source": lambda record: (record.source,),
        "source_class": lambda record: (record.source, record.class_name),
    }
    if strategy not in key_functions:
        raise ValueError(f"Estrategia de balance desconocida: {strategy}")

    grouped: dict[tuple[str, ...], list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[key_functions[strategy](record)].append(record)
    return {
        key: sorted(grouped[key], key=lambda record: record.relative_path)
        for key in sorted(grouped)
    }


def build_balanced_tf_dataset(
    records: Sequence[ImageRecord],
    project_root: str | Path,
    image_size: Sequence[int],
    batch_size: int,
    *,
    strategy: str,
    target_per_group: int,
    seed: int,
    num_workers: int = 0,
    crop_to_square: bool = True,
) -> tuple[Any, int]:
    """Entrega épocas balanceadas que remuestrean todo el corpus disponible.

    Cada grupo de balance se baraja y se repite de forma independiente, y la
    época se arma en turnos estrictos entre grupos. A diferencia de un
    submuestreo fijo previo al entrenamiento, aquí la fuente mayoritaria aporta
    ``target_per_group`` imágenes **distintas** en cada época hasta agotar su
    corpus, y la minoritaria se repite con aumento en vez de desperdiciar datos.

    Devuelve el dataset y el número de pasos que compone una época.
    """

    if not records:
        raise ValueError("No hay registros para crear tf.data.Dataset.")
    if target_per_group <= 0:
        raise ValueError("target_per_group debe ser mayor que cero.")
    if batch_size <= 0:
        raise ValueError("batch_size debe ser mayor que cero.")
    if len(image_size) != 2 or any(int(value) <= 0 for value in image_size):
        raise ValueError("image_size debe contener [alto, ancho] positivos.")
    if num_workers < 0:
        raise ValueError("num_workers no puede ser negativo.")

    try:
        import tensorflow as tf
    except ImportError as error:  # pragma: no cover
        raise ImportError("TensorFlow es necesario para construir el dataset.") from error

    grouped = balance_groups(records, strategy)
    height, width = (int(value) for value in image_size)

    def decode(path: Any, label: Any, orientation: Any) -> tuple[Any, Any]:
        encoded = tf.io.read_file(path)
        image = tf.io.decode_image(encoded, channels=3, expand_animations=False)
        image.set_shape((None, None, 3))
        image = apply_exif_orientation(image, orientation)
        if crop_to_square:
            image = center_crop_square(image)
        image = tf.image.resize(
            image, (height, width), method="bilinear", antialias=True
        )
        image = tf.clip_by_value(tf.cast(image, tf.float32), 0.0, 255.0)
        return image, tf.cast(label, tf.int32)

    group_datasets: list[Any] = []
    for offset, (key, group) in enumerate(sorted(grouped.items())):
        paths = [str(record.resolve(project_root)) for record in group]
        labels = [record.class_index for record in group]
        orientations = [record.orientation for record in group]
        group_dataset = tf.data.Dataset.from_tensor_slices(
            (paths, labels, orientations)
        )
        group_dataset = group_dataset.shuffle(
            buffer_size=len(group),
            seed=seed + offset,
            reshuffle_each_iteration=True,
        ).repeat()
        group_datasets.append(group_dataset)

    epoch_size = target_per_group * len(group_datasets)
    if len(group_datasets) == 1:
        dataset = group_datasets[0]
    else:
        # Turnos estrictos: cada época contiene exactamente target_per_group
        # referencias de cada grupo, sin depender de un sorteo multinomial.
        selector = tf.data.Dataset.range(len(group_datasets)).repeat()
        dataset = tf.data.Dataset.choose_from_datasets(group_datasets, selector)
    dataset = dataset.take(epoch_size)
    # Rompe el patrón cíclico para que un batch no sea siempre la misma
    # secuencia de grupos.
    dataset = dataset.shuffle(
        buffer_size=min(epoch_size, 10_000),
        seed=seed,
        reshuffle_each_iteration=True,
    )

    options = tf.data.Options()
    options.experimental_deterministic = True
    if num_workers:
        options.threading.private_threadpool_size = num_workers
    dataset = dataset.with_options(options)
    parallel_calls = num_workers if num_workers else tf.data.AUTOTUNE
    dataset = dataset.map(decode, num_parallel_calls=parallel_calls)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    steps_per_epoch = math.ceil(epoch_size / batch_size)
    return dataset, steps_per_epoch



def apply_exif_orientation(image: Any, orientation: Any) -> Any:
    """Aplica las ocho orientaciones EXIF usando solo operaciones TensorFlow."""

    import tensorflow as tf

    orientation = tf.cast(orientation, tf.int32)
    orientation = tf.where(
        tf.logical_and(orientation >= 1, orientation <= 8), orientation, 1
    )

    def transpose() -> Any:
        return tf.transpose(image, perm=(1, 0, 2))

    transformed = tf.switch_case(
        orientation - 1,
        branch_fns=(
            lambda: image,
            lambda: tf.image.flip_left_right(image),
            lambda: tf.image.rot90(image, k=2),
            lambda: tf.image.flip_up_down(image),
            transpose,
            lambda: tf.image.rot90(image, k=3),
            lambda: tf.image.rot90(transpose(), k=2),
            lambda: tf.image.rot90(image, k=1),
        ),
    )
    transformed.set_shape((None, None, 3))
    return transformed


def manifest_counts(records: Iterable[ImageRecord]) -> dict[str, Any]:
    records = list(records)
    return {
        "total": len(records),
        "by_split": dict(sorted(Counter(record.split for record in records).items())),
        "by_source": dict(sorted(Counter(record.source for record in records).items())),
        "by_class": dict(sorted(Counter(record.class_name for record in records).items())),
        "by_source_class_split": {
            "|".join(key): count
            for key, count in sorted(
                Counter(
                    (record.source, record.class_name, record.split)
                    for record in records
                ).items()
            )
        },
    }


__all__ = [
    "ImageRecord",
    "InvalidImage",
    "MANIFEST_FIELDS",
    "SPLITS",
    "ScanResult",
    "assign_splits",
    "apply_exif_orientation",
    "balance_records",
    "build_balanced_tf_dataset",
    "build_tf_dataset",
    "center_crop_square",
    "manifest_counts",
    "prepare_manifest",
    "read_manifest_file",
    "records_for_split",
    "scan_dataset",
    "validate_manifest",
    "write_manifest",
]
