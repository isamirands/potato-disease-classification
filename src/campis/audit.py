"""Auditoría de integridad y distribución para CAMPIS e Irish."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from .config import AppConfig
from .data import ImageRecord, InvalidImage, ScanResult, scan_dataset
from .utils import atomic_write_json


LOGGER = logging.getLogger("campis.audit")


def summarize_scan(scan: ScanResult) -> dict[str, Any]:
    records = list(scan.records)
    by_source = Counter(record.source for record in records)
    by_class = Counter(record.class_name for record in records)
    by_camera = Counter(record.camera_type for record in records)
    by_source_class = Counter((record.source, record.class_name) for record in records)
    extensions = Counter(Path(record.relative_path).suffix.casefold() for record in records)
    dimensions = Counter(
        (record.width, record.height)
        for record in records
        if record.width > 0 and record.height > 0
    )
    orientations = Counter(record.orientation for record in records)

    paths_by_hash: dict[str, list[str]] = defaultdict(list)
    labels_by_hash: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.sha256:
            paths_by_hash[record.sha256].append(record.relative_path)
            labels_by_hash[record.sha256].add(record.class_name)
    duplicate_groups = {
        digest: paths for digest, paths in paths_by_hash.items() if len(paths) > 1
    }
    label_conflicts = {
        digest: sorted(labels)
        for digest, labels in labels_by_hash.items()
        if len(labels) > 1
    }

    total = len(records)
    return {
        "total_valid_images": total,
        "total_size_bytes": sum(record.size_bytes for record in records),
        "by_source": dict(sorted(by_source.items())),
        "source_percentages": {
            source: round(count * 100 / total, 4) if total else 0.0
            for source, count in sorted(by_source.items())
        },
        "by_camera_type": dict(sorted(by_camera.items())),
        "by_class": dict(sorted(by_class.items())),
        "by_source_and_class": {
            f"{source}|{class_name}": count
            for (source, class_name), count in sorted(by_source_class.items())
        },
        "extensions": dict(sorted(extensions.items())),
        "most_common_dimensions": [
            {"width": width, "height": height, "count": count}
            for (width, height), count in dimensions.most_common(20)
        ],
        "dimensions_available": bool(dimensions),
        "effective_exif_orientations": {
            str(orientation): count
            for orientation, count in sorted(orientations.items())
        },
        "invalid_images": [
            {
                "relative_path": invalid.relative_path,
                "source": invalid.source,
                "class_name": invalid.class_name,
                "error": invalid.error,
            }
            for invalid in scan.invalid_images
        ],
        "exact_duplicate_group_count": len(duplicate_groups),
        "exact_duplicate_file_count": sum(len(paths) for paths in duplicate_groups.values()),
        "exact_duplicate_examples": [
            {"sha256": digest, "paths": paths[:20]}
            for digest, paths in list(sorted(duplicate_groups.items()))[:20]
        ],
        "conflicting_label_hashes": label_conflicts,
        "hashes_calculated": any(record.sha256 for record in records),
    }


def verify_tensorflow_decoding(
    records: Sequence[ImageRecord],
    project_root: str | Path,
    *,
    num_workers: int = 8,
    progress_interval: int = 5000,
) -> tuple[InvalidImage, ...]:
    """Decodifica cada imagen con el mismo decodificador que usa el entrenamiento.

    Pillow y TensorFlow no comparten decodificador de JPEG, y no siempre
    coinciden en qué archivo es recuperable. Un archivo que Pillow acepta pero
    ``tf.io.decode_image`` rechaza aborta el entrenamiento a mitad de una época,
    cuando ya se perdieron minutos. Esta comprobación es la única concluyente,
    porque ejerce exactamente el camino de ``tf.data``.

    Las operaciones eager de TensorFlow liberan el GIL mientras descomprimen,
    así que el trabajo se reparte en hilos.
    """

    try:
        import tensorflow as tf
    except ImportError as error:  # pragma: no cover - depende del entorno
        raise ImportError(
            "TensorFlow es necesario para verificar la decodificación real."
        ) from error

    if num_workers < 1:
        raise ValueError("num_workers debe ser mayor que cero.")

    def check(record: ImageRecord) -> InvalidImage | None:
        try:
            encoded = tf.io.read_file(str(record.resolve(project_root)))
            image = tf.io.decode_image(encoded, channels=3, expand_animations=False)
            if image.shape.rank != 3:
                raise ValueError(
                    f"El tensor decodificado tiene rango {image.shape.rank}, no 3."
                )
        except Exception as error:  # noqa: BLE001 - cualquier fallo lo descalifica
            return InvalidImage(
                relative_path=record.relative_path,
                source=record.source,
                class_name=record.class_name,
                error=str(error).splitlines()[0][:300],
            )
        return None

    failures: list[InvalidImage] = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for index, result in enumerate(pool.map(check, records), start=1):
            if result is not None:
                failures.append(result)
                LOGGER.warning(
                    "TensorFlow no decodifica %s: %s",
                    result.relative_path,
                    result.error,
                )
            if progress_interval and index % progress_interval == 0:
                LOGGER.info("Imágenes decodificadas: %s", index)

    return tuple(failures)


def audit_dataset(
    config: AppConfig,
    *,
    verify_images: bool | None = None,
    hash_duplicates: bool | None = None,
    output_path: str | Path | None = None,
    decode_with_tensorflow: bool = False,
) -> tuple[dict[str, Any], ScanResult]:
    """Ejecuta la auditoría y opcionalmente guarda un reporte JSON."""

    scan = scan_dataset(
        config,
        verify_images=verify_images,
        hash_duplicates=hash_duplicates,
    )
    summary = summarize_scan(scan)
    summary["tensorflow_decode_checked"] = bool(decode_with_tensorflow)
    summary["tensorflow_decode_failures"] = []
    if decode_with_tensorflow:
        failures = verify_tensorflow_decoding(
            scan.records,
            config.paths.project_root,
            num_workers=max(1, config.training.num_workers or 8),
        )
        summary["tensorflow_decode_failures"] = [
            {
                "relative_path": failure.relative_path,
                "source": failure.source,
                "class_name": failure.class_name,
                "error": failure.error,
            }
            for failure in failures
        ]
    if output_path is not None:
        atomic_write_json(output_path, summary)
    return summary, scan


def format_summary(summary: dict[str, Any]) -> str:
    lines = [
        "=== AUDITORÍA DEL DATASET ===",
        f"Imágenes válidas: {summary['total_valid_images']:,}",
    ]
    for source, count in summary["by_source"].items():
        percentage = summary["source_percentages"][source]
        lines.append(f"  {source}: {count:,} ({percentage:.2f} %)")
    lines.append("Por clase:")
    for class_name, count in summary["by_class"].items():
        lines.append(f"  {class_name}: {count:,}")
    lines.extend(
        [
            f"Imágenes inválidas: {len(summary['invalid_images']):,}",
            f"Grupos duplicados exactos: {summary['exact_duplicate_group_count']:,}",
        ]
    )
    if not summary["hashes_calculated"]:
        lines.append("Duplicados: no calculados (auditoría rápida).")
    if summary.get("tensorflow_decode_checked"):
        failures = summary.get("tensorflow_decode_failures", [])
        lines.append(f"Ilegibles para TensorFlow: {len(failures):,}")
        for failure in failures[:10]:
            lines.append(f"  {failure['relative_path']}")
        if len(failures) > 10:
            lines.append(f"  ... y {len(failures) - 10:,} más")
    if summary["conflicting_label_hashes"]:
        lines.append("ERROR: existen hashes idénticos con etiquetas distintas.")
    return "\n".join(lines)


__all__ = [
    "audit_dataset",
    "format_summary",
    "summarize_scan",
    "verify_tensorflow_decoding",
]
