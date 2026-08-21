"""Orquestación de las etapas auditables del experimento CAMPIS."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import AppConfig
from .data import (
    ImageRecord,
    build_balanced_tf_dataset,
    build_tf_dataset,
    manifest_counts,
    prepare_manifest,
    read_manifest_file,
    records_for_split,
    validate_manifest,
    write_manifest,
)
from .evaluation import evaluate_model, save_evaluation_artifacts
from .exporting import export_model, save_labels_json, verify_tflite_parity
from .modeling import build_model
from .reporting import build_report
from .training import (
    build_domain_validation_callback,
    compute_class_weights,
    train_two_phases,
)
from .utils import atomic_write_json, create_run_dir, set_global_seed


LOGGER = logging.getLogger("campis.pipeline")

# Muestras del test que se comparan entre Keras y TFLite.
PARITY_SAMPLE_SIZE = 64
# Margenes esperables de la cuantizacion dinamica a int8 sobre un softmax,
# medidos en el percentil 99 y no en el maximo: la distribucion del error es
# muy sesgada y su maximo crece con el numero de muestras comparadas.
QUANTISED_ABSOLUTE_TOLERANCE = 0.05
QUANTISED_MEAN_TOLERANCE = 0.01


def _validate_against_config(
    records: list[ImageRecord], config: AppConfig
) -> None:
    expected_classes = {
        class_name: index for index, class_name in enumerate(config.data.classes)
    }
    source_configs = {source.name: source for source in config.data.sources}
    expected_sources = set(source_configs)
    expected_cameras = {
        source.name: source.camera_type for source in config.data.sources
    }
    observed_sources = {record.source for record in records}
    observed_classes = {record.class_name for record in records}
    if observed_sources != expected_sources:
        raise ValueError(
            "Las fuentes del manifest no coinciden con la configuración: "
            f"esperadas={sorted(expected_sources)}, observadas={sorted(observed_sources)}."
        )
    if observed_classes != set(expected_classes):
        raise ValueError(
            "Las clases del manifest no coinciden con la configuración: "
            f"esperadas={sorted(expected_classes)}, observadas={sorted(observed_classes)}."
        )
    for record in records:
        if record.source not in expected_sources:
            raise ValueError(f"Fuente inesperada en manifest: {record.source}")
        if expected_classes.get(record.class_name) != record.class_index:
            raise ValueError(
                "El orden de clases del manifest no coincide con la configuración: "
                f"{record.class_name} -> {record.class_index}."
            )
        if expected_cameras[record.source] != record.camera_type:
            raise ValueError(
                f"camera_type no coincide para {record.source}: "
                f"{record.camera_type!r} != {expected_cameras[record.source]!r}."
            )
        record_path = record.resolve(config.paths.project_root).resolve()
        expected_class_dir = (
            source_configs[record.source].path / record.class_name
        ).resolve()
        try:
            record_path.relative_to(expected_class_dir)
        except ValueError as error:
            raise ValueError(
                "La ruta del manifest está fuera de su fuente/clase configurada: "
                f"{record.relative_path} no pertenece a {expected_class_dir}."
            ) from error
        if not record_path.is_file():
            raise FileNotFoundError(
                f"Imagen del manifest no encontrada: {record.relative_path}"
            )


def load_or_prepare_manifest(
    config: AppConfig,
    *,
    force_prepare: bool = False,
) -> list[ImageRecord]:
    if config.paths.manifest_path.is_file() and not force_prepare:
        records = read_manifest_file(config.paths.manifest_path)
        if config.data.hash_duplicates and any(not record.sha256 for record in records):
            LOGGER.warning(
                "El manifest existente no contiene hashes aunque hash_duplicates=true; "
                "no se puede garantizar agrupación de duplicados exactos."
            )
    else:
        records, _ = prepare_manifest(config)
    validate_manifest(records)
    _validate_against_config(records, config)
    return records


def build_augmentation(config: AppConfig) -> Any:
    """Construye solo las transformaciones habilitadas en TOML."""

    augmentation = config.augmentation
    if not any(
        (
            augmentation.flip,
            augmentation.rotation,
            augmentation.translation,
            augmentation.zoom,
            augmentation.contrast,
        )
    ):
        return False

    import tensorflow as tf

    layers: list[Any] = []
    seed = config.data.seed
    if augmentation.flip:
        layers.append(tf.keras.layers.RandomFlip("horizontal", seed=seed))
    if augmentation.rotation:
        layers.append(
            tf.keras.layers.RandomRotation(
                augmentation.rotation, fill_mode="nearest", seed=seed + 1
            )
        )
    if augmentation.translation:
        layers.append(
            tf.keras.layers.RandomTranslation(
                augmentation.translation,
                augmentation.translation,
                fill_mode="nearest",
                seed=seed + 2,
            )
        )
    if augmentation.zoom:
        layers.append(
            tf.keras.layers.RandomZoom(
                (-augmentation.zoom, augmentation.zoom),
                (-augmentation.zoom, augmentation.zoom),
                fill_mode="nearest",
                seed=seed + 3,
            )
        )
    if augmentation.contrast:
        layers.append(
            tf.keras.layers.RandomContrast(
                augmentation.contrast, seed=seed + 4
            )
        )
    return tf.keras.Sequential(layers, name="online_augmentation")


def _datasets(
    config: AppConfig, records: list[ImageRecord]
) -> tuple[Any, Any, list[ImageRecord], int, int | None]:
    train_records = records_for_split(records, "train")
    validation_records = records_for_split(records, "val")

    if config.data.balance_strategy == "none":
        balanced_train = len(train_records)
        steps_per_epoch = None
        train_dataset = build_tf_dataset(
            balanced_train,
            config.paths.project_root,
            config.model.image_size,
            config.training.batch_size,
            training=True,
            seed=config.data.seed,
            num_workers=config.training.num_workers,
            crop_to_square=config.model.crop_to_square,
        )
    else:
        # El balance se resuelve dentro de tf.data: cada época vuelve a
        # muestrear, así que la fuente mayoritaria aporta imágenes distintas
        # en vez de repetir siempre el mismo subconjunto congelado.
        train_dataset, steps_per_epoch = build_balanced_tf_dataset(
            train_records,
            config.paths.project_root,
            config.model.image_size,
            config.training.batch_size,
            strategy=config.data.balance_strategy,
            target_per_group=config.data.samples_per_source_class,
            seed=config.data.seed,
            num_workers=config.training.num_workers,
            crop_to_square=config.model.crop_to_square,
        )
        balanced_train = steps_per_epoch * config.training.batch_size

    validation_dataset = build_tf_dataset(
        validation_records,
        config.paths.project_root,
        config.model.image_size,
        config.training.batch_size,
        training=False,
        seed=config.data.seed,
        num_workers=config.training.num_workers,
        crop_to_square=config.model.crop_to_square,
    )
    return (
        train_dataset,
        validation_dataset,
        train_records,
        balanced_train,
        steps_per_epoch,
    )


def train_pipeline(
    config: AppConfig,
    *,
    force_prepare: bool = False,
    run_name: str | None = None,
) -> Path:
    """Prepara, entrena, evalúa y exporta una ejecución completa."""

    set_global_seed(config.data.seed)
    records = load_or_prepare_manifest(config, force_prepare=force_prepare)
    run_dir = create_run_dir(config.paths.artifacts_dir, name=run_name)
    atomic_write_json(run_dir / "config.resolved.json", asdict(config))
    atomic_write_json(run_dir / "manifest_summary.json", manifest_counts(records))
    write_manifest(records, run_dir / "manifest.csv")
    save_labels_json(run_dir / "labels.json", config.data.classes)

    (
        train_dataset,
        validation_dataset,
        train_records,
        balanced_train,
        steps_per_epoch,
    ) = _datasets(config, records)
    validation_records = records_for_split(records, "val")
    LOGGER.info(
        "Entrenamiento: %s referencias únicas, %s por época tras balance.",
        len(train_records),
        balanced_train,
    )

    model = build_model(
        image_size=config.model.image_size,
        num_classes=len(config.data.classes),
        dense_units=config.model.dense_units,
        dropout=config.model.dropout,
        weights=config.model.weights,
        seed=config.data.seed,
        augmentation=build_augmentation(config),
    )
    class_weights = (
        compute_class_weights(train_records)
        if config.data.balance_strategy == "none"
        else None
    )
    validation_callback = build_domain_validation_callback(
        validation_dataset,
        [record.source for record in validation_records],
        [record.class_index for record in validation_records],
    )
    model, histories, best_phase = train_two_phases(
        model,
        train_dataset,
        validation_dataset,
        run_dir,
        head_epochs=config.training.head_epochs,
        fine_tune_epochs=config.training.fine_tune_epochs,
        head_learning_rate=config.training.head_lr,
        fine_tune_learning_rate=config.training.fine_tune_lr,
        fine_tune_fraction=config.model.fine_tune_fraction,
        patience=config.training.patience,
        class_weights=class_weights,
        validation_callback=validation_callback,
        monitor="val_balanced_loss",
        steps_per_epoch=steps_per_epoch,
    )
    atomic_write_json(
        run_dir / "training_summary.json",
        {"best_phase": best_phase, "history": histories},
    )

    test_records = records_for_split(records, "test")
    test_dataset = build_tf_dataset(
        test_records,
        config.paths.project_root,
        config.model.image_size,
        config.training.batch_size,
        training=False,
        seed=config.data.seed,
        num_workers=config.training.num_workers,
        crop_to_square=config.model.crop_to_square,
    )
    metrics, probabilities = evaluate_model(
        model, test_dataset, test_records, config.data.classes
    )
    save_evaluation_artifacts(
        metrics,
        probabilities,
        test_records,
        config.data.classes,
        run_dir / "evaluation",
    )

    exported = export_model(
        run_dir / "model.keras",
        run_dir / "model.tflite",
        config.data.classes,
        labels_path=run_dir / "model.labels.json",
        metadata_path=run_dir / "model.metadata.json",
        optimize=config.model.tflite_optimize,
        extra_metadata={
            "camera_domains": {
                source.name: source.camera_type for source in config.data.sources
            },
            "best_training_phase": best_phase,
            "crop_to_square": config.model.crop_to_square,
        },
    )
    # Cuatro imagenes no bastan para afirmar que dos runtimes coinciden, y
    # menos aun para observar un cambio de clase, que es el fallo que importa.
    # Se muestrea a lo largo de todo el test, no solo su cabecera, para cubrir
    # ambas camaras y las tres clases.
    parity_sample_size = min(PARITY_SAMPLE_SIZE, len(test_records))
    step = max(1, len(test_records) // parity_sample_size)
    parity_records = test_records[::step][:parity_sample_size]
    sample_dataset = build_tf_dataset(
        parity_records,
        config.paths.project_root,
        config.model.image_size,
        parity_sample_size,
        training=False,
        seed=config.data.seed,
        num_workers=config.training.num_workers,
        crop_to_square=config.model.crop_to_square,
    )
    sample_images, _ = next(iter(sample_dataset))
    # La cuantizacion dinamica mueve las probabilidades unas centesimas: exigir
    # 0,02 declaraba rota una conversion sana. Sin cuantizar, la conversion es
    # aritmetica pura y sigue midiendose al limite de la precision float32.
    parity = verify_tflite_parity(
        model,
        exported.tflite_path,
        sample_images.numpy(),
        absolute_tolerance=(
            QUANTISED_ABSOLUTE_TOLERANCE if config.model.tflite_optimize else 1e-5
        ),
        mean_absolute_tolerance=(
            QUANTISED_MEAN_TOLERANCE if config.model.tflite_optimize else 2e-6
        ),
        relative_tolerance=0.05 if config.model.tflite_optimize else 1e-4,
    )
    LOGGER.info(
        "Paridad TFLite: %s clases coincidentes de %s, error p%g %.5f, medio %.5f, "
        "máximo %.5f",
        parity.sample_count - parity.class_flips,
        parity.sample_count,
        parity.error_percentile,
        parity.percentile_absolute_error,
        parity.mean_absolute_error,
        parity.max_absolute_error,
    )
    atomic_write_json(run_dir / "tflite_parity.json", parity.to_dict())
    if not parity.passed:
        raise RuntimeError(
            "La exportación TFLite no superó la prueba de paridad: "
            f"error absoluto p{parity.error_percentile:g} "
            f"{parity.percentile_absolute_error:.4g} "
            f"(tolerancia {parity.absolute_tolerance:.4g}), "
            f"máximo {parity.max_absolute_error:.4g}, "
            f"medio {parity.mean_absolute_error:.4g} "
            f"(tolerancia {parity.mean_absolute_tolerance:.4g}), "
            f"{parity.unexplained_class_flips} cambios de clase no explicados "
            f"sobre {parity.sample_count} muestras. Detalle en "
            f"{run_dir / 'tflite_parity.json'}."
        )

    # Las figuras van al final y sobre archivos ya escritos: si el dibujo
    # falla, el modelo y sus métricas ya están a salvo en disco y el reporte
    # se puede repetir con `run.py report` sin volver a entrenar.
    try:
        figures = build_report(
            run_dir,
            config.paths.project_root,
            config.model.image_size,
            crop_to_square=config.model.crop_to_square,
            seed=config.data.seed,
        )
        LOGGER.info("Figuras generadas: %s", len(figures))
    except Exception as error:  # noqa: BLE001 - el reporte no invalida la corrida
        LOGGER.error(
            "No se pudieron generar las figuras (%s). El entrenamiento sí "
            "terminó; reintenta con: run.py report --run %s",
            error,
            run_dir,
        )

    LOGGER.info("Ejecución completada: %s", run_dir)
    return run_dir


def evaluate_existing_model(
    config: AppConfig,
    model_path: str | Path,
    output_dir: str | Path,
    *,
    split: str = "test",
) -> dict[str, Any]:
    import tensorflow as tf

    records = read_manifest_file(config.paths.manifest_path)
    validate_manifest(records)
    _validate_against_config(records, config)
    selected = records_for_split(records, split)
    dataset = build_tf_dataset(
        selected,
        config.paths.project_root,
        config.model.image_size,
        config.training.batch_size,
        training=False,
        seed=config.data.seed,
        num_workers=config.training.num_workers,
        crop_to_square=config.model.crop_to_square,
    )
    model = tf.keras.models.load_model(model_path, compile=False)
    metrics, probabilities = evaluate_model(
        model, dataset, selected, config.data.classes
    )
    save_evaluation_artifacts(
        metrics, probabilities, selected, config.data.classes, output_dir
    )
    return metrics


def report_existing_run(
    config: AppConfig, run_dir: str | Path
) -> list[Path]:
    """Regenera las figuras de una ejecución ya terminada."""

    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de la ejecución: {run_dir}")
    return build_report(
        run_dir,
        config.paths.project_root,
        config.model.image_size,
        crop_to_square=config.model.crop_to_square,
        seed=config.data.seed,
    )


__all__ = [
    "build_augmentation",
    "evaluate_existing_model",
    "load_or_prepare_manifest",
    "report_existing_run",
    "train_pipeline",
]
