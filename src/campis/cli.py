"""Interfaz de línea de comandos del pipeline CAMPIS."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

from .audit import audit_dataset, format_summary
from .config import AppConfig, load_config
from .data import manifest_counts, prepare_manifest, read_manifest_file
from .exporting import export_model
from .inference import discover_inputs, load_labels, predict_files
from .pipeline import (
    evaluate_existing_model,
    report_existing_run,
    train_pipeline,
)
from .utils import atomic_write_json, setup_logging


DEFAULT_CONFIG = Path("configs/default.toml")


def _add_common(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="TOML de configuración (predeterminado: configs/default.toml).",
    )
    subparser.add_argument(
        "--verbose", action="store_true", help="Muestra diagnóstico detallado."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="campis",
        description="Pipeline reproducible de clasificación CAMPIS/Irish.",
    )
    subparsers = parser.add_subparsers(dest="command")

    audit = subparsers.add_parser("audit", help="Audita integridad y distribución.")
    _add_common(audit)
    audit.add_argument(
        "--quick",
        action="store_true",
        help="Solo cuenta archivos; omite apertura y hashes.",
    )
    audit.add_argument(
        "--verify-images",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Abre y verifica cada imagen.",
    )
    audit.add_argument(
        "--decode-tf",
        action="store_true",
        help=(
            "Decodifica cada imagen con TensorFlow, el mismo decodificador del "
            "entrenamiento. Detecta los archivos que Pillow acepta y tf.data "
            "rechaza."
        ),
    )
    audit.add_argument(
        "--hash-duplicates",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Calcula SHA-256 para detectar duplicados exactos.",
    )
    audit.add_argument("--output", type=Path, help="Ruta opcional del reporte JSON.")

    prepare = subparsers.add_parser(
        "prepare", help="Genera el manifest train/val/test sin copiar imágenes."
    )
    _add_common(prepare)
    prepare.add_argument(
        "--force", action="store_true", help="Reemplaza el manifest existente."
    )
    prepare.add_argument(
        "--verify-images",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    prepare.add_argument(
        "--hash-duplicates",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    train = subparsers.add_parser(
        "train", help="Ejecuta preparación, entrenamiento, evaluación y exportación."
    )
    _add_common(train)
    train.add_argument("--force-prepare", action="store_true")
    train.add_argument("--run-name", help="Sufijo legible para la ejecución.")

    evaluate = subparsers.add_parser(
        "evaluate", help="Evalúa un modelo Keras usando el manifest."
    )
    _add_common(evaluate)
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--split", choices=("train", "val", "test"), default="test")
    evaluate.add_argument("--output", type=Path)

    export = subparsers.add_parser(
        "export", help="Convierte un modelo .keras a .tflite con metadatos."
    )
    _add_common(export)
    export.add_argument("--model", type=Path, required=True)
    export.add_argument("--output", type=Path)
    export.add_argument(
        "--optimize", action=argparse.BooleanOptionalAction, default=None
    )

    report = subparsers.add_parser(
        "report", help="Genera las figuras de una ejecucion ya terminada."
    )
    _add_common(report)
    report.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Carpeta artifacts/runs/<fecha>-<nombre> de la ejecucion.",
    )

    predict = subparsers.add_parser(
        "predict", help="Predice una imagen o todas las imágenes de una carpeta."
    )
    _add_common(predict)
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--labels", type=Path)
    predict.add_argument("--output", type=Path)

    doctor = subparsers.add_parser(
        "doctor", help="Comprueba dependencias, GPU y rutas configuradas."
    )
    _add_common(doctor)
    return parser


def _output_path(path: Path | None, config: AppConfig, default: str) -> Path:
    if path is None:
        return config.paths.artifacts_dir / default
    return path if path.is_absolute() else config.paths.project_root / path


def _command_audit(args: argparse.Namespace, config: AppConfig) -> int:
    verify = False if args.quick else args.verify_images
    hashes = False if args.quick else args.hash_duplicates
    output = _output_path(
        args.output, config, "reports/dataset_audit.json"
    )
    summary, _ = audit_dataset(
        config,
        verify_images=verify,
        hash_duplicates=hashes,
        output_path=output,
        decode_with_tensorflow=args.decode_tf,
    )
    print(format_summary(summary))
    print(f"Reporte: {output}")
    problems = (
        summary["invalid_images"]
        or summary["conflicting_label_hashes"]
        or summary.get("tensorflow_decode_failures")
    )
    return 1 if problems else 0


def _command_prepare(args: argparse.Namespace, config: AppConfig) -> int:
    if config.paths.manifest_path.exists() and not args.force:
        raise FileExistsError(
            f"El manifest ya existe: {config.paths.manifest_path}. Usa --force para regenerarlo."
        )
    records, _ = prepare_manifest(
        config,
        verify_images=args.verify_images,
        hash_duplicates=args.hash_duplicates,
    )
    print(json.dumps(manifest_counts(records), ensure_ascii=False, indent=2))
    print(f"Manifest: {config.paths.manifest_path}")
    return 0


def _command_train(args: argparse.Namespace, config: AppConfig) -> int:
    run_dir = train_pipeline(
        config,
        force_prepare=args.force_prepare,
        run_name=args.run_name,
    )
    print(f"Ejecución completada: {run_dir}")
    return 0


def _command_evaluate(args: argparse.Namespace, config: AppConfig) -> int:
    output = _output_path(
        args.output,
        config,
        f"evaluations/{args.model.stem}-{args.split}",
    )
    metrics = evaluate_existing_model(
        config, args.model, output, split=args.split
    )
    print(json.dumps(metrics["global"], ensure_ascii=False, indent=2, default=str))
    print(f"Artefactos: {output}")
    return 0


def _command_export(args: argparse.Namespace, config: AppConfig) -> int:
    output = args.output or args.model.with_suffix(".tflite")
    optimize = config.model.tflite_optimize if args.optimize is None else args.optimize
    artifacts = export_model(
        args.model,
        output,
        config.data.classes,
        optimize=optimize,
        extra_metadata={
            "camera_domains": {
                source.name: source.camera_type for source in config.data.sources
            }
        },
    )
    print(f"TFLite: {artifacts.tflite_path}")
    print(f"Etiquetas: {artifacts.labels_path}")
    print(f"Metadatos: {artifacts.metadata_path}")
    return 0


def _command_report(args: argparse.Namespace, config: AppConfig) -> int:
    run_dir = args.run if args.run.is_absolute() else config.paths.project_root / args.run
    figures = report_existing_run(config, run_dir)
    if not figures:
        print("No se genero ninguna figura: faltan los artefactos de la ejecucion.")
        return 1
    for figure in figures:
        print(figure)
    print(f"Figuras: {len(figures)} en {run_dir / 'figures'}")
    return 0


def _command_predict(args: argparse.Namespace, config: AppConfig) -> int:
    suffix = args.model.suffix.casefold()
    if suffix not in {".keras", ".tflite"}:
        raise ValueError("--model debe terminar en .keras o .tflite.")
    runtime = "keras" if suffix == ".keras" else "tflite"
    labels: Sequence[str] = (
        load_labels(args.labels) if args.labels else config.data.classes
    )
    files = discover_inputs(args.input, config.data.extensions)
    predictions = predict_files(
        args.model,
        files,
        labels,
        config.model.image_size,
        runtime=runtime,
        batch_size=config.training.batch_size,
        crop_to_square=config.model.crop_to_square,
    )
    payload = [prediction.to_dict() for prediction in predictions]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output:
        atomic_write_json(args.output, payload)
    return 0


def _command_doctor(config: AppConfig) -> int:
    import platform

    print(f"Python: {platform.python_version()}")
    print(f"Proyecto: {config.paths.project_root}")
    for source in config.data.sources:
        print(f"Dataset {source.name} ({source.camera_type}): {source.path}")
    try:
        import tensorflow as tf

        print(f"TensorFlow: {tf.__version__}")
        print(f"GPU detectadas: {len(tf.config.list_physical_devices('GPU'))}")
    except ImportError:
        print("TensorFlow: NO INSTALADO")
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = list(argv) if argv is not None else sys.argv[1:]
    if not arguments:
        parser.print_help()
        return 0
    args = parser.parse_args(arguments)
    logger = setup_logging(verbose=getattr(args, "verbose", False))

    try:
        config = load_config(args.config)
        handlers = {
            "audit": _command_audit,
            "prepare": _command_prepare,
            "train": _command_train,
            "evaluate": _command_evaluate,
            "export": _command_export,
            "report": _command_report,
            "predict": _command_predict,
            "doctor": lambda namespace, loaded: _command_doctor(loaded),
        }
        return handlers[args.command](args, config)
    except KeyboardInterrupt:
        logger.error("Operación cancelada por el usuario.")
        return 130
    except Exception as error:
        if getattr(args, "verbose", False):
            logger.exception("La operación falló.")
        else:
            logger.error("%s", error)
        return 2


__all__ = ["build_parser", "main"]
