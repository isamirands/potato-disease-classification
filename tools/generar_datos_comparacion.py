"""Construye el JSON consumido por la web de comparación TFLite."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def test_signature(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        paths = sorted(
            row["relative_path"]
            for row in csv.DictReader(handle)
            if row["split"] == "test"
        )
    return hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()


def training_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(row["split"] == "train" for row in csv.DictReader(handle))


def flip_breakdown(path: Path, tflite: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Cuenta desacuerdos Keras/TFLite por fuente sobre el test completo."""

    agreement = tflite.get("keras_tflite_agreement", {})
    existing = agreement.get("by_source")
    if existing:
        return existing

    def predictions(directory: str) -> dict[str, dict[str, str]]:
        with (path / directory / "predictions.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            return {row["relative_path"]: row for row in csv.DictReader(handle)}

    keras = predictions("evaluation")
    lite = predictions("evaluation_tflite")
    totals: dict[str, int] = {}
    flips: dict[str, int] = {}
    for relative_path, keras_row in keras.items():
        lite_row = lite.get(relative_path)
        if lite_row is None:
            continue
        source = keras_row["source"]
        totals[source] = totals.get(source, 0) + 1
        if keras_row["predicted_class"] != lite_row["predicted_class"]:
            flips[source] = flips.get(source, 0) + 1
    return {
        source: {
            "samples": samples,
            "class_flips": flips.get(source, 0),
            "class_flip_rate": flips.get(source, 0) / samples if samples else 0.0,
        }
        for source, samples in sorted(totals.items())
    }


def load_run(path: Path, key: str, label: str, description: str) -> dict[str, Any]:
    path = path.resolve()
    keras = load_json(path / "evaluation" / "metrics.json")
    tflite = load_json(path / "evaluation_tflite" / "metrics.json")
    config = load_json(path / "config.resolved.json")
    parity = load_json(path / "tflite_parity.json")
    runtime = tflite["runtime"]
    return {
        "key": key,
        "label": label,
        "description": description,
        "run": path.name,
        "train_samples": training_rows(path / "manifest.csv"),
        "test_signature": test_signature(path / "manifest.csv"),
        "test_samples": tflite["global"]["samples"],
        "keras": keras,
        "tflite": tflite,
        "parity_sample": parity,
        "mobile": {
            "model_bytes": runtime["model_bytes"],
            "mean_ms_host": runtime["mean_inference_ms_per_image"],
            "images_per_second_host": runtime["images_per_second"],
            "threads": runtime["threads"],
            "input_dtype": runtime["input_dtype"],
            "output_dtype": runtime["output_dtype"],
            "input_shape": runtime["input_shape"],
            "class_agreement_rate": tflite["keras_tflite_agreement"][
                "class_agreement_rate"
            ],
            "class_flips": tflite["keras_tflite_agreement"]["class_flips"],
            "class_flips_by_source": flip_breakdown(path, tflite),
            "sha256": runtime["model_sha256"],
        },
        "training": {
            "image_size": config["model"]["image_size"],
            "head_epochs": config["training"]["head_epochs"],
            "fine_tune_epochs": config["training"]["fine_tune_epochs"],
            "seed": config["data"]["seed"],
            "optimized": config["model"]["tflite_optimize"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--model1", type=Path, required=True)
    parser.add_argument(
        "--model2",
        type=Path,
        help="Corrida CAMPIS + Irish con la mitad de los datos. Puede añadirse después de su entrenamiento.",
    )
    parser.add_argument(
        "--model3",
        type=Path,
        help="Corrida solo Irish con la mitad de los datos. Puede agregarse despues de su entrenamiento.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/comparacion-tflite/data.json")
    )
    args = parser.parse_args()

    runs = [
        load_run(
            args.reference,
            "reference",
            "Modelo_COMPLETO",
            "CAMPIS + Irish completos",
        ),
        load_run(
            args.model1,
            "campis",
            "Modelo_CAMPIS",
            "Solo CAMPIS en train",
        ),
    ]
    if args.model3 is not None:
        runs.append(
            load_run(
                args.model3,
                "irish",
                "Modelo_IRISH",
                "Solo Irish completo en train",
            )
        )
    if args.model2 is not None:
        runs.append(
            load_run(
                args.model2,
                "mixed_low",
                "Modelo_MIXTO_LOW",
                "CAMPIS + Irish al 50 % en train",
            )
        )
    signatures = {run["test_signature"] for run in runs}
    sample_counts = {run["test_samples"] for run in runs}
    if len(signatures) != 1 or len(sample_counts) != 1:
        raise RuntimeError("Las corridas no comparten exactamente el mismo test.")

    document = {
        "schema_version": 1,
        "title": "CAMPIS · Comparación TFLite",
        "methodology": {
            "common_test": True,
            "test_samples": runs[0]["test_samples"],
            "test_signature_sha256": runs[0]["test_signature"],
            "note": (
                "Los modelos disponibles se evaluaron con exactamente las mismas imágenes. "
                "La latencia es una referencia reproducible en la CPU del equipo de "
                "entrenamiento, no una medición en un teléfono físico."
            ),
        },
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Escrito: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
