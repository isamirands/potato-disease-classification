"""Evalúa un modelo TFLite sobre todo el test de su propia corrida."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campis.data import build_tf_dataset, read_manifest_file, records_for_split
from campis.evaluation import compute_metrics, save_evaluation_artifacts
from campis.exporting import create_tflite_interpreter, predict_tflite


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _keras_predictions(run: Path) -> dict[str, str]:
    path = run / "evaluation" / "predictions.csv"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            row["relative_path"]: row["predicted_class"]
            for row in csv.DictReader(handle)
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    run = args.run.resolve()
    config = json.loads((run / "config.resolved.json").read_text(encoding="utf-8"))
    records = records_for_split(
        read_manifest_file(run / "manifest.csv", require_all_splits=False), "test"
    )
    labels_document = json.loads((run / "labels.json").read_text(encoding="utf-8"))
    classes = labels_document.get("labels", labels_document)
    project_root = Path(config["paths"]["project_root"])
    image_size = tuple(config["model"]["image_size"])
    dataset = build_tf_dataset(
        records,
        project_root,
        image_size,
        args.batch_size,
        training=False,
        seed=int(config["data"]["seed"]),
        num_workers=int(config["training"]["num_workers"]),
        crop_to_square=bool(config["model"].get("crop_to_square", False)),
    )

    model_path = run / "model.tflite"
    interpreter = create_tflite_interpreter(model_path, num_threads=args.threads)
    probabilities: list[np.ndarray] = []
    inference_seconds = 0.0
    wall_start = time.perf_counter()
    for index, (images, _labels) in enumerate(dataset):
        batch = images.numpy()
        if index == 0:
            predict_tflite(interpreter, batch[:1])  # calentamiento fuera de medida
        started = time.perf_counter()
        probabilities.append(predict_tflite(interpreter, batch))
        inference_seconds += time.perf_counter() - started
        done = sum(len(item) for item in probabilities)
        if done % 512 < len(batch):
            print(f"{run.name}: {done}/{len(records)}", flush=True)
    wall_seconds = time.perf_counter() - wall_start
    output = np.concatenate(probabilities, axis=0)
    truth = [record.class_index for record in records]
    sources = [record.source for record in records]
    metrics = compute_metrics(truth, output, sources, classes)

    keras = _keras_predictions(run)
    tflite_labels = [classes[index] for index in output.argmax(axis=1)]
    compared_by_source: dict[str, int] = {}
    flips_by_source: dict[str, int] = {}
    for record, predicted in zip(records, tflite_labels):
        keras_prediction = keras.get(record.relative_path)
        if keras_prediction is None:
            continue
        compared_by_source[record.source] = compared_by_source.get(record.source, 0) + 1
        if keras_prediction != predicted:
            flips_by_source[record.source] = flips_by_source.get(record.source, 0) + 1
    comparable = [
        keras.get(record.relative_path) == predicted
        for record, predicted in zip(records, tflite_labels)
        if record.relative_path in keras
    ]
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    metrics["runtime"] = {
        "engine": "TensorFlow Lite Interpreter",
        "host_note": "Referencia en CPU del equipo de entrenamiento; no es latencia de teléfono.",
        "threads": args.threads,
        "warmup_samples": 1,
        "inference_seconds": inference_seconds,
        "wall_seconds_including_decode": wall_seconds,
        "mean_inference_ms_per_image": inference_seconds * 1000 / len(records),
        "images_per_second": len(records) / inference_seconds,
        "model_bytes": model_path.stat().st_size,
        "model_sha256": _sha256(model_path),
        "input_dtype": np.dtype(input_detail["dtype"]).name,
        "output_dtype": np.dtype(output_detail["dtype"]).name,
        "input_shape": [int(value) for value in input_detail["shape_signature"]],
    }
    metrics["keras_tflite_agreement"] = {
        "samples": len(comparable),
        "class_agreement_rate": float(np.mean(comparable)) if comparable else None,
        "class_flips": int(len(comparable) - sum(comparable)),
        "by_source": {
            source: {
                "samples": samples,
                "class_flips": flips_by_source.get(source, 0),
                "class_flip_rate": (
                    flips_by_source.get(source, 0) / samples if samples else 0.0
                ),
            }
            for source, samples in sorted(compared_by_source.items())
        },
    }
    output_dir = run / "evaluation_tflite"
    save_evaluation_artifacts(metrics, output, records, classes, output_dir)
    print(json.dumps({
        "run": run.name,
        "accuracy": metrics["global"]["accuracy"],
        "macro_f1": metrics["global"]["macro_f1"],
        "campis_accuracy": metrics["by_source"]["campis"]["accuracy"],
        "irish_accuracy": metrics["by_source"]["irish"]["accuracy"],
        "mean_ms": metrics["runtime"]["mean_inference_ms_per_image"],
        "agreement": metrics["keras_tflite_agreement"]["class_agreement_rate"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
