"""Servidor local del monitor de entrenamientos CAMPIS en curso."""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "docs" / "monitor-entrenamiento"
RUNS = ROOT / "artifacts" / "runs"


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def load_csv(path: Path, phase: str) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return []
    result = []
    for row in rows:
        clean: dict[str, Any] = {"phase": phase}
        for key, value in row.items():
            if value in (None, ""):
                clean[key] = None
            else:
                try:
                    clean[key] = float(value)
                except ValueError:
                    clean[key] = value
        result.append(clean)
    return result


def run_status(path: Path) -> dict[str, Any]:
    head = load_csv(path / "history_head.csv", "head")
    fine = load_csv(path / "history_fine_tune.csv", "fine_tune")
    history = head + fine
    metrics = load_json(path / "evaluation" / "metrics.json")
    parity = load_json(path / "tflite_parity.json")
    summary = load_json(path / "training_summary.json")
    manifest = load_json(path / "manifest_summary.json")
    model_exists = (path / "model.keras").is_file()
    tflite_exists = (path / "model.tflite").is_file()
    if tflite_exists:
        state = "exportado"
    elif model_exists:
        state = "evaluando / convirtiendo"
    elif fine:
        state = "fine-tuning"
    else:
        state = "entrenando cabeza"
    return {
        "name": path.name,
        "label": (
            "Modelo 1 · solo CAMPIS" if "modelo1" in path.name
            else "Modelo 2 · CAMPIS + Irish 50 %" if "modelo2" in path.name
            else "Último modelo · referencia"
        ),
        "state": state,
        "history": history,
        "epochs_complete": len(history),
        "phase": history[-1]["phase"] if history else "head",
        "best_phase": summary.get("best_phase"),
        "train_samples": manifest.get("by_split", {}).get("train"),
        "metrics": metrics,
        "parity": parity,
        "model_exists": model_exists,
        "tflite_exists": tflite_exists,
    }


def status_document() -> dict[str, Any]:
    candidates = []
    for pattern in (
        "*sin-sesiones",
        "*modelo1-solo-campis",
        "*modelo2-mitad-campis-irish",
        "*modelo3-solo-irish-50",
    ):
        matches = sorted(RUNS.glob(pattern), reverse=True)
        if matches:
            candidates.append(matches[0])
    return {"runs": [run_status(path) for path in candidates]}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - API de BaseHTTPRequestHandler
        route = urlparse(self.path).path
        if route == "/api/status":
            payload = json.dumps(status_document(), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        relative = "index.html" if route in ("", "/") else unquote(route.lstrip("/"))
        target = (WEB / relative).resolve()
        try:
            target.relative_to(WEB.resolve())
        except ValueError:
            self.send_error(403)
            return
        if not target.is_file():
            self.send_error(404)
            return
        payload = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Monitor: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
