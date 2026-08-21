"""Utilidades pequenas y reutilizables del proyecto CAMPIS."""

from __future__ import annotations

import csv
import importlib
import json
import logging
import os
import random
import re
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


LOGGER_NAME = "campis"
_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(
    level: int | str = logging.INFO,
    *,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Configura una sola vez el logger del paquete y devuelve su instancia."""

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(_LOG_FORMAT)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        destination = Path(log_file).expanduser().resolve(strict=False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(destination, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def setup_logging(
    *, verbose: bool = False, log_file: str | Path | None = None
) -> logging.Logger:
    """Atajo para CLIs: INFO por defecto y DEBUG con ``verbose``."""

    return configure_logging(
        logging.DEBUG if verbose else logging.INFO,
        log_file=log_file,
    )


def get_logger(name: str | None = None) -> logging.Logger:
    """Obtiene el logger raiz de CAMPIS o uno de sus hijos."""

    if not name:
        resolved_name = LOGGER_NAME
    elif name == LOGGER_NAME or name.startswith(f"{LOGGER_NAME}."):
        resolved_name = name
    else:
        resolved_name = f"{LOGGER_NAME}.{name}"
    return logging.getLogger(resolved_name)


def set_global_seed(seed: int, *, deterministic: bool = True) -> None:
    """Fija semillas de Python, NumPy y, si esta instalado, TensorFlow.

    TensorFlow se importa dentro de la funcion para que auditar configuraciones
    o manifests no pague su elevado tiempo de importacion.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed debe ser un entero >= 0.")

    # Solo afecta procesos Python hijos; el intérprete actual eligió su hash
    # seed antes de ejecutar esta función.
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)

    numpy = importlib.import_module("numpy")
    numpy.random.seed(seed)

    try:
        tensorflow = importlib.import_module("tensorflow")
    except ModuleNotFoundError as error:
        if error.name != "tensorflow":
            raise
        get_logger(__name__).debug(
            "TensorFlow no esta instalado; se fijaron solo Python y NumPy."
        )
        return

    tensorflow.keras.utils.set_random_seed(seed)
    if deterministic:
        try:
            tensorflow.config.experimental.enable_op_determinism()
        except (AttributeError, RuntimeError):
            get_logger(__name__).warning(
                "TensorFlow no pudo activar operaciones deterministas."
            )


def _json_default(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"No se puede serializar {type(value).__name__} como JSON.")


def atomic_write_json(
    path: str | Path,
    data: Any,
    *,
    indent: int = 2,
) -> Path:
    """Escribe JSON UTF-8 y lo publica atomicamente con ``os.replace``."""

    destination = Path(path).expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(
                data,
                temporary,
                ensure_ascii=False,
                indent=indent,
                default=_json_default,
            )
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return destination


def read_json(path: str | Path) -> Any:
    """Lee un documento JSON UTF-8."""

    with Path(path).expanduser().open("r", encoding="utf-8") as file:
        return json.load(file)


def atomic_write_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> Path:
    """Escribe registros de diccionarios en CSV mediante reemplazo atomico."""

    iterator = iter(rows)
    try:
        first = dict(next(iterator))
    except StopIteration:
        first = None

    columns = list(fieldnames) if fieldnames is not None else list(first or {})
    if not columns:
        raise ValueError("fieldnames es obligatorio cuando rows esta vacio.")
    if len(set(columns)) != len(columns):
        raise ValueError("fieldnames contiene nombres duplicados.")

    destination = Path(path).expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8-sig",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            writer = csv.DictWriter(temporary, fieldnames=columns, extrasaction="raise")
            writer.writeheader()
            if first is not None:
                writer.writerow(first)
            for row in iterator:
                writer.writerow(dict(row))
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return destination


def read_csv(path: str | Path) -> list[dict[str, str]]:
    """Lee un CSV como una lista de registros; no realiza conversion de tipos."""

    with Path(path).expanduser().open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def run_timestamp(now: datetime | None = None) -> str:
    """Devuelve un identificador UTC ordenable, por ejemplo 20260817T012030Z."""

    instant = now or datetime.now(timezone.utc)
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=timezone.utc)
    return instant.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-_").lower()
    if not slug:
        raise ValueError("El nombre de ejecucion no contiene caracteres validos.")
    return slug


def create_run_dir(
    artifacts_dir: str | Path,
    *,
    name: str | None = None,
    now: datetime | None = None,
) -> Path:
    """Crea un directorio de ejecucion unico bajo ``artifacts_dir/runs``."""

    runs_dir = Path(artifacts_dir).expanduser().resolve(strict=False) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    base_name = run_timestamp(now)
    if name is not None:
        base_name = f"{base_name}-{_slug(name)}"

    candidate = runs_dir / base_name
    suffix = 1
    while True:
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            candidate = runs_dir / f"{base_name}-{suffix:02d}"
            suffix += 1


# Alias legible para codigo que prefiera el verbo "write" al inicio.
write_json_atomic = atomic_write_json
write_csv_atomic = atomic_write_csv
