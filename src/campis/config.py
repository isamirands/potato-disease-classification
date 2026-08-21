"""Configuracion tipada y validada del pipeline CAMPIS.

Las rutas declaradas en TOML son portables: ``project_root`` se interpreta
respecto al directorio del archivo de configuracion y el resto de rutas se
interpreta respecto a ``project_root``.
"""

from __future__ import annotations

import math
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


_SPLIT_NAMES = ("train", "val", "test")
_BALANCE_STRATEGIES = {"none", "class", "source", "source_class"}
_TF_DECODE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


@dataclass(frozen=True, slots=True)
class SourceConfig:
    """Un dominio de adquisicion de imagenes."""

    name: str
    path: Path
    camera_type: str


@dataclass(frozen=True, slots=True)
class PathsConfig:
    """Rutas principales ya resueltas a valores absolutos."""

    project_root: Path
    artifacts_dir: Path
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Contrato, particionado y balance del dataset."""

    classes: tuple[str, ...]
    extensions: tuple[str, ...]
    seed: int
    ratios: dict[str, float]
    verify_images: bool
    balance_strategy: str
    samples_per_source_class: int
    sources: tuple[SourceConfig, ...]
    hash_duplicates: bool = True
    group_campis_by_minutes: int = 5
    group_sequence_gap: int = 8
    max_images_per_source_class: int = 0


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Arquitectura y transferencia de aprendizaje."""

    image_size: tuple[int, int]
    dense_units: tuple[int, ...]
    dropout: float
    weights: str | None
    fine_tune_fraction: float
    tflite_optimize: bool
    crop_to_square: bool = True


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Hiperparametros del ajuste de cabeza y fine-tuning."""

    batch_size: int
    head_epochs: int
    fine_tune_epochs: int
    head_lr: float
    fine_tune_lr: float
    patience: int
    num_workers: int


@dataclass(frozen=True, slots=True)
class AugmentationConfig:
    """Aumento de datos aplicado exclusivamente al split de entrenamiento."""

    flip: bool
    rotation: float
    translation: float
    zoom: float
    contrast: float


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Configuracion completa de una ejecucion."""

    paths: PathsConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    augmentation: AugmentationConfig


def _table(value: Any, section: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"La seccion [{section}] debe ser una tabla TOML.")
    return value


def _reject_unknown(
    table: Mapping[str, Any], allowed: set[str], section: str
) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ValueError(
            f"Claves desconocidas en [{section}]: {', '.join(unknown)}"
        )


def _required(table: Mapping[str, Any], key: str, section: str) -> Any:
    if key not in table:
        raise ValueError(f"Falta la clave obligatoria [{section}].{key}.")
    return table[key]


def _non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} debe ser texto no vacio.")
    return value.strip()


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} debe ser true o false.")
    return value


def _integer(value: Any, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} debe ser un entero >= {minimum}.")
    return value


def _number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    maximum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} debe ser numerico.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} debe ser finito.")
    if minimum is not None and number < minimum:
        raise ValueError(f"{label} debe ser >= {minimum}.")
    if maximum is not None:
        invalid = number > maximum if maximum_inclusive else number >= maximum
        if invalid:
            operator = "<=" if maximum_inclusive else "<"
            raise ValueError(f"{label} debe ser {operator} {maximum}.")
    return number


def _string_sequence(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} debe ser una lista TOML no vacia.")
    result = tuple(_non_empty_string(item, f"{label}[{index}]") for index, item in enumerate(value))
    folded = [item.casefold() for item in result]
    if len(set(folded)) != len(folded):
        raise ValueError(f"{label} contiene valores duplicados.")
    return result


def _integer_sequence(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} debe ser una lista TOML no vacia.")
    return tuple(
        _integer(item, f"{label}[{index}]", minimum=1)
        for index, item in enumerate(value)
    )


def _resolve_path(base: Path, value: Any, label: str) -> Path:
    raw = _non_empty_string(value, label)
    expanded = Path(os.path.expandvars(raw)).expanduser()
    if not expanded.is_absolute():
        expanded = base / expanded
    return expanded.resolve(strict=False)


def _parse_paths(raw: Mapping[str, Any], config_dir: Path) -> PathsConfig:
    _reject_unknown(raw, {"project_root", "artifacts_dir", "manifest_path"}, "paths")
    project_root = _resolve_path(
        config_dir, _required(raw, "project_root", "paths"), "paths.project_root"
    )
    if not project_root.is_dir():
        raise FileNotFoundError(f"project_root no existe o no es directorio: {project_root}")

    artifacts_dir = _resolve_path(
        project_root,
        _required(raw, "artifacts_dir", "paths"),
        "paths.artifacts_dir",
    )
    if artifacts_dir.exists() and not artifacts_dir.is_dir():
        raise ValueError(f"artifacts_dir existe pero no es directorio: {artifacts_dir}")

    manifest_path = _resolve_path(
        project_root,
        _required(raw, "manifest_path", "paths"),
        "paths.manifest_path",
    )
    if manifest_path.exists() and not manifest_path.is_file():
        raise ValueError(f"manifest_path existe pero no es archivo: {manifest_path}")
    if manifest_path.suffix.casefold() != ".csv":
        raise ValueError("paths.manifest_path debe terminar en .csv.")

    return PathsConfig(
        project_root=project_root,
        artifacts_dir=artifacts_dir,
        manifest_path=manifest_path,
    )


def _parse_sources(value: Any, project_root: Path) -> tuple[SourceConfig, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("data.sources debe contener al menos una fuente.")

    sources: list[SourceConfig] = []
    for index, source_value in enumerate(value):
        section = f"data.sources[{index}]"
        source = _table(source_value, section)
        _reject_unknown(source, {"name", "path", "camera_type"}, section)
        name = _non_empty_string(_required(source, "name", section), f"{section}.name")
        camera_type = _non_empty_string(
            _required(source, "camera_type", section), f"{section}.camera_type"
        )
        path = _resolve_path(
            project_root, _required(source, "path", section), f"{section}.path"
        )
        if not path.is_dir():
            raise FileNotFoundError(f"La fuente '{name}' no existe o no es directorio: {path}")
        sources.append(SourceConfig(name=name, path=path, camera_type=camera_type))

    names = [source.name.casefold() for source in sources]
    if len(set(names)) != len(names):
        raise ValueError("Los nombres de data.sources deben ser unicos.")
    paths = [os.path.normcase(str(source.path)) for source in sources]
    if len(set(paths)) != len(paths):
        raise ValueError("Cada data.sources.path debe apuntar a un directorio distinto.")
    return tuple(sources)


def _parse_ratios(value: Any) -> dict[str, float]:
    ratios_table = _table(value, "data.ratios")
    _reject_unknown(ratios_table, set(_SPLIT_NAMES), "data.ratios")
    missing = [name for name in _SPLIT_NAMES if name not in ratios_table]
    if missing:
        raise ValueError(f"Faltan ratios para: {', '.join(missing)}")
    ratios = {
        name: _number(
            ratios_table[name], f"data.ratios.{name}", minimum=0.0, maximum=1.0
        )
        for name in _SPLIT_NAMES
    }
    if any(value == 0.0 for value in ratios.values()):
        raise ValueError("Los ratios train, val y test deben ser mayores que cero.")
    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Los ratios train, val y test deben sumar 1.0.")
    return ratios


def _parse_data(raw: Mapping[str, Any], project_root: Path) -> DataConfig:
    _reject_unknown(
        raw,
        {
            "classes",
            "extensions",
            "seed",
            "ratios",
            "verify_images",
            "balance_strategy",
            "samples_per_source_class",
            "sources",
            "hash_duplicates",
            "group_campis_by_minutes",
            "group_sequence_gap",
            "max_images_per_source_class",
        },
        "data",
    )
    classes = _string_sequence(_required(raw, "classes", "data"), "data.classes")
    if len(classes) < 2:
        raise ValueError("data.classes debe contener al menos dos clases.")
    for class_name in classes:
        if Path(class_name).name != class_name or class_name in {".", ".."}:
            raise ValueError(f"Nombre de clase no valido como directorio: {class_name!r}")

    raw_extensions = _string_sequence(
        _required(raw, "extensions", "data"), "data.extensions"
    )
    extensions = tuple(
        extension.casefold() if extension.startswith(".") else f".{extension.casefold()}"
        for extension in raw_extensions
    )
    if len(set(extensions)) != len(extensions):
        raise ValueError("data.extensions contiene extensiones equivalentes duplicadas.")
    unsupported_extensions = sorted(set(extensions) - _TF_DECODE_EXTENSIONS)
    if unsupported_extensions:
        raise ValueError(
            "TensorFlow no decodifica estas data.extensions en el pipeline actual: "
            + ", ".join(unsupported_extensions)
        )

    sources = _parse_sources(_required(raw, "sources", "data"), project_root)
    for source in sources:
        missing_classes = [
            class_name for class_name in classes if not (source.path / class_name).is_dir()
        ]
        if missing_classes:
            raise FileNotFoundError(
                f"La fuente '{source.name}' no contiene estas clases: "
                f"{', '.join(missing_classes)}"
            )

    strategy = _non_empty_string(
        _required(raw, "balance_strategy", "data"), "data.balance_strategy"
    ).casefold()
    if strategy not in _BALANCE_STRATEGIES:
        allowed = ", ".join(sorted(_BALANCE_STRATEGIES))
        raise ValueError(f"data.balance_strategy debe ser uno de: {allowed}.")

    return DataConfig(
        classes=classes,
        extensions=extensions,
        seed=_integer(raw.get("seed", 42), "data.seed", minimum=0),
        ratios=_parse_ratios(_required(raw, "ratios", "data")),
        verify_images=_boolean(
            _required(raw, "verify_images", "data"), "data.verify_images"
        ),
        balance_strategy=strategy,
        samples_per_source_class=_integer(
            _required(raw, "samples_per_source_class", "data"),
            "data.samples_per_source_class",
            minimum=1,
        ),
        sources=sources,
        hash_duplicates=_boolean(
            raw.get("hash_duplicates", True), "data.hash_duplicates"
        ),
        group_campis_by_minutes=_integer(
            raw.get("group_campis_by_minutes", 5),
            "data.group_campis_by_minutes",
            minimum=0,
        ),
        group_sequence_gap=_integer(
            raw.get("group_sequence_gap", 8),
            "data.group_sequence_gap",
            minimum=0,
        ),
        max_images_per_source_class=_integer(
            raw.get("max_images_per_source_class", 0),
            "data.max_images_per_source_class",
            minimum=0,
        ),
    )


def _parse_model(raw: Mapping[str, Any]) -> ModelConfig:
    _reject_unknown(
        raw,
        {
            "image_size",
            "dense_units",
            "dropout",
            "weights",
            "fine_tune_fraction",
            "tflite_optimize",
            "crop_to_square",
        },
        "model",
    )
    image_size = _integer_sequence(
        _required(raw, "image_size", "model"), "model.image_size"
    )
    if len(image_size) != 2:
        raise ValueError("model.image_size debe contener exactamente [alto, ancho].")
    if any(dimension < 32 for dimension in image_size):
        raise ValueError("MobileNetV3 requiere model.image_size >= [32, 32].")

    raw_weights = raw.get("weights", "imagenet")
    if raw_weights is None:
        weights = None
    elif isinstance(raw_weights, str):
        normalized_weights = raw_weights.strip().casefold()
        weights = None if normalized_weights in {"", "none", "null"} else raw_weights.strip()
    else:
        raise ValueError("model.weights debe ser texto, una cadena vacia o 'none'.")
    if weights is not None and weights.casefold() != "imagenet":
        raise ValueError("model.weights solo admite 'imagenet', 'none' o una cadena vacia.")

    return ModelConfig(
        image_size=(image_size[0], image_size[1]),
        dense_units=_integer_sequence(
            _required(raw, "dense_units", "model"), "model.dense_units"
        ),
        dropout=_number(
            _required(raw, "dropout", "model"),
            "model.dropout",
            minimum=0.0,
            maximum=1.0,
            maximum_inclusive=False,
        ),
        weights=weights,
        fine_tune_fraction=_number(
            _required(raw, "fine_tune_fraction", "model"),
            "model.fine_tune_fraction",
            minimum=0.0,
            maximum=1.0,
        ),
        tflite_optimize=_boolean(
            raw.get("tflite_optimize", True), "model.tflite_optimize"
        ),
        crop_to_square=_boolean(
            raw.get("crop_to_square", True), "model.crop_to_square"
        ),
    )


def _parse_training(raw: Mapping[str, Any]) -> TrainingConfig:
    _reject_unknown(
        raw,
        {
            "batch_size",
            "head_epochs",
            "fine_tune_epochs",
            "head_lr",
            "fine_tune_lr",
            "patience",
            "num_workers",
        },
        "training",
    )
    head_epochs = _integer(
        _required(raw, "head_epochs", "training"), "training.head_epochs", minimum=1
    )
    fine_tune_epochs = _integer(
        _required(raw, "fine_tune_epochs", "training"),
        "training.fine_tune_epochs",
        minimum=0,
    )
    head_lr = _number(
        _required(raw, "head_lr", "training"),
        "training.head_lr",
        minimum=0.0,
    )
    fine_tune_lr = _number(
        _required(raw, "fine_tune_lr", "training"),
        "training.fine_tune_lr",
        minimum=0.0,
    )
    if head_lr == 0.0 or fine_tune_lr == 0.0:
        raise ValueError("training.head_lr y training.fine_tune_lr deben ser mayores que cero.")

    return TrainingConfig(
        batch_size=_integer(
            _required(raw, "batch_size", "training"),
            "training.batch_size",
            minimum=1,
        ),
        head_epochs=head_epochs,
        fine_tune_epochs=fine_tune_epochs,
        head_lr=head_lr,
        fine_tune_lr=fine_tune_lr,
        patience=_integer(
            _required(raw, "patience", "training"),
            "training.patience",
            minimum=0,
        ),
        num_workers=_integer(
            _required(raw, "num_workers", "training"),
            "training.num_workers",
            minimum=0,
        ),
    )


def _parse_augmentation(raw: Mapping[str, Any]) -> AugmentationConfig:
    _reject_unknown(
        raw, {"flip", "rotation", "translation", "zoom", "contrast"}, "augmentation"
    )
    return AugmentationConfig(
        flip=_boolean(_required(raw, "flip", "augmentation"), "augmentation.flip"),
        rotation=_number(
            _required(raw, "rotation", "augmentation"),
            "augmentation.rotation",
            minimum=0.0,
            maximum=1.0,
        ),
        translation=_number(
            _required(raw, "translation", "augmentation"),
            "augmentation.translation",
            minimum=0.0,
            maximum=1.0,
        ),
        zoom=_number(
            _required(raw, "zoom", "augmentation"),
            "augmentation.zoom",
            minimum=0.0,
            maximum=1.0,
        ),
        contrast=_number(
            _required(raw, "contrast", "augmentation"),
            "augmentation.contrast",
            minimum=0.0,
            maximum=1.0,
        ),
    )


def load_config(path: str | Path) -> AppConfig:
    """Carga un TOML, resuelve sus rutas y valida el contrato completo.

    ``artifacts_dir`` y el archivo de manifest no se crean durante la carga y
    pueden no existir aun. En cambio, el proyecto, cada fuente y sus carpetas
    de clase deben existir para fallar temprano ante una ruta equivocada.
    """

    config_path = Path(path).expanduser().resolve(strict=False)
    if not config_path.is_file():
        raise FileNotFoundError(f"No se encontro la configuracion TOML: {config_path}")

    try:
        with config_path.open("rb") as file:
            raw_value = tomllib.load(file)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"TOML invalido en {config_path}: {error}") from error

    raw = _table(raw_value, "raiz")
    _reject_unknown(raw, {"paths", "data", "model", "training", "augmentation"}, "raiz")
    for section in ("paths", "data", "model", "training", "augmentation"):
        if section not in raw:
            raise ValueError(f"Falta la seccion obligatoria [{section}].")

    paths = _parse_paths(_table(raw["paths"], "paths"), config_path.parent)
    data = _parse_data(_table(raw["data"], "data"), paths.project_root)
    model = _parse_model(_table(raw["model"], "model"))
    training = _parse_training(_table(raw["training"], "training"))
    augmentation = _parse_augmentation(
        _table(raw["augmentation"], "augmentation")
    )

    if model.fine_tune_fraction == 0.0 and training.fine_tune_epochs > 0:
        raise ValueError(
            "training.fine_tune_epochs debe ser 0 cuando model.fine_tune_fraction es 0."
        )

    return AppConfig(
        paths=paths,
        data=data,
        model=model,
        training=training,
        augmentation=augmentation,
    )
