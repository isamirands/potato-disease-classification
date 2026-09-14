"""Crea manifests comparables a partir de una corrida de referencia.

No copia, mueve ni modifica imágenes. Val y test se conservan byte por byte;
solo se filtran las filas de entrenamiento.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Iterable[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def stable_key(seed: int, *parts: str) -> str:
    payload = "|".join((str(seed), *parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def half_by_source_class(
    train_rows: list[dict[str, str]], seed: int
) -> list[dict[str, str]]:
    """Retiene ~50 % por fuente/clase sin separar grupos de captura."""

    buckets: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in train_rows:
        buckets[(row["source"], row["class_name"])].append(row)

    selected: list[dict[str, str]] = []
    for (source, class_name), rows in sorted(buckets.items()):
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            group = row.get("group_id") or row.get("sha256") or row["relative_path"]
            groups[group].append(row)
        ordered = sorted(
            groups.items(),
            key=lambda item: stable_key(seed, source, class_name, item[0]),
        )
        target = math.ceil(len(rows) * 0.5)
        chosen: list[dict[str, str]] = []
        for _group, group_rows in ordered:
            current_gap = abs(target - len(chosen))
            next_gap = abs(target - (len(chosen) + len(group_rows)))
            if len(chosen) < target and (next_gap <= current_gap or not chosen):
                chosen.extend(group_rows)
        selected.extend(chosen)
    selected_paths = {row["relative_path"] for row in selected}
    return [row for row in train_rows if row["relative_path"] in selected_paths]


def signature(rows: Iterable[dict[str, str]]) -> str:
    paths = sorted(row["relative_path"] for row in rows)
    return hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()


def counts(rows: list[dict[str, str]]) -> dict[str, object]:
    by_split = Counter(row["split"] for row in rows)
    by_source = Counter(row["source"] for row in rows)
    by_group = Counter(
        f'{row["split"]}/{row["source"]}/{row["class_name"]}' for row in rows
    )
    return {
        "total": len(rows),
        "by_split": dict(sorted(by_split.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_split_source_class": dict(sorted(by_group.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/manifests"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = args.reference_run / "manifest.csv"
    rows = read_rows(source)
    if not rows:
        raise SystemExit(f"Manifest vacío: {source}")
    fields = list(rows[0])
    held_out = [row for row in rows if row["split"] in {"val", "test"}]
    train = [row for row in rows if row["split"] == "train"]

    model1_train = [row for row in train if row["source"] == "campis"]
    model2_train = half_by_source_class(train, args.seed)
    model3_train = [row for row in train if row["source"] == "irish"]
    variants = {
        "modelo1_solo_campis.csv": model1_train + held_out,
        "modelo2_mitad_campis_irish.csv": model2_train + held_out,
        "modelo3_solo_irish.csv": model3_train + held_out,
    }
    for name, variant in variants.items():
        # Mantiene el orden original para evaluación y trazabilidad.
        wanted = {row["relative_path"] for row in variant}
        ordered = [row for row in rows if row["relative_path"] in wanted]
        write_rows(args.output_dir / name, ordered, fields)

    reference_test = [row for row in rows if row["split"] == "test"]
    report = {
        "reference_run": str(args.reference_run.resolve()),
        "seed": args.seed,
        "policy": {
            "validation": "idéntica a la referencia",
            "test": "idéntico a la referencia",
            "model1_train": "solo fuente campis",
            "model2_train": "50 % determinista por fuente/clase, grupos indivisibles",
            "model3_train": "solo fuente Irish completa",
        },
        "common_test_signature_sha256": signature(reference_test),
        "reference": counts(rows),
        "model1": counts(variants["modelo1_solo_campis.csv"]),
        "model2": counts(variants["modelo2_mitad_campis_irish.csv"]),
        "model3": counts(variants["modelo3_solo_irish.csv"]),
    }
    report_path = args.output_dir / "protocolo_comparacion.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
