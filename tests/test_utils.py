from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # Permite ``unittest discover -s tests``.
    import _bootstrap  # type: ignore[no-redef]  # noqa: F401

from campis.utils import (
    atomic_write_csv,
    atomic_write_json,
    create_run_dir,
    read_csv,
    read_json,
    run_timestamp,
)


class _Status(Enum):
    READY = "ready"


@dataclass(frozen=True)
class _Payload:
    path: Path
    status: _Status


class UtilsTests(unittest.TestCase):
    def test_json_atomic_round_trip_supports_project_types(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "nested" / "payload.json"
            payload = _Payload(path=Path("images/leaf.jpg"), status=_Status.READY)

            returned = atomic_write_json(destination, payload)

            self.assertEqual(returned, destination.resolve())
            self.assertEqual(
                read_json(destination),
                {"path": str(Path("images/leaf.jpg")), "status": "ready"},
            )
            self.assertEqual(list(destination.parent.glob(f".{destination.name}.*.tmp")), [])

    def test_csv_atomic_round_trip_and_empty_input_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "tables" / "records.csv"
            rows = [
                {"class": "healthy", "count": 2},
                {"class": "late_blight", "count": 1},
            ]

            atomic_write_csv(destination, rows)

            self.assertEqual(
                read_csv(destination),
                [
                    {"class": "healthy", "count": "2"},
                    {"class": "late_blight", "count": "1"},
                ],
            )
            with self.assertRaisesRegex(ValueError, "fieldnames"):
                atomic_write_csv(Path(directory) / "empty.csv", [])

    def test_timestamp_is_utc_and_run_directories_are_unique(self) -> None:
        local_time = datetime(2026, 8, 16, 20, 30, tzinfo=timezone(timedelta(hours=-5)))
        with tempfile.TemporaryDirectory() as directory:
            first = create_run_dir(directory, name="Prueba CAMPIS", now=local_time)
            second = create_run_dir(directory, name="Prueba CAMPIS", now=local_time)

            self.assertEqual(run_timestamp(local_time), "20260817T013000Z")
            self.assertEqual(first.name, "20260817T013000Z-prueba-campis")
            self.assertEqual(second.name, "20260817T013000Z-prueba-campis-01")
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())


if __name__ == "__main__":
    unittest.main()
