"""Shared helpers: seeding, device selection, checkpoint I/O, and a lightweight CSV metrics logger."""

import csv
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def save_checkpoint(path: Path, model, optimizer, epoch: int, val_acc: float, config_dict: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc,
            "config": config_dict,
        },
        path,
    )


class MetricsLogger:
    """Appends per-epoch metrics to experiments/<model_name>_<timestamp>/metrics.csv."""

    def __init__(self, experiment_dir: str, model_name: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(experiment_dir) / f"{model_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.csv"
        self._header_written = False

    def log(self, **metrics: Any) -> None:
        write_header = not self._header_written and not self.metrics_path.exists()
        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            if write_header:
                writer.writeheader()
                self._header_written = True
            writer.writerow(metrics)
