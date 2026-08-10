"""Generic training entrypoint, parameterized entirely by --config. Adding a new architecture
never requires touching this file: add a models/<arch>.py + configs/<arch>.yaml instead.
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

from potato_disease.config import Config
from potato_disease.data.dataset import get_dataloaders
from potato_disease.models import get_model
from potato_disease.utils import MetricsLogger, resolve_device, save_checkpoint, set_seed


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> tuple[float, float]:
    model.train() if train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def train(cfg: Config) -> None:
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)

    train_loader, val_loader, _test_loader, classes = get_dataloaders(cfg)

    model = get_model(
        cfg.model.name,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
        freeze_backbone=cfg.model.freeze_backbone,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    logger = MetricsLogger(cfg.output.experiment_dir, cfg.model.name)
    best_val_acc = 0.0
    best_ckpt = Path(cfg.output.checkpoint_dir) / cfg.model.name / logger.run_dir.name / "best.pt"
    last_ckpt = Path(cfg.output.checkpoint_dir) / cfg.model.name / logger.run_dir.name / "last.pt"

    for epoch in range(1, cfg.train.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        print(
            f"[{cfg.model.name}] epoch {epoch}/{cfg.train.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        logger.log(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
        )

        save_checkpoint(last_ckpt, model, optimizer, epoch, val_acc, asdict(cfg))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(best_ckpt, model, optimizer, epoch, val_acc, asdict(cfg))

    print(f"Clases: {classes}")
    print(f"Mejor val_acc: {best_val_acc:.4f} -> {best_ckpt}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Ruta a configs/<arquitectura>.yaml")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
