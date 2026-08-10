"""Generic evaluation entrypoint: loads a checkpoint and reports test-set metrics."""

import argparse

import torch
from sklearn.metrics import classification_report, confusion_matrix

from potato_disease.config import Config
from potato_disease.data.dataset import get_dataloaders
from potato_disease.models import get_model
from potato_disease.utils import resolve_device


def evaluate(cfg: Config, checkpoint_path: str) -> None:
    device = resolve_device(cfg.train.device)
    _train_loader, _val_loader, test_loader, classes = get_dataloaders(cfg)

    model = get_model(
        cfg.model.name,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
        freeze_backbone=cfg.model.freeze_backbone,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            all_preds.extend(outputs.argmax(1).cpu().tolist())
            all_labels.extend(labels.tolist())

    print(classification_report(all_labels, all_preds, target_names=classes))
    print("Matriz de confusión:")
    print(confusion_matrix(all_labels, all_preds))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Ruta a configs/<arquitectura>.yaml")
    parser.add_argument("--checkpoint", required=True, help="Ruta al checkpoint .pt a evaluar")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    evaluate(cfg, args.checkpoint)


if __name__ == "__main__":
    main()
