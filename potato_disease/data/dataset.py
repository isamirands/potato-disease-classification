"""Dataloader construction from an ImageFolder-structured processed dataset."""

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from potato_disease.config import Config
from potato_disease.data.transforms import get_eval_transforms, get_train_transforms


def get_dataloaders(cfg: Config) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Build train/val/test dataloaders from cfg.data.data_dir/{train,val,test}."""
    data_dir = Path(cfg.data.data_dir)

    train_ds = ImageFolder(data_dir / "train", transform=get_train_transforms(cfg.data.image_size))
    val_ds = ImageFolder(data_dir / "val", transform=get_eval_transforms(cfg.data.image_size))
    test_ds = ImageFolder(data_dir / "test", transform=get_eval_transforms(cfg.data.image_size))

    train_loader = DataLoader(
        train_ds, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers
    )

    return train_loader, val_loader, test_loader, train_ds.classes
