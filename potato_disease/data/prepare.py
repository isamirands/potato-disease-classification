"""CLI: split data/raw/<class>/* into data/processed/{train,val,test}/<class>/* (ImageFolder layout)."""

import argparse
import random
import shutil
from pathlib import Path


def split_dataset(
    raw_dir: Path,
    processed_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    rng = random.Random(seed)
    class_dirs = [d for d in sorted(raw_dir.iterdir()) if d.is_dir()]
    if not class_dirs:
        raise FileNotFoundError(
            f"No se encontraron carpetas de clase en {raw_dir}. Ver data/README.md."
        )

    for class_dir in class_dirs:
        images = [p for p in sorted(class_dir.iterdir()) if p.is_file()]
        rng.shuffle(images)

        n_train = int(len(images) * train_ratio)
        n_val = int(len(images) * val_ratio)
        splits = {
            "train": images[:n_train],
            "val": images[n_train : n_train + n_val],
            "test": images[n_train + n_val :],
        }

        for split_name, split_images in splits.items():
            out_dir = processed_dir / split_name / class_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for img_path in split_images:
                shutil.copy2(img_path, out_dir / img_path.name)

        print(f"{class_dir.name}: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_dataset(
        Path(args.raw_dir),
        Path(args.processed_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
