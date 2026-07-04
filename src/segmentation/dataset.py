from __future__ import annotations

import logging
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger(__name__)


class SEMSegmentationDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        transform: A.Compose | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

        self.image_paths = sorted(
            p for p in self.image_dir.glob("*")
            if p.suffix.lower() in {".png", ".jpg", ".tif", ".tiff", ".bmp"}
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

        logger.info("Dataset: %d images from %s", len(self.image_paths), self.image_dir)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        mask_path = self.mask_dir / (img_path.stem + ".png")

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].long()
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return {"image": image, "mask": mask, "path": str(img_path)}


def build_dataloaders(
    cfg,
    train_transform: A.Compose,
    valid_transform: A.Compose,
) -> tuple[DataLoader, DataLoader]:
    full_dataset = SEMSegmentationDataset(
        image_dir="data/processed/images",
        mask_dir="data/processed/masks",
        transform=None,
    )
    n_total = len(full_dataset)
    n_train = int(n_total * cfg.data.train_split)
    n_valid = n_total - n_train
    train_ds, valid_ds = random_split(
        full_dataset,
        [n_train, n_valid],
        generator=torch.Generator().manual_seed(42),
    )

    train_ds.dataset.transform = train_transform
    valid_ds.dataset.transform = valid_transform

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    return train_loader, valid_loader
