"""CNN-классификатор срастаний: карта вероятности «тонких» по тайлам.

resnet18, обученный на weak labels (train_grain_cnn.py). Пайплайн использует
карту для переклассификации зёрен морфологического этапа: класс зерна =
порог по средней вероятности тайлов, накрывающих зерно.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class GrainCNN:
    def __init__(self, checkpoint: str | Path, device: torch.device | None = None):
        import torchvision

        self.device = device or (torch.device("cuda") if torch.cuda.is_available()
                                 else torch.device("cpu"))
        self.model = None
        ckpt = Path(checkpoint)
        if not ckpt.exists():
            logger.warning("grain_cnn чекпойнт не найден: %s — используется морфология", ckpt)
            return
        model = torchvision.models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load(ckpt, map_location=self.device, weights_only=True))
        model.to(self.device).eval()
        self.model = model
        logger.info("GrainCNN загружен: %s", ckpt)

    @property
    def available(self) -> bool:
        return self.model is not None

    @torch.no_grad()
    def prob_map(self, rgb: np.ndarray, tile: int = 512, batch: int = 64) -> np.ndarray:
        """Карта P(тонкие срастания) HxW (float32), собранная по тайлам."""
        H, W = rgb.shape[:2]
        stride = tile // 2
        ys = list(range(0, max(H - tile, 0) + 1, stride)) or [0]
        xs = list(range(0, max(W - tile, 0) + 1, stride)) or [0]
        if ys[-1] != max(H - tile, 0):
            ys.append(max(H - tile, 0))
        if xs[-1] != max(W - tile, 0):
            xs.append(max(W - tile, 0))

        grid = np.zeros((len(ys), len(xs)), np.float32)
        coords, tiles = [], []

        def flush() -> None:
            if not tiles:
                return
            arr = np.stack(tiles).astype(np.float32) / 255.0
            arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
            t = torch.from_numpy(arr.transpose(0, 3, 1, 2)).to(self.device)
            p = torch.softmax(self.model(t), dim=1)[:, 1].cpu().numpy()
            for prob, (iy, ix) in zip(p, coords):
                grid[iy, ix] = prob
            coords.clear()
            tiles.clear()

        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                patch = rgb[y:min(y + tile, H), x:min(x + tile, W)]
                if patch.shape[0] < tile or patch.shape[1] < tile:
                    padded = np.zeros((tile, tile, 3), np.uint8)
                    padded[:patch.shape[0], :patch.shape[1]] = patch
                    patch = padded
                tiles.append(cv2.resize(patch, (224, 224), interpolation=cv2.INTER_AREA))
                coords.append((iy, ix))
                if len(tiles) == batch:
                    flush()
        flush()
        return cv2.resize(grid, (W, H), interpolation=cv2.INTER_LINEAR)
