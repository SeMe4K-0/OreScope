"""I/O helpers: чтение/запись изображений с кириллическими путями (Windows)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread_rgb(path: str | Path) -> np.ndarray:
    """Читает изображение в RGB uint8; работает с не-ASCII путями."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot decode image: {path}")
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:
        img = img[..., :3]
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imwrite(path: str | Path, img: np.ndarray) -> None:
    """Пишет RGB/gray uint8; работает с не-ASCII путями."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(path.suffix if path.suffix else ".png", img)
    if not ok:
        raise ValueError(f"Cannot encode image for {path}")
    buf.tofile(str(path))
