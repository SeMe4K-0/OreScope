from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path = "config/config.yaml") -> DictConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return OmegaConf.create(raw)


def setup_logging(cfg: DictConfig) -> logging.Logger:
    level = getattr(logging, cfg.logging.level.upper(), logging.INFO)
    fmt = cfg.logging.format
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    log_file = Path(cfg.logging.file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
    return logging.getLogger("materials_analysis")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def colorize_mask(
    mask: np.ndarray,
    colors: list[list[int]],
) -> np.ndarray:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, color in enumerate(colors):
        rgb[mask == class_idx] = color
    return rgb


def overlay_mask(
    image: np.ndarray,
    mask_rgb: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    return (image * (1 - alpha) + mask_rgb * alpha).astype(np.uint8)


def normalize_image(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx - mn > 1e-8:
        img = (img - mn) / (mx - mn)
    return img


def to_tensor(image: np.ndarray) -> torch.Tensor:
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
    return tensor


def save_dict_as_yaml(data: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
