"""Сегментация талька: U-Net (тайлово) + постобработка talc &= ~sulfide.

Fallback без чекпойнта — текстурная эвристика (тёмная рассеянная фаза:
умеренно тёмные пиксели матрицы с высокой локальной плотностью мелких
тёмных включений); хуже U-Net, но не требует обучения.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import DictConfig

from src.segmentation.model import load_model
from src.sulfides import filter_small
from src.tiling import predict_tiled

logger = logging.getLogger(__name__)


@dataclass
class TalcResult:
    mask: np.ndarray          # bool
    probability: np.ndarray   # float32 0..1 (карта уверенности = вероятность талька)
    mean_confidence: float    # средняя уверенность решения (0..1)
    used_model: str           # "unet" | "texture-fallback"


class TalcSegmenter:
    def __init__(self, cfg: DictConfig, checkpoint: str | Path = "models/talc_best.pt",
                 device: torch.device | None = None):
        self.cfg = cfg
        self.device = device or (torch.device("cuda") if torch.cuda.is_available()
                                 else torch.device("cpu"))
        self.model = None
        ckpt = Path(checkpoint)
        if ckpt.exists():
            try:
                self.model = load_model(cfg, ckpt, self.device)
                logger.info("Talc U-Net загружен: %s", ckpt)
            except Exception:
                logger.exception("Не удалось загрузить talc U-Net, будет fallback")

    def predict(self, rgb_norm: np.ndarray, valid: np.ndarray,
                sulfide: np.ndarray) -> TalcResult:
        t = self.cfg.tiling
        mp = rgb_norm.shape[0] * rgb_norm.shape[1] / 1e6

        if self.model is not None:
            tta = mp <= float(t.tta_max_mp)
            prob = predict_tiled(self.model, rgb_norm, self.device,
                                 tile=int(t.tile), overlap=int(t.overlap),
                                 batch=int(t.batch_size), tta=tta)
            used = "unet"
        else:
            prob = self._texture_prob(rgb_norm, valid)
            used = "texture-fallback"

        thr = float(getattr(self.cfg.rule, "talc_prob_thresh", 0.5))
        mask = (prob > thr) & valid & ~sulfide
        mask = filter_small(mask, int(self.cfg.postprocessing.min_object_size))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool)

        conf_map = np.abs(prob - 0.5) * 2.0
        mean_conf = float(conf_map[valid].mean()) if valid.any() else 0.0
        return TalcResult(mask=mask, probability=prob, mean_confidence=mean_conf, used_model=used)

    @staticmethod
    def _texture_prob(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Эвристика: плотность мелкой тёмной 'сыпи' в нерудной матрице."""
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        med = float(np.median(gray[valid])) if valid.any() else 128.0
        dark_fines = ((gray < med * 0.75) & valid).astype(np.float32)
        density = cv2.boxFilter(dark_fines, -1, (61, 61))
        prob = np.clip(density * 2.2, 0, 1)
        return prob.astype(np.float32)
