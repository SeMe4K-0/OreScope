"""Предобработка OM-изображений: маска валидных пикселей, выравнивание
освещения, нормализация цвета, CLAHE, подавление царапин."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from omegaconf import DictConfig


@dataclass
class Preprocessed:
    rgb: np.ndarray          # нормализованное RGB (для визуализации и U-Net)
    gray: np.ndarray         # выровненный по освещению серый (для порога сульфидов)
    valid: np.ndarray        # bool: пиксели образца (без выколок/пустот)
    valid_fraction: float
    gray_raw: np.ndarray | None = None  # яркость ДО выравнивания освещения:
    # массивные сульфиды на весь кадр поглощаются фоном при выравнивании и
    # пропадают из gray — абсолютная ветка детектора работает по gray_raw


def compute_valid_mask(rgb: np.ndarray, cfg: DictConfig) -> np.ndarray:
    """Валидные пиксели образца. Инвалидны только крупные, почти чёрные и
    плоские области (выколки, пустоты, паддинг панорам) — тёмная матрица
    панорам с текстурой остаётся валидной."""
    p = cfg.preprocessing.valid_mask
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    k = int(p.var_kernel)
    mean = cv2.boxFilter(gray.astype(np.float32), -1, (k, k))
    sq = cv2.boxFilter((gray.astype(np.float32)) ** 2, -1, (k, k))
    std = np.sqrt(np.maximum(sq - mean ** 2, 0))

    invalid = (gray < int(p.dark_thresh)) & (std < float(p.var_thresh))
    from src.sulfides import filter_small
    invalid = filter_small(invalid, int(p.min_hole_area))
    invalid = cv2.morphologyEx(invalid.astype(np.uint8), cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(bool)
    return ~invalid


def normalize_color_profile(rgb: np.ndarray, cfg: DictConfig) -> np.ndarray:
    """Перенос Рейнхарда (LAB mean/std) к референсному профилю домена ч1.

    Снимает доменный сдвиг тёмных нейтральных снимков (ч2, панорамы), на которых
    U-Net талька, обученный на жёлто-зелёном тракте ч1, теряет чувствительность.
    Для изображений, уже близких к референсу (сам домен ч1), — идентичность.
    Статистики считаются по даунскейлу, преобразование — полосами (панорамы)."""
    cn = getattr(cfg.preprocessing, "color_norm", None)
    if cn is None or not bool(cn.enabled):
        return rgb
    ref_mean = np.array(list(cn.ref_mean), np.float32)
    ref_std = np.array(list(cn.ref_std), np.float32)

    h, w = rgb.shape[:2]
    s = 1024 / max(h, w)
    small = cv2.resize(rgb, (max(int(w * s), 8), max(int(h * s), 8)),
                       interpolation=cv2.INTER_AREA) if s < 1.0 else rgb
    lab_small = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32).reshape(-1, 3)
    mean = lab_small.mean(axis=0)
    std = np.maximum(lab_small.std(axis=0), 1e-3)

    if (np.abs(mean - ref_mean) < float(cn.tol_mean)).all() and \
            (np.abs(std - ref_std) < float(cn.tol_std)).all():
        return rgb

    out = np.empty_like(rgb)
    stripe = int(cn.stripe)
    for y0 in range(0, h, stripe):
        y1 = min(y0 + stripe, h)
        lab = cv2.cvtColor(rgb[y0:y1], cv2.COLOR_RGB2LAB).astype(np.float32)
        lab = (lab - mean) / std * ref_std + ref_mean
        out[y0:y1] = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return out


def preprocess(rgb: np.ndarray, cfg: DictConfig) -> Preprocessed:
    pp = cfg.preprocessing

    rgb = normalize_color_profile(rgb, cfg)

    if int(pp.median_ksize) >= 3:
        rgb = cv2.medianBlur(rgb, int(pp.median_ksize))

    valid = compute_valid_mask(rgb, cfg)

    # выравнивание освещения по L-каналу (вычитание крупномасштабного фона);
    # фон оценивается на 1/8 даунскейле — эквивалентно гауссу sigma на полном
    # разрешении, но на два порядка быстрее для панорам
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0]
    sigma = int(pp.illumination_sigma)
    h, w = L.shape
    ds = 8
    small = cv2.resize(L, (max(w // ds, 8), max(h // ds, 8)), interpolation=cv2.INTER_AREA)
    bg_small = cv2.GaussianBlur(small, (0, 0), max(sigma / ds, 3))
    bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)
    L_flat = np.clip(L - bg + float(bg.mean()), 0, 255)

    # серый баланс (gray-world) по валидным пикселям
    rgb_f = rgb.astype(np.float32)
    means = [max(float(rgb_f[..., c][valid].mean()), 1.0) for c in range(3)]
    gray_mean = sum(means) / 3.0
    for c in range(3):
        rgb_f[..., c] = np.clip(rgb_f[..., c] * (gray_mean / means[c]), 0, 255)

    lab_bal = cv2.cvtColor(rgb_f.astype(np.uint8), cv2.COLOR_RGB2LAB)
    lab_bal[..., 0] = L_flat.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=float(pp.clahe_clip),
                            tileGridSize=(int(pp.clahe_grid), int(pp.clahe_grid)))
    lab_clahe = lab_bal.copy()
    lab_clahe[..., 0] = clahe.apply(lab_bal[..., 0])
    rgb_norm = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    return Preprocessed(
        rgb=rgb_norm,
        gray=L_flat.astype(np.uint8),
        valid=valid,
        valid_fraction=float(valid.mean()),
        gray_raw=L.astype(np.uint8),
    )
