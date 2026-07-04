"""Сегментация сульфидов и классификация срастаний (обычные / тонкие).

Подход зонный и масштабо-инвариантный:
  1. Маска сульфидов — порог по яркости (Otsu по валидным пикселям с
     защитой от вырождения на тёмных панорамах: медиана + k*MAD).
  2. Зоны срастаний — closing маски сульфидов ядром, привязанным к
     медианному размеру зерна, + заливка дыр. Каждая зона = связная область.
  3. Степень замещения зоны = 1 − (площадь сульфидов в зоне / площадь зоны).
     Отношение площадей — инвариант масштаба. Зона с замещением выше порога
     → тонкие срастания, иначе → обычные.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from omegaconf import DictConfig


def filter_small(mask: np.ndarray, min_px: int) -> np.ndarray:
    """Убирает связные компоненты меньше min_px (bool -> bool)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_px
    return keep[labels]


@dataclass
class GrainAnalysis:
    sulfide_mask: np.ndarray        # bool: все сульфиды
    class_mask: np.ndarray          # uint8: 0 фон, 1 обычные, 2 тонкие (только пиксели сульфидов)
    threshold: float                # использованный порог яркости
    ordinary_px: int = 0
    fine_px: int = 0
    zones: list[dict] = field(default_factory=list)


def sulfide_mask(gray: np.ndarray, valid: np.ndarray, cfg: DictConfig,
                 gray_raw: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Бинарная маска сульфидов по выровненному серому каналу.

    gray_raw (яркость ДО выравнивания освещения) включает абсолютную ветку:
    массивные сульфиды на большую часть кадра поглощаются фоном при
    выравнивании и в gray неотличимы от матрицы (наблюдалось: кадр с
    массивным агрегатом -> 0% сульфидов). Порог якорится на матрицу
    (25-й перцентиль + abs_delta, не ниже abs_min)."""
    s = cfg.sulfides
    vals = gray[valid]
    if vals.size < 100:
        return np.zeros_like(gray, dtype=bool), 255.0

    # Otsu — первичный порог. Отвергаем его только когда он разрезает
    # текстуру матрицы (слабый контраст классов) — случай тёмных панорам
    # с долей сульфидов << 1%.
    t_otsu, _ = cv2.threshold(vals.reshape(-1, 1).astype(np.uint8), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t_otsu = float(t_otsu)
    above = vals[vals > t_otsu]
    below = vals[vals <= t_otsu]
    contrast = (float(above.mean()) if above.size else 255.0) - (float(below.mean()) if below.size else 0.0)

    if contrast >= 40.0 and above.size > 0:
        t = t_otsu
    else:
        med = float(np.median(vals))
        sigma = 1.4826 * (float(np.median(np.abs(vals.astype(np.float32) - med))) + 1e-6)
        t = med + 8.0 * sigma
    t = float(np.clip(t, 20.0, 250.0))

    mask = (gray > t) & valid
    if gray_raw is not None:
        p25 = float(np.percentile(gray_raw[valid], 25))
        t_abs = max(p25 + float(getattr(s, "abs_delta", 60.0)),
                    float(getattr(s, "abs_min", 120.0)))
        mask |= (gray_raw > t_abs) & valid
    mask = filter_small(mask, int(s.min_grain_px))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).astype(bool)
    return mask, t


def _fill_holes(binary: np.ndarray) -> np.ndarray:
    flooded = binary.astype(np.uint8).copy()
    h, w = flooded.shape
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flooded, canvas, (0, 0), 1)
    holes = (flooded == 0) & (~binary)
    return binary | holes


def classify_grains(sulf: np.ndarray, cfg: DictConfig) -> GrainAnalysis:
    """По-зёренная классификация: обычные (1) vs тонкие (2) срастания.

    Для каждого зерна строится оболочка (closing ядром, пропорциональным
    размеру зерна, + заливка дыр). Степень замещения = 1 − area/envelope —
    отношение площадей, инвариантное к масштабу съёмки. Массивное зерно с
    редкими включениями -> низкое замещение -> обычное; ажурное/скелетное
    зерно -> высокое замещение -> тонкое.
    """
    g = cfg.grains
    out = GrainAnalysis(sulfide_mask=sulf, class_mask=np.zeros(sulf.shape, np.uint8), threshold=0.0)
    if not sulf.any():
        return out

    repl_thresh = float(g.replacement_thresh)
    dens_thresh = float(g.density_thresh)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(sulf.astype(np.uint8), connectivity=8)
    class_mask = np.zeros(sulf.shape, np.uint8)
    H, W = sulf.shape

    # карта локальной плотности сульфидов; окно привязано к характерной
    # толщине зёрен (средняя удвоенная дистанция до фона по пикселям сульфидов),
    # поэтому признак инвариантен к разрешению съёмки
    dist = cv2.distanceTransform(sulf.astype(np.uint8), cv2.DIST_L2, 3)
    t_char = 2.0 * float(dist[sulf].mean())
    w_dens = int(np.clip(4.0 * t_char, 31, 401))
    if w_dens % 2 == 0:
        w_dens += 1
    density = cv2.boxFilter(sulf.astype(np.float32), -1, (w_dens, w_dens))

    for i in range(1, n):
        x, y, w, h, area = (int(stats[i, j]) for j in
                            (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH,
                             cv2.CC_STAT_HEIGHT, cv2.CC_STAT_AREA))
        equiv_d = 2.0 * np.sqrt(area / np.pi)
        k = int(np.clip(0.5 * equiv_d, 3, 61))
        if k % 2 == 0:
            k += 1
        pad = k
        y0, y1 = max(0, y - pad), min(H, y + h + pad)
        x0, x1 = max(0, x - pad), min(W, x + w + pad)
        crop = (labels[y0:y1, x0:x1] == i).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        closed = cv2.morphologyEx(crop, cv2.MORPH_CLOSE, kernel)
        env = _fill_holes(closed.astype(bool))
        env_area = int(env.sum())
        replacement = 1.0 - area / env_area if env_area > 0 else 0.0

        # средняя локальная плотность сульфидов по пикселям зерна:
        # зерно в разреженном окружении -> тонкое срастание
        mean_dens = float(density[y0:y1, x0:x1][crop.astype(bool)].mean())

        is_fine = replacement >= repl_thresh or mean_dens < dens_thresh
        cls = 2 if is_fine else 1
        class_mask[y0:y1, x0:x1][crop.astype(bool)] = cls
        out.zones.append({
            "area": area,
            "envelope": env_area,
            "replacement": round(replacement, 3),
            "density": round(mean_dens, 3),
            "class": "fine" if cls == 2 else "ordinary",
        })

    out.class_mask = class_mask
    out.ordinary_px = int((class_mask == 1).sum())
    out.fine_px = int((class_mask == 2).sum())
    return out
