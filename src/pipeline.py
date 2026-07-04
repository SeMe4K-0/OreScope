"""Оркестратор полного анализа одного изображения.

Единая точка для дашборда, пакетного прогона и оценки качества:
загрузка -> предобработка -> сульфиды -> срастания -> тальк -> правило ->
маски/оверлей/метрики + JSON-лог параметров (воспроизводимость).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.classify import OreVerdict, classify_ore
from src.grain_cnn import GrainCNN
from src.io_utils import imread_rgb
from src.preprocessing import Preprocessed, preprocess
from src.sulfides import GrainAnalysis, classify_grains, sulfide_mask
from src.talc import TalcResult, TalcSegmenter

logger = logging.getLogger(__name__)

CLASS_COLORS = {1: (0, 200, 0), 2: (255, 0, 0), 3: (40, 90, 255)}


@dataclass
class AnalysisResult:
    verdict: OreVerdict
    class_mask: np.ndarray        # uint8: 0 фон, 1 обычные, 2 тонкие, 3 тальк
    valid: np.ndarray             # bool
    probability: np.ndarray       # float32 — карта вероятности талька
    rgb: np.ndarray               # исходное RGB (возможно после даунскейла)
    scale_applied: float          # коэффициент даунскейла (1.0 = нет)
    talc_model: str
    talc_confidence: float
    elapsed_s: float
    params_log: dict = field(default_factory=dict)
    grain_stats: dict = field(default_factory=dict)


def _override_grains_with_cnn(grains: GrainAnalysis, prob_map: np.ndarray) -> GrainAnalysis:
    """Переклассификация зёрен по карте вероятности CNN (порог 0.5 по среднему)."""
    sulf = grains.sulfide_mask
    n, labels = cv2.connectedComponents(sulf.astype(np.uint8), connectivity=8)
    means = np.zeros(n, np.float64)
    counts = np.zeros(n, np.int64)
    np.add.at(means, labels.ravel(), prob_map.ravel())
    np.add.at(counts, labels.ravel(), 1)
    means = means / np.maximum(counts, 1)
    cls_per_label = np.where(means >= 0.5, 2, 1).astype(np.uint8)
    cls_per_label[0] = 0
    grains.class_mask = cls_per_label[labels] * sulf.astype(np.uint8)
    grains.ordinary_px = int((grains.class_mask == 1).sum())
    grains.fine_px = int((grains.class_mask == 2).sum())
    return grains


def analyze_image(source: str | Path | np.ndarray, cfg: DictConfig,
                  segmenter: TalcSegmenter, um_per_px: float | None = None,
                  grain_cnn: GrainCNN | None = None) -> AnalysisResult:
    t0 = time.time()
    rgb = imread_rgb(source) if isinstance(source, (str, Path)) else source
    h0, w0 = rgb.shape[:2]

    scale = 1.0
    max_side = int(cfg.tiling.downscale_max_side)
    if max_side > 0 and max(h0, w0) > max_side:
        scale = max_side / max(h0, w0)
        rgb = cv2.resize(rgb, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
        logger.info("Даунскейл %dx%d -> %dx%d", w0, h0, rgb.shape[1], rgb.shape[0])

    pre: Preprocessed = preprocess(rgb, cfg)
    sulf, thresh = sulfide_mask(pre.gray, pre.valid, cfg, gray_raw=pre.gray_raw)
    grains: GrainAnalysis = classify_grains(sulf, cfg)
    grain_model = "morphology"
    if grain_cnn is not None and grain_cnn.available:
        pmap = grain_cnn.prob_map(pre.rgb)
        grains = _override_grains_with_cnn(grains, pmap)
        grain_model = "cnn+morphology"
    talc: TalcResult = segmenter.predict(pre.rgb, pre.valid, sulf)

    class_mask = grains.class_mask.copy()
    class_mask[talc.mask] = 3

    # зоны оталькования: компоненты локальной плотности тальковой фазы, у которых
    # заполнение фазой >= zone_fill_min (реальные зоны заполнены на ~26%, разреженные
    # ложные срабатывания дают «жидкие» гало ~10%). Правило 10% ТЗ — по зонам,
    # как в экспертной разметке; калибровка по 39 парам ч1 (bias -1.9 п.п.)
    zone_frac = float(getattr(cfg.rule, "zone_window_frac", 0.06))
    zone_dens = float(getattr(cfg.rule, "zone_density", 0.03))
    zone_fill_min = float(getattr(cfg.rule, "zone_fill_min", 0.12))
    tm = talc.mask
    ds = 4 if max(tm.shape) > 8000 else 1  # экономия памяти на панорамах
    if ds > 1:
        tm_small = cv2.resize(tm.astype(np.uint8), (tm.shape[1] // ds, tm.shape[0] // ds),
                              interpolation=cv2.INTER_AREA).astype(np.float32)
    else:
        tm_small = tm.astype(np.float32)
    win = max(31, int(zone_frac * max(tm_small.shape)) | 1)
    dens = cv2.boxFilter(tm_small, -1, (win, win))
    weak = (dens > zone_dens).astype(np.uint8)
    n_z, zone_lab = cv2.connectedComponents(weak, connectivity=4)
    area = np.bincount(zone_lab.ravel(), minlength=n_z).astype(np.float64)
    phase = np.bincount(zone_lab.ravel(), weights=tm_small.astype(np.float64).ravel(),
                        minlength=n_z)
    keep = np.where(phase / np.maximum(area, 1) >= zone_fill_min)[0]
    zone = np.isin(zone_lab, keep[keep != 0])
    talc_zone_px = int(round(float(zone.mean()) * int(pre.valid.sum())))

    verdict = classify_ore(
        talc_px=int(talc.mask.sum()),
        ordinary_px=grains.ordinary_px,
        fine_px=grains.fine_px,
        valid_px=int(pre.valid.sum()),
        cfg=cfg,
        talc_zone_px=talc_zone_px,
    )
    if talc.mean_confidence < 0.55 and talc.used_model == "unet":
        verdict.warnings.append(
            f"Низкая средняя уверенность модели талька ({talc.mean_confidence:.2f}) — "
            "рекомендуется экспертная проверка.")
        verdict.review_needed = True
        verdict.conclusion = verdict.conclusion + " " + verdict.warnings[-1]

    elapsed = time.time() - t0
    params_log = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": str(source) if isinstance(source, (str, Path)) else "<array>",
        "original_size": [w0, h0],
        "scale_applied": scale,
        "um_per_px": um_per_px,
        "sulfide_threshold": round(float(thresh), 1),
        "grain_model": grain_model,
        "talc_model": talc.used_model,
        "elapsed_s": round(elapsed, 1),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    return AnalysisResult(
        verdict=verdict,
        class_mask=class_mask,
        valid=pre.valid,
        probability=talc.probability,
        rgb=rgb,
        scale_applied=scale,
        talc_model=talc.used_model,
        talc_confidence=talc.mean_confidence,
        elapsed_s=elapsed,
        params_log=params_log,
        grain_stats={"n_grains": len(grains.zones)},
    )


def make_overlay(res: AnalysisResult, alpha: float = 0.55, max_side: int = 4096) -> np.ndarray:
    """Цветовой оверлей (зелёный/красный/синий) с даунскейлом для браузера."""
    rgb = res.rgb
    mask = res.class_mask
    h, w = rgb.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = rgb.copy()
    for cls, color in CLASS_COLORS.items():
        m = mask == cls
        out[m] = ((1 - alpha) * out[m] + alpha * np.array(color)).astype(np.uint8)
    return out


def make_uncertainty_vis(res: AnalysisResult, max_side: int = 4096) -> np.ndarray:
    """Карта уверенности (вероятность талька) как heatmap."""
    p = res.probability
    h, w = p.shape
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        p = cv2.resize(p, (int(w * s), int(h * s)), interpolation=cv2.INTER_LINEAR)
    u8 = (np.clip(p, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)


def metrics_table(res: AnalysisResult, um_per_px: float | None = None) -> list[dict]:
    """Таблица метрик; при известном µm/px добавляются абсолютные площади."""
    v = res.verdict
    scale = res.scale_applied
    rows = [
        {"metric": "Общая доля сульфидов", "value_pct": v.sulfide_percent},
        {"metric": "Обычные срастания", "value_pct": v.ordinary_percent},
        {"metric": "Тонкие срастания", "value_pct": v.fine_percent},
        {"metric": "Доля тонких среди сульфидов", "value_pct": v.fine_share},
        {"metric": "Тальк (фаза)", "value_pct": v.talc_percent},
        {"metric": "Зоны оталькования", "value_pct": v.talc_zone_percent},
    ]
    if um_per_px:
        um_eff = um_per_px / scale  # после даунскейла пиксель крупнее
        px_area_mm2 = (um_eff / 1000.0) ** 2
        valid_px = int(res.valid.sum())
        for row, px in zip(rows, [
            int((res.class_mask == 1).sum() + (res.class_mask == 2).sum()),
            int((res.class_mask == 1).sum()),
            int((res.class_mask == 2).sum()),
            None,
            int((res.class_mask == 3).sum()),
        ]):
            if px is not None:
                row["area_mm2"] = round(px * px_area_mm2, 3)
        rows.append({"metric": "Площадь образца", "value_pct": 100.0,
                     "area_mm2": round(valid_px * px_area_mm2, 3)})
    return rows


def save_params_log(res: AnalysisResult, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(res.params_log, ensure_ascii=False, indent=2),
                          encoding="utf-8")
