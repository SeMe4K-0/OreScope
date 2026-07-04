"""Калибровка порога бинаризации вероятности талька по TRAIN-изображениям
(минимизация MAE доли талька). Val не участвует в подборе.

Запуск: py -3.11 scripts/sweep_talc_thresh.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import preprocess  # noqa: E402
from src.io_utils import imread_rgb  # noqa: E402
from src.sulfides import filter_small, sulfide_mask  # noqa: E402
from src.talc import TalcSegmenter  # noqa: E402
from utils import load_config  # noqa: E402

ORIG_DIR = ROOT / "ore_data" / "Фото руд по сортам. ч1" / "Оталькованные руды"
MASK_DIR = ROOT / "data" / "talc_masks"


def find_orig(stem: str) -> Path | None:
    for ext in (".JPG", ".jpg", ".jpeg", ".png"):
        p = ORIG_DIR / (stem + ext)
        if p.exists():
            return p
    return None


def main() -> None:
    cfg = load_config(str(ROOT / "config" / "config.yaml"))
    seg = TalcSegmenter(cfg, ROOT / "models" / "talc_best.pt")

    stems = [p.stem for p in sorted(MASK_DIR.glob("*.png")) if find_orig(p.stem)]
    rng = np.random.default_rng(42)
    order = rng.permutation(len(stems))
    n_val = max(1, int(len(stems) * 0.2))
    val_idx = set(order[:n_val].tolist())

    data = []  # (is_val, prob, valid, sulf, gt_zone)
    for i, stem in enumerate(stems):
        rgb = imread_rgb(find_orig(stem))
        gt_zone = cv2.imdecode(np.fromfile(str(MASK_DIR / f"{stem}.png"), np.uint8),
                               cv2.IMREAD_GRAYSCALE) > 127
        pre = preprocess(rgb, cfg)
        sulf, _ = sulfide_mask(pre.gray, pre.valid, cfg)
        res = seg.predict(pre.rgb, pre.valid, sulf)
        gt = gt_zone & ~sulf & pre.valid
        data.append((i in val_idx, res.probability, pre.valid, sulf, gt))
        print(f"  [{i+1}/{len(stems)}] {stem}", flush=True)

    print(f"\n{'thr':>5} | {'train MAE':>9} | {'val MAE':>7}")
    best = None
    for thr in np.arange(0.35, 0.86, 0.05):
        maes = {True: [], False: []}
        for is_val, prob, valid, sulf, gt in data:
            pred = (prob > thr) & valid & ~sulf
            pred = filter_small(pred, int(cfg.postprocessing.min_object_size))
            vp = max(int(valid.sum()), 1)
            maes[is_val].append(abs(100.0 * gt.sum() / vp - 100.0 * pred.sum() / vp))
        tr, vl = float(np.mean(maes[False])), float(np.mean(maes[True]))
        print(f"{thr:5.2f} | {tr:9.2f} | {vl:7.2f}")
        if best is None or tr < best[1]:
            best = (float(thr), tr, vl)
    print(f"\nЛучший порог по train: {best[0]:.2f} (train MAE {best[1]:.2f}, val MAE {best[2]:.2f})")


if __name__ == "__main__":
    main()
