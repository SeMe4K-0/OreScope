"""Визуальный смоук-тест baseline: сульфиды + классификация срастаний.

Прогоняет несколько образцов, сохраняет оверлеи в outputs/debug/.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classify import classify_ore  # noqa: E402
from src.io_utils import imread_rgb, imwrite  # noqa: E402
from src.preprocessing import preprocess  # noqa: E402
from src.sulfides import classify_grains, sulfide_mask  # noqa: E402
from utils import load_config  # noqa: E402

SAMPLES = [
    ("ryadovaya", ROOT / "ore_data/Фото руд по сортам. ч1/Рядовые руды/2539589-1.JPG"),
    ("trudnoobog", ROOT / "ore_data/Фото руд по сортам. ч1/Труднообогатимые руды/2539439-3.JPG"),
    ("otalk", ROOT / "ore_data/Фото руд по сортам. ч1/Оталькованные руды/2550374-2 10х.JPG"),
    ("ch2_ryad", ROOT / "ore_data/Фото руд по сортам. ч2/рядовые/1.jpg"),
    ("ch2_tonk", ROOT / "ore_data/Фото руд по сортам. ч2/тонкие/-4.jpg"),
]


def main() -> None:
    cfg = load_config(str(ROOT / "config" / "config.yaml"))
    out_dir = ROOT / "outputs" / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, path in SAMPLES:
        t0 = time.time()
        rgb = imread_rgb(path)
        pre = preprocess(rgb, cfg)
        sulf, thresh = sulfide_mask(pre.gray, pre.valid, cfg)
        ga = classify_grains(sulf, cfg)
        verdict = classify_ore(0, ga.ordinary_px, ga.fine_px, int(pre.valid.sum()), cfg)

        overlay = rgb.copy()
        overlay[ga.class_mask == 1] = (0.35 * overlay[ga.class_mask == 1] + 0.65 * np.array([0, 200, 0])).astype(np.uint8)
        overlay[ga.class_mask == 2] = (0.35 * overlay[ga.class_mask == 2] + 0.65 * np.array([255, 0, 0])).astype(np.uint8)
        overlay[~pre.valid] = (0.5 * overlay[~pre.valid]).astype(np.uint8)

        dt = time.time() - t0
        print(f"{name}: t={thresh:.0f} | сульфиды {verdict.sulfide_percent:.1f}% "
              f"(обычные {verdict.ordinary_percent:.1f}% / тонкие {verdict.fine_percent:.1f}%) | "
              f"доля тонких {verdict.fine_share:.0f}% | зон {len(ga.zones)} | {dt:.1f}s")
        imwrite(out_dir / f"{name}_overlay.jpg", overlay)
        imwrite(out_dir / f"{name}_gray.jpg", pre.gray)


if __name__ == "__main__":
    main()
