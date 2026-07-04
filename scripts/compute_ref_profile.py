"""Референсный цветовой профиль (LAB mean/std) по оригиналам ч1.

Используется предобработкой для нормализации цвета доменов ч2/панорам
к профилю обучающей выборки талька. Результат вписывается в config.yaml.

Запуск: py -3.11 scripts/compute_ref_profile.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import imread_rgb  # noqa: E402

ORIG_DIR = ROOT / "ore_data" / "Фото руд по сортам. ч1" / "Оталькованные руды"


def main() -> None:
    files = sorted(p for p in ORIG_DIR.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    means, stds = [], []
    for p in files:
        rgb = imread_rgb(p)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        flat = lab.reshape(-1, 3)
        means.append(flat.mean(axis=0))
        stds.append(flat.std(axis=0))
    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)
    print(f"файлов: {len(files)}")
    print(f"ref_mean: [{mean[0]:.1f}, {mean[1]:.1f}, {mean[2]:.1f}]")
    print(f"ref_std:  [{std[0]:.1f}, {std[1]:.1f}, {std[2]:.1f}]")


if __name__ == "__main__":
    main()
