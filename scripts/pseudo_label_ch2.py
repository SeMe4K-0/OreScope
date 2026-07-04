"""Псевдоразметка талька для «ч2/оталькованные» (масок эксперта там нет).

Модель v2 (обучена на уточнённых масках ч1, вход нормализуется к профилю ч1)
размечает тальковую фазу; серая зона вероятностей и невалид идут в ignore.
Результат — пары для дообучения (--extra-dir у train_talc.py) + contact-sheets
для обязательной визуальной проверки перед использованием.

Выход: data/talc_masks_ch2/{stem}.png, {stem}_ignore.png, {stem}_rgb.jpg;
       outputs/talc_review_ch2/ — оверлеи, contact-sheets, report.csv.

Запуск: py -3.11 scripts/pseudo_label_ch2.py [--p-hi 0.6] [--p-lo 0.35]
        py -3.11 scripts/pseudo_label_ch2.py --reject "-8" "152_"   # удалить отбракованные
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import imread_rgb, imwrite  # noqa: E402
from src.preprocessing import preprocess  # noqa: E402
from src.segmentation.model import load_model  # noqa: E402
from src.sulfides import filter_small, sulfide_mask  # noqa: E402
from src.tiling import predict_tiled  # noqa: E402
from utils import get_device, load_config  # noqa: E402

SRC = ROOT / "ore_data" / "Фото руд по сортам. ч2" / "оталькованные"
OUT_DIR = ROOT / "data" / "talc_masks_ch2"
REVIEW_DIR = ROOT / "outputs" / "talc_review_ch2"


def reject(stems: list[str]) -> None:
    n = 0
    for stem in stems:
        for suffix in (".png", "_ignore.png", "_rgb.jpg"):
            p = OUT_DIR / f"{stem}{suffix}"
            if p.exists():
                p.unlink()
                n += 1
    print(f"Удалено файлов: {n}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/talc_best.pt")
    parser.add_argument("--p-hi", type=float, default=0.6, help="вероятность выше -> тальк")
    parser.add_argument("--p-lo", type=float, default=0.35, help="между p-lo и p-hi -> ignore")
    parser.add_argument("--max-side", type=int, default=2272, help="даунскейл к масштабу ч1")
    parser.add_argument("--reject", nargs="*", default=None,
                        help="удалить перечисленные stem'ы после визуальной проверки")
    args = parser.parse_args()

    if args.reject is not None:
        reject(args.reject)
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(ROOT / "config" / "config.yaml"))
    device = get_device()
    model = load_model(cfg, ROOT / args.checkpoint, device)

    files = sorted(p for p in SRC.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    rows, thumbs = [], []
    for i, path in enumerate(files):
        rgb = imread_rgb(path)
        h, w = rgb.shape[:2]
        s = args.max_side / max(h, w)
        if s < 1.0:
            rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        pre = preprocess(rgb, cfg)
        sulf, _ = sulfide_mask(pre.gray, pre.valid, cfg)
        prob = predict_tiled(model, pre.rgb, device, tile=1024, overlap=128, batch=4, tta=True)

        talc = (prob > args.p_hi) & pre.valid & ~sulf
        # мягкая согласованность с тёмной фазой (отсекает уверенные ложные на светлом)
        g = pre.gray.astype(np.float32)
        matrix = pre.valid & ~sulf
        med = float(np.median(g[matrix])) if matrix.any() else 128.0
        talc &= g < med
        talc = filter_small(talc, 12)

        ignore = (((prob > args.p_lo) & ~talc) | ~pre.valid) & ~sulf
        frac = 100.0 * talc.sum() / max(pre.valid.sum(), 1)

        stem = path.stem
        imwrite(OUT_DIR / f"{stem}.png", talc.astype(np.uint8) * 255)
        imwrite(OUT_DIR / f"{stem}_ignore.png", ignore.astype(np.uint8) * 255)
        imwrite(OUT_DIR / f"{stem}_rgb.jpg", pre.rgb)

        overlay = pre.rgb.copy()
        color = np.array([40, 90, 255], np.uint8)
        overlay[talc] = (0.45 * overlay[talc] + 0.55 * color).astype(np.uint8)
        imwrite(REVIEW_DIR / f"{stem}_overlay.jpg", overlay)

        th = cv2.resize(overlay, (500, 375))
        label = f"{i:02d} {stem} {frac:.1f}%"
        cv2.putText(th, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(th, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        thumbs.append(th)
        rows.append({"idx": i, "file": path.name, "talc_percent": round(frac, 2)})
        print(f"[{i:02d}] {path.name}: тальк {frac:.1f}%", flush=True)

    cols = 4
    for sheet_i in range(0, len(thumbs), 16):
        chunk = thumbs[sheet_i:sheet_i + 16]
        while len(chunk) % cols:
            chunk.append(np.full_like(chunk[0], 255))
        rows_img = [np.hstack(chunk[r:r + cols]) for r in range(0, len(chunk), cols)]
        imwrite(REVIEW_DIR / f"contact_sheet_{sheet_i // 16}.jpg", np.vstack(rows_img))

    with open(REVIEW_DIR / "report.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "file", "talc_percent"])
        writer.writeheader()
        writer.writerows(rows)

    vals = np.array([r["talc_percent"] for r in rows])
    print(f"\nИтого {len(rows)} фото: медиана {np.median(vals):.1f}%, "
          f">10%: {(vals > 10).sum()}, <3%: {(vals < 3).sum()}")
    print(f"Проверка: {REVIEW_DIR}")


if __name__ == "__main__":
    main()
