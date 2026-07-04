"""Калибровка порогов классификации срастаний по image-level меткам.

Только неоталькованные классы (рядовые/тонкие) — метка «оталькованная» не
определяет тип срастаний. Для каждого фото извлекаются признаки зёрен
(replacement, density, area), кэшируются, затем сетка порогов подбирается
по macro-F1. Отдельно печатается F1 по подвыборкам (ч1-5x / ч1-10x / ч2) —
проверка масштабо-инвариантности.

Запуск: py -3.11 scripts/calibrate_grains.py [--max-side 2048] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import imread_rgb  # noqa: E402
from src.preprocessing import preprocess  # noqa: E402
from src.sulfides import classify_grains, sulfide_mask  # noqa: E402
from utils import load_config  # noqa: E402

DATA = ROOT / "ore_data"
SOURCES = [
    (DATA / "Фото руд по сортам. ч1" / "Рядовые руды", "ordinary", "ch1"),
    (DATA / "Фото руд по сортам. ч1" / "Труднообогатимые руды", "fine", "ch1"),
    (DATA / "Фото руд по сортам. ч2" / "рядовые", "ordinary", "ch2"),
    (DATA / "Фото руд по сортам. ч2" / "тонкие", "fine", "ch2"),
]
CACHE = ROOT / "outputs" / "calib_grains.parquet"
RESULT = ROOT / "outputs" / "calib_result.json"


def subset_of(path_str: str, source: str) -> str:
    low = Path(path_str).stem.lower()
    if source == "ch1":
        if "5x" in low or "5х" in low:
            return "ch1_5x"
        if "10x" in low or "10х" in low:
            return "ch1_10x"
        return "ch1_na"
    return "ch2"


def extract_features(cfg, path: Path, max_side: int) -> list[dict]:
    rgb = imread_rgb(path)
    h, w = rgb.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    pre = preprocess(rgb, cfg)
    sulf, _ = sulfide_mask(pre.gray, pre.valid, cfg)
    ga = classify_grains(sulf, cfg)
    return [{"area": z["area"], "replacement": z["replacement"], "density": z["density"]}
            for z in ga.zones]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-side", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0, help="ограничить число фото на класс (отладка)")
    args = parser.parse_args()

    cfg = load_config(str(ROOT / "config" / "config.yaml"))
    CACHE.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    frames = []
    if CACHE.exists():
        prev = pd.read_parquet(CACHE)
        frames.append(prev)
        done = set(prev["image"].unique())
        print(f"Кэш: {len(done)} изображений уже обработано")

    todo = []
    for folder, label, source in SOURCES:
        files = sorted(p for p in folder.iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if args.limit:
            files = files[: args.limit]
        todo += [(p, label, source) for p in files if str(p) not in done]

    print(f"К обработке: {len(todo)} изображений")
    batch = []
    for i, (path, label, source) in enumerate(todo):
        try:
            feats = extract_features(cfg, path, args.max_side)
        except Exception as e:
            print(f"[ERR] {path.name}: {e}")
            continue
        for f in feats:
            f.update({"image": str(path), "label": label,
                      "subset": subset_of(str(path), source)})
        batch += feats
        if (i + 1) % 25 == 0 or i == len(todo) - 1:
            frames.append(pd.DataFrame(batch))
            pd.concat(frames, ignore_index=True).to_parquet(CACHE)
            print(f"  {i + 1}/{len(todo)} готово, зёрен в кэше: {sum(len(fr) for fr in frames)}")
            frames = [pd.concat(frames, ignore_index=True)]
            batch = []

    df = pd.concat(frames, ignore_index=True) if frames else pd.read_parquet(CACHE)

    # ── развёртка порогов ────────────────────────────────────────────────────
    images = df.groupby("image").agg(label=("label", "first"), subset=("subset", "first"))
    rts = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 2.0]      # 2.0 = выключен
    dts = [-1.0, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]  # -1 = выключен

    def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        f1s = []
        for cls in ("ordinary", "fine"):
            tp = ((y_pred == cls) & (y_true == cls)).sum()
            fp = ((y_pred == cls) & (y_true != cls)).sum()
            fn = ((y_pred != cls) & (y_true == cls)).sum()
            f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
        return float(np.mean(f1s))

    best = None
    for rt in rts:
        for dt in dts:
            fine_mask = (df["replacement"] >= rt) | (df["density"] < dt)
            fine_area = df["area"].where(fine_mask, 0).groupby(df["image"]).sum()
            tot_area = df.groupby("image")["area"].sum()
            share = (fine_area / tot_area).reindex(images.index).fillna(0)
            pred = np.where(share >= 0.5, "fine", "ordinary")
            f1 = macro_f1(images["label"].values, pred)
            if best is None or f1 > best["f1"]:
                per_subset = {}
                for sub in sorted(images["subset"].unique()):
                    m = images["subset"] == sub
                    per_subset[sub] = round(macro_f1(images["label"].values[m.values], pred[m.values]), 4)
                acc = float((pred == images["label"].values).mean())
                best = {"replacement_thresh": rt, "density_thresh": dt,
                        "f1": round(f1, 4), "accuracy": round(acc, 4),
                        "per_subset_f1": per_subset,
                        "n_images": int(len(images))}

    print("\nЛучшие пороги:", json.dumps(best, ensure_ascii=False, indent=2))
    RESULT.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Сохранено -> {RESULT}")


if __name__ == "__main__":
    main()
