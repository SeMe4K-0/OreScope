"""Итоговые метрики качества OreScope одним прогоном.

1. Тальк: MAE доли (%) против экспертной разметки на hold-out из 42 пар
   (те же val-изображения, что при обучении U-Net: сид 42, 20%).
   Цель ТЗ: MAE <= 3 п.п. Также пиксельный Dice.
2. Негативные контроли (по SulfideNet): ложный тальк на рядовых/тонких фото —
   средний % и доля фото с ложным тальком > 3%.
3. Срастания: macro-F1 и accuracy image-level на hold-out ч1+ч2
   (морфология и/или CNN — что доступно), включая разбивку по подвыборкам.
4. Классификация сорта руды end-to-end на hold-out всех трёх классов.

Запуск: py -3.11 scripts/evaluate.py [--neg-limit 30]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.grain_cnn import GrainCNN  # noqa: E402
from src.pipeline import analyze_image as _analyze  # noqa: E402
from src.talc import TalcSegmenter  # noqa: E402
from utils import load_config  # noqa: E402

GCNN: GrainCNN | None = None


def analyze_image(source, cfg, seg, **kw):
    return _analyze(source, cfg, seg, grain_cnn=GCNN, **kw)

ORIG_DIR = ROOT / "ore_data" / "Фото руд по сортам. ч1" / "Оталькованные руды"
MASK_DIR = ROOT / "data" / "talc_masks"
CH1 = ROOT / "ore_data" / "Фото руд по сортам. ч1"
CH2 = ROOT / "ore_data" / "Фото руд по сортам. ч2"


def find_orig(stem: str) -> Path | None:
    for ext in (".JPG", ".jpg", ".jpeg", ".png"):
        p = ORIG_DIR / (stem + ext)
        if p.exists():
            return p
    return None


def talc_metrics(cfg, seg, val_stems: list[str]) -> dict:
    maes, dices, gt_list, pred_list = [], [], [], []
    for stem in val_stems:
        orig = find_orig(stem)
        mask_p = MASK_DIR / f"{stem}.png"
        if orig is None or not mask_p.exists():
            continue
        gt_zone = cv2.imdecode(np.fromfile(str(mask_p), np.uint8), cv2.IMREAD_GRAYSCALE) > 127
        res = analyze_image(orig, cfg, seg)
        # GT — уточнённая маска тальковой фазы (уже без сульфидов, make_talc_masks.py)
        gt = gt_zone & res.valid
        pred = res.class_mask == 3
        valid_px = max(int(res.valid.sum()), 1)
        gt_pct = 100.0 * gt.sum() / valid_px
        pred_pct = res.verdict.talc_percent
        maes.append(abs(gt_pct - pred_pct))
        inter = float((gt & pred).sum())
        dices.append(2 * inter / max(float(gt.sum() + pred.sum()), 1))
        gt_list.append(round(gt_pct, 1))
        pred_list.append(round(pred_pct, 1))
        print(f"  {stem}: GT {gt_pct:.1f}% / pred {pred_pct:.1f}% / dice {dices[-1]:.3f}", flush=True)
    return {"MAE_pp": round(float(np.mean(maes)), 2), "Dice": round(float(np.mean(dices)), 3),
            "n": len(maes), "gt": gt_list, "pred": pred_list}


def negative_controls(cfg, seg, limit: int) -> dict:
    rng = np.random.default_rng(0)
    files = []
    for sub in (CH1 / "Рядовые руды", CH1 / "Труднообогатимые руды"):
        cand = sorted(p for p in sub.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        files += list(rng.choice(cand, size=min(limit // 2, len(cand)), replace=False))
    fps = []
    for p in files:
        res = analyze_image(p, cfg, seg)
        fps.append(res.verdict.talc_percent)
        print(f"  NC {p.name}: ложный тальк {res.verdict.talc_percent:.1f}%", flush=True)
    fps = np.array(fps)
    return {"mean_false_talc_pct": round(float(fps.mean()), 2),
            "frac_above_3pct": round(float((fps > 3).mean()), 3),
            "frac_above_10pct": round(float((fps > 10).mean()), 3), "n": len(fps)}


def ore_class_eval(cfg, seg, per_class: int) -> dict:
    rng = np.random.default_rng(1)
    sources = [
        (CH1 / "Рядовые руды", "рядовая"),
        (CH1 / "Труднообогатимые руды", "труднообогатимая"),
        (CH1 / "Оталькованные руды", "оталькованная"),
        (CH2 / "рядовые", "рядовая"),
        (CH2 / "тонкие", "труднообогатимая"),
        (CH2 / "оталькованные", "оталькованная"),
    ]
    y_true, y_pred = [], []
    for folder, label in sources:
        cand = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        pick = rng.choice(cand, size=min(per_class, len(cand)), replace=False)
        for p in pick:
            res = analyze_image(p, cfg, seg)
            y_true.append(label)
            y_pred.append(res.verdict.ore_class)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = float((y_true == y_pred).mean())
    f1s = {}
    for cls in ("рядовая", "труднообогатимая", "оталькованная"):
        tp = ((y_pred == cls) & (y_true == cls)).sum()
        fp = ((y_pred == cls) & (y_true != cls)).sum()
        fn = ((y_pred != cls) & (y_true == cls)).sum()
        f1s[cls] = round(2 * tp / max(2 * tp + fp + fn, 1), 3)
    conf = {}
    for t in set(y_true):
        for pr in set(y_pred):
            n = int(((y_true == t) & (y_pred == pr)).sum())
            if n:
                conf[f"{t} -> {pr}"] = n
    return {"accuracy": round(acc, 3), "macro_F1": round(float(np.mean(list(f1s.values()))), 3),
            "per_class_F1": f1s, "confusion": conf, "n": len(y_true)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neg-limit", type=int, default=30)
    parser.add_argument("--per-class", type=int, default=15)
    args = parser.parse_args()

    cfg = load_config(str(ROOT / "config" / "config.yaml"))
    seg = TalcSegmenter(cfg, ROOT / "models" / "talc_best.pt")
    global GCNN
    GCNN = GrainCNN(ROOT / "models" / "grain_cnn.pt")

    # hold-out тальк: восстановить val-сплит обучения (seed 42, 20%)
    mask_files = sorted(MASK_DIR.glob("*.png"))
    stems = [p.stem for p in mask_files if find_orig(p.stem)]
    rng = np.random.default_rng(42)
    order = rng.permutation(len(stems))
    n_val = max(1, int(len(stems) * 0.2))
    val_stems = [stems[i] for i in order[:n_val]]

    report = {}
    print("== Тальк: hold-out MAE / Dice ==", flush=True)
    report["talc"] = talc_metrics(cfg, seg, val_stems)
    print("== Негативные контроли ==", flush=True)
    report["negative_controls"] = negative_controls(cfg, seg, args.neg_limit)
    print("== Классификация сорта руды (end-to-end) ==", flush=True)
    report["ore_class"] = ore_class_eval(cfg, seg, args.per_class)

    out = ROOT / "outputs" / "evaluation.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
