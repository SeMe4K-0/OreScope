"""Свип порогов правила оталькованности по фактическим предсказаниям.

Собирает (файл, метка, фаза%, зоны%) на той же выборке, что evaluate.py
(seed 1, по 15 фото/класс/источник), затем перебирает пороги:
  оталькованная <=> зоны > z_thr И фаза > p_floor.
Кэширует предсказания в outputs/rule_sweep_data.csv (--recompute для пересбора).

Запуск: py -3.11 scripts/sweep_rule.py [--recompute]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.grain_cnn import GrainCNN  # noqa: E402
from src.pipeline import analyze_image  # noqa: E402
from src.talc import TalcSegmenter  # noqa: E402
from utils import load_config  # noqa: E402

CH1 = ROOT / "ore_data" / "Фото руд по сортам. ч1"
CH2 = ROOT / "ore_data" / "Фото руд по сортам. ч2"
CACHE = ROOT / "outputs" / "rule_sweep_data.csv"


def collect(per_class: int = 15) -> list[dict]:
    cfg = load_config(str(ROOT / "config" / "config.yaml"))
    seg = TalcSegmenter(cfg, ROOT / "models" / "talc_best.pt")
    gcnn = GrainCNN(ROOT / "models" / "grain_cnn.pt")
    rng = np.random.default_rng(1)
    sources = [
        (CH1 / "Рядовые руды", "рядовая", "ч1"),
        (CH1 / "Труднообогатимые руды", "труднообогатимая", "ч1"),
        (CH1 / "Оталькованные руды", "оталькованная", "ч1"),
        (CH2 / "рядовые", "рядовая", "ч2"),
        (CH2 / "тонкие", "труднообогатимая", "ч2"),
        (CH2 / "оталькованные", "оталькованная", "ч2"),
    ]
    rows = []
    for folder, label, dom in sources:
        cand = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        for p in rng.choice(cand, size=min(per_class, len(cand)), replace=False):
            res = analyze_image(p, cfg, seg, grain_cnn=gcnn)
            v = res.verdict
            rows.append({"file": p.name, "label": label, "domain": dom,
                         "phase": v.talc_percent, "zone": v.talc_zone_percent,
                         "fine": v.fine_percent, "ordinary": v.ordinary_percent,
                         "sulfide": v.sulfide_percent})
            print(f"{dom} {label:16s} {p.name:24s} фаза {v.talc_percent:5.2f}% "
                  f"зоны {v.talc_zone_percent:5.2f}%", flush=True)
    with open(CACHE, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def sweep(rows: list[dict]) -> None:
    classes = ("рядовая", "труднообогатимая", "оталькованная")
    best = None
    for z_thr in (8.0, 10.0, 12.0, 15.0, 20.0):
        for p_floor in (0.0, 2.0, 3.0, 4.0, 4.5, 5.0):
            y_true, y_pred = [], []
            for r in rows:
                talc_hit = float(r["zone"]) > z_thr and float(r["phase"]) > p_floor
                if talc_hit:
                    pred = "оталькованная"
                elif float(r["fine"]) >= float(r["ordinary"]):
                    pred = "труднообогатимая"
                else:
                    pred = "рядовая"
                y_true.append(r["label"])
                y_pred.append(pred)
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            acc = float((y_true == y_pred).mean())
            f1s = []
            for cls in classes:
                tp = ((y_pred == cls) & (y_true == cls)).sum()
                fp = ((y_pred == cls) & (y_true != cls)).sum()
                fn = ((y_pred != cls) & (y_true == cls)).sum()
                f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
            mf1 = float(np.mean(f1s))
            line = (f"zone>{z_thr:4.1f} & phase>{p_floor:3.1f}: acc={acc:.3f} "
                    f"macroF1={mf1:.3f} (ряд {f1s[0]:.2f} / трудн {f1s[1]:.2f} / от {f1s[2]:.2f})")
            print(line)
            if best is None or mf1 > best[0]:
                best = (mf1, line)
    print("\nЛУЧШЕЕ:", best[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    if CACHE.exists() and not args.recompute:
        with open(CACHE, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        print(f"кэш: {len(rows)} строк из {CACHE.name}")
    else:
        rows = collect()
    sweep(rows)


if __name__ == "__main__":
    main()
