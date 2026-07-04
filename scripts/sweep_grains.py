"""Расширенная развёртка правил классификации срастаний по кэшу признаков зёрен."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs" / "calib_grains.parquet"


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1s = []
    for cls in ("ordinary", "fine"):
        tp = ((y_pred == cls) & (y_true == cls)).sum()
        fp = ((y_pred == cls) & (y_true != cls)).sum()
        fn = ((y_pred != cls) & (y_true == cls)).sum()
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f1s))


def main() -> None:
    df = pd.read_parquet(CACHE)
    images = df.groupby("image").agg(label=("label", "first"), subset=("subset", "first"))
    y = images["label"].values
    tot_area = df.groupby("image")["area"].sum()

    best = None
    # Правило A: доля площади тонких (dens<dt | repl>=rt) >= sc
    for rt in [0.2, 0.3, 0.4, 2.0]:
        for dt in [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
            fine_mask = (df["replacement"] >= rt) | (df["density"] < dt)
            fine_area = df["area"].where(fine_mask, 0).groupby(df["image"]).sum()
            share = (fine_area / tot_area).reindex(images.index).fillna(0)
            for sc in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
                pred = np.where(share >= sc, "fine", "ordinary")
                f1 = macro_f1(y, pred)
                if best is None or f1 > best["f1"]:
                    best = {"rule": "share", "rt": rt, "dt": dt, "share_cutoff": sc, "f1": round(f1, 4)}

    # Правило B: средневзвешенная по площади плотность < wt
    wd = (df["density"] * df["area"]).groupby(df["image"]).sum() / tot_area
    wd = wd.reindex(images.index).fillna(1)
    for wt in np.arange(0.4, 0.95, 0.025):
        pred = np.where(wd < wt, "fine", "ordinary")
        f1 = macro_f1(y, pred)
        if f1 > best["f1"]:
            best = {"rule": "wmean_density", "wt": round(float(wt), 3), "f1": round(f1, 4)}

    # Правило C: средневзвешенное замещение
    wr = (df["replacement"] * df["area"]).groupby(df["image"]).sum() / tot_area
    wr = wr.reindex(images.index).fillna(0)
    for wt in np.arange(0.05, 0.6, 0.025):
        pred = np.where(wr >= wt, "fine", "ordinary")
        f1 = macro_f1(y, pred)
        if f1 > best["f1"]:
            best = {"rule": "wmean_replacement", "wt": round(float(wt), 3), "f1": round(f1, 4)}

    print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
