"""Полный анализ изображения/папки: маски, оверлей, CSV/PDF/GeoJSON, JSON-лог.

Запуск:
  py -3.11 scripts/analyze.py --input "ore_data/Панорамы/4.jpg"
  py -3.11 scripts/analyze.py --input "ore_data/Панорамы" --out outputs/panoramas
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.exporters import export_csv, export_geojson, export_pdf  # noqa: E402
from src.grain_cnn import GrainCNN  # noqa: E402
from src.io_utils import imwrite  # noqa: E402
from src.pipeline import analyze_image, make_overlay, make_uncertainty_vis, save_params_log  # noqa: E402
from src.talc import TalcSegmenter  # noqa: E402
from utils import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="outputs/analysis")
    parser.add_argument("--um-per-px", type=float, default=0.0)
    parser.add_argument("--checkpoint", default="models/talc_best.pt")
    args = parser.parse_args()

    cfg = load_config(str(ROOT / "config" / "config.yaml"))
    seg = TalcSegmenter(cfg, ROOT / args.checkpoint)
    gcnn = GrainCNN(ROOT / "models" / "grain_cnn.pt")
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    src = Path(args.input)
    if not src.is_absolute():
        src = ROOT / src
    files = ([src] if src.is_file() else
             sorted(p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")))
    um = args.um_per_px or None

    for path in files:
        print(f"=== {path.name} ===", flush=True)
        try:
            res = analyze_image(path, cfg, seg, um_per_px=um, grain_cnn=gcnn)
        except Exception as e:
            print(f"[ERR] {e}")
            continue
        stem = path.stem
        imwrite(out_dir / f"{stem}_overlay.jpg", make_overlay(res))
        imwrite(out_dir / f"{stem}_uncertainty.jpg", make_uncertainty_vis(res))
        export_csv(res, out_dir / f"{stem}_metrics.csv", um)
        export_pdf(res, out_dir / f"{stem}_report.pdf", sample_id=stem, um_per_px=um)
        n_poly = export_geojson(res, out_dir / f"{stem}.geojson",
                                min_area_px=int(cfg.export.geojson_min_area_px),
                                eps_frac=float(cfg.export.geojson_epsilon_frac))
        save_params_log(res, out_dir / f"{stem}_params.json")
        v = res.verdict
        print(f"  {v.ore_class.upper()} | тальк {v.talc_percent}% | сульфиды {v.sulfide_percent}% "
              f"(об. {v.ordinary_percent}% / тонк. {v.fine_percent}%) | "
              f"talc-модель {res.talc_model} | GeoJSON {n_poly} полиг. | {res.elapsed_s:.0f} c",
              flush=True)
        del res
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
