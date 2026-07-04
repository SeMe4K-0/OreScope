"""Экспорт результатов: CSV, GeoJSON (3 класса, упрощение Douglas-Peucker),
PDF-отчёт (кириллица через системный Arial), JSON-лог параметров."""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from src.pipeline import AnalysisResult, make_overlay, metrics_table

logger = logging.getLogger(__name__)

GEO_CLASSES = {1: "ordinary_sulfide", 2: "fine_sulfide", 3: "talc"}


def export_csv(res: AnalysisResult, path: str | Path, um_per_px: float | None = None) -> None:
    rows = metrics_table(res, um_per_px)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["metric", "value_pct"] + (["area_mm2"] if um_per_px else [])
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
        w.writerow({})
        w.writerow({"metric": f"Класс руды: {res.verdict.ore_class}"})
        w.writerow({"metric": res.verdict.conclusion})


def export_geojson(res: AnalysisResult, path: str | Path,
                   min_area_px: int = 400, eps_frac: float = 0.01) -> int:
    """Векторизация масок в GeoJSON (пиксельные координаты, y вниз).
    Смежные мелкие зёрна агрегируются closing'ом перед векторизацией."""
    features = []
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for cls, name in GEO_CLASSES.items():
        m = (res.class_mask == cls).astype(np.uint8)
        if not m.any():
            continue
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area_px:
                continue
            eps = eps_frac * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
            if len(approx) < 3:
                continue
            ring = [[int(x), int(y)] for x, y in approx] + [[int(approx[0][0]), int(approx[0][1])]]
            features.append({
                "type": "Feature",
                "properties": {"class": name, "area_px": int(area)},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            })
    geo = {"type": "FeatureCollection",
           "properties": {"crs": "pixel", "note": "y axis points down (image coordinates)"},
           "features": features}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(geo, ensure_ascii=False), encoding="utf-8")
    return len(features)


def export_pdf(res: AnalysisResult, path: str | Path, sample_id: str = "",
               um_per_px: float | None = None) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas as pdfcanvas

    font_name = "Helvetica"
    for cand, fname in [("C:/Windows/Fonts/arial.ttf", "Arial"),
                        ("C:/Windows/Fonts/calibri.ttf", "Calibri"),
                        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans")]:
        if Path(cand).exists():
            pdfmetrics.registerFont(TTFont(fname, cand))
            font_name = fname
            break

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    c = pdfcanvas.Canvas(str(path), pagesize=A4)
    W, H = A4

    c.setFont(font_name, 16)
    c.drawString(20 * mm, H - 20 * mm, f"OreScope — отчёт по образцу {sample_id}")
    c.setFont(font_name, 10)
    c.drawString(20 * mm, H - 27 * mm, res.params_log.get("timestamp", ""))

    # оверлей
    overlay = make_overlay(res, max_side=1600)
    tmp = path.with_suffix(".overlay_tmp.jpg")
    cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))[1].tofile(str(tmp))
    img_w = 170 * mm
    img_h = img_w * overlay.shape[0] / overlay.shape[1]
    c.drawImage(str(tmp), 20 * mm, H - 35 * mm - img_h, width=img_w, height=img_h)
    tmp.unlink(missing_ok=True)

    y = H - 45 * mm - img_h
    c.setFont(font_name, 12)
    c.drawString(20 * mm, y, "Количественные метрики (в % валидной площади):")
    y -= 7 * mm
    c.setFont(font_name, 10)
    for row in metrics_table(res, um_per_px):
        line = f"• {row['metric']}: {row.get('value_pct', '')}%"
        if "area_mm2" in row:
            line += f"  ({row['area_mm2']} мм²)"
        c.drawString(24 * mm, y, line)
        y -= 6 * mm

    y -= 4 * mm
    c.setFont(font_name, 12)
    c.drawString(20 * mm, y, f"Класс руды: {res.verdict.ore_class.upper()}")
    y -= 8 * mm
    c.setFont(font_name, 10)
    for chunk in _wrap(res.verdict.conclusion, 95):
        c.drawString(20 * mm, y, chunk)
        y -= 5 * mm

    c.setFont(font_name, 8)
    c.drawString(20 * mm, 12 * mm,
                 f"Модель талька: {res.talc_model} | уверенность {res.talc_confidence:.2f} | "
                 f"время анализа {res.elapsed_s:.0f} с | легенда: зелёный=обычные, красный=тонкие, синий=тальк")
    c.save()


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines
