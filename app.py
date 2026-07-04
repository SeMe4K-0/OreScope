"""OreScope backend — FastAPI под SPA-фронт AlloyScope (партии → образцы → анализ → ревью).

Контракт API повторяет D:/nornikel/app.py (против него написан фронт), но анализ
идёт через orescope-пайплайн v3: цветонормализация доменов, U-Net тальковой фазы,
зоны оталькования, CNN срастаний. Очередь обрабатывается фоновым воркером.
"""
from __future__ import annotations

import asyncio
import gc
import io
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response

from src.grain_cnn import GrainCNN
from src.io_utils import imwrite
from src.pipeline import analyze_image, make_overlay, make_uncertainty_vis
from src.talc import TalcSegmenter
from utils import load_config, setup_logging

ROOT = Path(__file__).resolve().parent
cfg = load_config(str(ROOT / "config" / "config.yaml"))
setup_logging(cfg)
logger = logging.getLogger("orescope")

FRONTEND_DIR = ROOT / "frontend"
FILES_DIR = ROOT / "outputs" / "webapp"
FILES_DIR.mkdir(parents=True, exist_ok=True)

SEGMENTER = TalcSegmenter(cfg, ROOT / "models" / "talc_best.pt")
GRAIN_CNN = GrainCNN(ROOT / "models" / "grain_cnn.pt")

DISPLAY_MAX = 2000
ORE_CLASSES = ["Рядовая", "Труднообогатимая", "Оталькованная"]
CLASS_CAP = {"рядовая": "Рядовая", "труднообогатимая": "Труднообогатимая",
             "оталькованная": "Оталькованная", "не классифицирована": "Не классифицирована"}
# палитра маски = контракт applyReal (пиксели маски сопоставляются с rgb палитры)
PALETTE = [
    {"idx": 0, "name": "Матрица", "rgb": [128, 132, 126]},
    {"idx": 1, "name": "Обычные срастания", "rgb": [0, 200, 0]},
    {"idx": 2, "name": "Тонкие срастания", "rgb": [255, 0, 0]},
    {"idx": 3, "name": "Тальк", "rgb": [40, 90, 255]},
]

BATCHES: dict[str, dict] = {}
SAMPLES: dict[str, dict] = {}
_LOCK = threading.Lock()

STATE_DIR = FILES_DIR / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_VERSION = 3  # менять при изменении семантики результатов -> кэш пересчитается
# v3: абсолютная ветка детектора сульфидов (массивные агрегаты, кейс RYAD-110)


def _persist(sample: dict) -> None:
    """Результаты и вердикты переживают рестарт сервера (JSON на образец)."""
    import json
    try:
        (STATE_DIR / f"{sample['id']}.json").write_text(
            json.dumps({"v": RESULTS_VERSION, "status": sample["status"],
                        "results": sample["results"],
                        "expert_verdict": sample["expert_verdict"],
                        "expert_ore_class": sample["expert_ore_class"]},
                       ensure_ascii=False), encoding="utf-8")
    except Exception:
        logger.exception("persist failed for %s", sample["id"])


def _restore(sample: dict) -> bool:
    import json
    p = STATE_DIR / f"{sample['id']}.json"
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False
    if data.get("v") != RESULTS_VERSION or not data.get("results"):
        return False
    sample["results"] = data["results"]
    sample["status"] = data.get("status", "Готово")
    sample["expert_verdict"] = data.get("expert_verdict")
    sample["expert_ore_class"] = data.get("expert_ore_class")
    return True


def _next_sample_id() -> str:
    return f"NN-{4146 + len(SAMPLES)}"


# ── Анализ одного образца через пайплайн v3 ──────────────────────────────────
def _grain_polys(disp_mask: np.ndarray, per_class: int = 30) -> tuple[list[dict], int]:
    """Контуры срастаний обоих классов для карты (нормированные координаты).

    Тонкие (класс 2) -> sev 3 (красные), обычные (класс 1) -> sev 1 (зелёные)."""
    h, w = disp_mask.shape
    polys: list[dict] = []
    fine_count = 0
    for cls, sev, kind in ((2, 3, "Тонкое срастание"), (1, 1, "Обычное срастание")):
        m = (disp_mask == cls).astype(np.uint8)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
        if cls == 2:
            fine_count = max(n - 1, 0)
        order = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1][:per_class] + 1
        for lab in order:
            area = int(stats[lab, cv2.CC_STAT_AREA])
            if area < 12:
                continue
            comp = (labels == lab).astype(np.uint8)
            cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cs:
                continue
            cnt = max(cs, key=cv2.contourArea)
            eps = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
            if len(approx) < 3:
                continue
            cx, cy = cents[lab]
            polys.append({
                "x": round(float(cx) / w, 4), "y": round(float(cy) / h, 4),
                "r": round(float(np.sqrt(area / np.pi)) / max(h, w), 4),
                "sev": sev, "kind": kind, "replacement": None,
                "pts": [{"x": round(float(px) / w, 4), "y": round(float(py) / h, 4)}
                        for px, py in approx],
            })
    return polys, fine_count


def _run_analysis_sync(sample: dict) -> dict:
    sid = sample["id"]
    img_path = sample["files"].get("om") or sample["files"].get("sem")
    if not img_path:
        raise ValueError("No OM/SEM image uploaded for this sample")

    res = analyze_image(img_path, cfg, SEGMENTER,
                        um_per_px=sample.get("um_per_px"), grain_cnn=GRAIN_CNN)
    v = res.verdict
    h, w = res.class_mask.shape

    scale = min(1.0, DISPLAY_MAX / max(h, w))
    if scale < 1.0:
        dw, dh = int(w * scale), int(h * scale)
        disp_rgb = cv2.resize(res.rgb, (dw, dh), interpolation=cv2.INTER_AREA)
        disp_mask = cv2.resize(res.class_mask, (dw, dh), interpolation=cv2.INTER_NEAREST)
    else:
        disp_rgb, disp_mask = res.rgb, res.class_mask

    # тонкие фазы (срастания, тальк) слегка расширяются в фон, чтобы рассеянные
    # частицы выживали при downsample маски в браузере (nearest, ~1024px)
    disp_mask = disp_mask.copy()
    for cls in (2, 3):
        m = cv2.dilate((disp_mask == cls).astype(np.uint8), np.ones((3, 3), np.uint8))
        disp_mask[(m > 0) & (disp_mask == 0)] = cls

    mask_color = np.zeros((*disp_mask.shape, 3), np.uint8)
    for p in PALETTE:
        mask_color[disp_mask == p["idx"]] = p["rgb"]

    imwrite(FILES_DIR / f"{sid}_original.jpg", disp_rgb)
    imwrite(FILES_DIR / f"{sid}_mask.png", mask_color)
    imwrite(FILES_DIR / f"{sid}_overlay.jpg", make_overlay(res, max_side=DISPLAY_MAX))
    imwrite(FILES_DIR / f"{sid}_uncertainty.jpg", make_uncertainty_vis(res, max_side=DISPLAY_MAX))

    defect_polys, fine_count = _grain_polys(disp_mask)

    ore_class_final = CLASS_CAP.get(v.ore_class, v.ore_class)
    confidence = round(float(res.talc_confidence) * 100, 1)
    review = bool(v.review_needed)
    review_reason = "; ".join(v.warnings) if v.warnings else ""

    matrix_pct = round(max(0.0, 100.0 - v.sulfide_percent - v.talc_percent), 2)
    phase_fractions = {
        "Матрица": matrix_pct,
        "Обычные срастания": v.ordinary_percent,
        "Тонкие срастания": v.fine_percent,
        "Тальк": v.talc_percent,
    }

    img_lines = [
        f"Сегментация: U-Net v3 (тальковая фаза) + CNN срастаний, модель талька — {res.talc_model}",
        f"Сульфиды всего: {v.sulfide_percent:.2f}% площади "
        f"(обычные — {v.ordinary_percent:.2f}%, тонкие — {v.fine_percent:.2f}%)",
        f"Доля тонких среди сульфидов: {v.fine_share:.0f}%",
        f"Тальк: фаза {v.talc_percent:.2f}% · зоны оталькования {v.talc_zone_percent:.2f}% "
        f"(порог правила 10% — по зонам)",
        "Зона оталькования — не сама фаза талька, а участок породы вокруг её скоплений: "
        "тальк в шлифе рассеян мелкими вкраплениями, и площадь одной только фазы обычно "
        "меньше 10% даже у явно оталькованной руды. Зона восстанавливается из фазы по "
        "локальной плотности вкраплений (включает матрицу и сульфиды между ними) и "
        "калибрована под контуры, которыми геолог обводил зоны при разметке — именно к "
        "ней применяется порог 10% из ТЗ.",
    ]
    cls_lines = [
        f"Класс руды: {ore_class_final} — уверенность модели талька {confidence:.0f}%",
        f"Зоны оталькования {v.talc_zone_percent:.1f}% ({'>' if v.talc_zone_percent > 10 else '≤'} 10%); "
        f"преобладание {'тонких' if v.fine_percent >= v.ordinary_percent else 'обычных'} срастаний",
        v.conclusion,
        f"Проверка экспертом: {'требуется — ' + review_reason if review else 'штатная (этап потока)'}",
    ]

    return {
        "original_url": f"/api/files/{sid}_original.jpg",
        "overlay_url": f"/api/files/{sid}_overlay.jpg",
        "mask_url": f"/api/files/{sid}_mask.png",
        "uncertainty_url": f"/api/files/{sid}_uncertainty.jpg",
        "palette": PALETTE,
        "bg_idx": -1,
        "defect_polys": defect_polys,
        "modality": "OM",
        "phase_fractions": phase_fractions,
        "ore_class": ore_class_final,
        "class_probs": {},
        "model_used": "нейросеть (v3)",
        "conclusion": v.conclusion,
        "sulfide_pct": v.sulfide_percent,
        "talc_pct": v.talc_percent,
        "talc_zone_pct": v.talc_zone_percent,
        "normal_share": round(100.0 - v.fine_share, 1),
        "fine_share": v.fine_share,
        "um_per_px": sample.get("um_per_px"),
        "dominant_phase": ore_class_final,
        "dominant_pct": v.talc_percent,
        "grain_count": int(res.grain_stats.get("n_grains", 0)),
        "fine_count": fine_count,
        "review_needed": review,
        "review_reason": review_reason,
        "confidence": confidence,
        "report": {"image": img_lines, "classification": cls_lines},
    }


# ── FastAPI ──────────────────────────────────────────────────────────────────
app = FastAPI(title="OreScope API")


@app.get("/", response_class=HTMLResponse)
async def root():
    return (FRONTEND_DIR / "AlloyScope.dc.html").read_text(encoding="utf-8")


@app.get("/support.js")
async def support_js():
    return FileResponse(str(FRONTEND_DIR / "support.js"), media_type="application/javascript")


@app.get("/api/files/{filename}")
async def serve_file(filename: str):
    path = FILES_DIR / filename
    if not path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(str(path))


@app.get("/api/batches")
async def list_batches():
    result = []
    for bid, b in BATCHES.items():
        sample_ids = [sid for sid, s in SAMPLES.items() if s["batch_id"] == bid]
        ready = sum(1 for sid in sample_ids
                    if SAMPLES[sid]["status"] in ("Готово", "Подтверждён", "Отклонён"))
        result.append({**b, "sample_count": len(sample_ids), "ready_count": ready,
                       "samples": sample_ids})
    return result


@app.post("/api/batches")
async def create_batch(body: dict):
    bid = body.get("id") or f"P-{len(BATCHES) + 1}"
    label = body.get("label", bid)
    if bid not in BATCHES:
        BATCHES[bid] = {"id": bid, "label": label, "sub": "OM-шлифы",
                        "created": str(datetime.now().date())}
    return BATCHES[bid]


@app.delete("/api/batches/{batch_id}")
async def delete_batch(batch_id: str):
    if batch_id not in BATCHES:
        raise HTTPException(404, "Batch not found")
    for sid in [s for s, x in SAMPLES.items() if x["batch_id"] == batch_id]:
        SAMPLES.pop(sid, None)
        for f in FILES_DIR.glob(f"{sid}_*"):
            try:
                f.unlink()
            except OSError:
                pass
    BATCHES.pop(batch_id, None)
    return {"deleted": batch_id}


def _summary_row(s: dict) -> dict:
    """Облегчённая версия образца для списков партии: без defect_polys/palette/report —
    те тяжёлые поля нужны только на экране конкретного образца (GET /api/samples/{id}).
    Полный объект на партию из 500+ образцов раздувал ответ и вешал список на минуты."""
    r = s.get("results") or {}
    return {
        "id": s["id"], "batch_id": s["batch_id"], "status": s["status"],
        "modalities": s.get("modalities", []), "created": s.get("created"),
        "expert_verdict": s.get("expert_verdict"), "expert_ore_class": s.get("expert_ore_class"),
        "results": ({
            "ore_class": r.get("ore_class"), "talc_pct": r.get("talc_pct"),
            "talc_zone_pct": r.get("talc_zone_pct"), "sulfide_pct": r.get("sulfide_pct"),
            "fine_count": r.get("fine_count"), "confidence": r.get("confidence"),
            "review_needed": r.get("review_needed"), "review_reason": r.get("review_reason"),
        } if r else None),
    }


@app.get("/api/batches/{batch_id}/samples")
async def list_samples(batch_id: str):
    if batch_id not in BATCHES:
        raise HTTPException(404)
    return [_summary_row(s) for s in SAMPLES.values() if s["batch_id"] == batch_id]


@app.get("/api/samples/{sample_id}")
async def get_sample(sample_id: str):
    if sample_id not in SAMPLES:
        raise HTTPException(404)
    return SAMPLES[sample_id]


@app.delete("/api/samples/{sample_id}")
async def delete_sample(sample_id: str):
    if sample_id not in SAMPLES:
        raise HTTPException(404, "Sample not found")
    SAMPLES.pop(sample_id, None)
    for f in FILES_DIR.glob(f"{sample_id}_*"):
        try:
            f.unlink()
        except OSError:
            pass
    return {"deleted": sample_id}


@app.post("/api/samples/{sample_id}/review")
async def review_sample(sample_id: str, body: dict):
    """Вердикт эксперта. body: {verdict: approved|rejected, ore_class?: класс эксперта}."""
    if sample_id not in SAMPLES:
        raise HTTPException(404, "Sample not found")
    verdict = body.get("verdict")
    if verdict not in ("approved", "rejected"):
        raise HTTPException(400, "verdict must be 'approved' or 'rejected'")
    sample = SAMPLES[sample_id]
    sample["expert_verdict"] = verdict
    sample["status"] = "Подтверждён" if verdict == "approved" else "Отклонён"

    ore_class = body.get("ore_class")
    if ore_class in ORE_CLASSES:
        sample["expert_ore_class"] = ore_class
        if sample.get("results"):
            sample["results"]["ore_class"] = ore_class
            sample["results"]["expert_overridden"] = True
    _persist(sample)
    return sample


@app.post("/api/samples")
async def create_sample(body: dict):
    batch_id = body.get("batch_id", "P-1")
    sid = body.get("id") or _next_sample_id()
    if batch_id not in BATCHES:
        BATCHES[batch_id] = {"id": batch_id, "label": batch_id, "sub": "только что",
                             "created": str(datetime.now().date())}
    SAMPLES[sid] = {
        "id": sid, "batch_id": batch_id, "status": "Новый", "files": {},
        "results": None, "created": datetime.now().strftime("%H:%M"),
        "modalities": [], "expert_verdict": None, "expert_ore_class": None,
        "um_per_px": body.get("um_per_px"),
    }
    return SAMPLES[sid]


@app.post("/api/samples/{sample_id}/files")
async def upload_sample_file(sample_id: str, modality: str = Form(...),
                             file: UploadFile = File(...)):
    if sample_id not in SAMPLES:
        raise HTTPException(404, "Sample not found")
    content = await file.read()
    ext = Path(file.filename).suffix.lower()
    dest = FILES_DIR / f"{sample_id}_{modality}{ext}"
    dest.write_bytes(content)
    SAMPLES[sample_id]["files"][modality] = str(dest)
    mods = SAMPLES[sample_id].setdefault("modalities", [])
    if modality.upper() not in mods:
        mods.append(modality.upper())
    SAMPLES[sample_id]["status"] = "Загружено"
    return {"path": str(dest), "modality": modality,
            "preview_url": f"/api/files/{dest.name}"}


@app.post("/api/samples/{sample_id}/analyze")
async def analyze_sample(sample_id: str, body: dict = {}):
    if sample_id not in SAMPLES:
        raise HTTPException(404, "Sample not found")
    sample = SAMPLES[sample_id]
    if body.get("um_per_px") is not None:
        try:
            sample["um_per_px"] = float(body["um_per_px"]) or None
        except (TypeError, ValueError):
            pass
    if sample.get("results"):
        return {"status": "done", "results": sample["results"]}
    # если воркер уже обрабатывает — дождаться
    if sample["status"] == "Обработка":
        for _ in range(600):
            await asyncio.sleep(1.0)
            if sample.get("results") or sample["status"] == "Ошибка":
                break
        if sample.get("results"):
            return {"status": "done", "results": sample["results"]}
        raise HTTPException(500, "Analysis failed")
    with _LOCK:
        if sample["status"] == "Обработка":
            raise HTTPException(409, "Already processing")
        sample["status"] = "Обработка"
    try:
        results = await asyncio.to_thread(_run_analysis_sync, sample)
        sample["results"] = results
        sample["status"] = "Проверка" if results.get("review_needed") else "Готово"
        _persist(sample)
        return {"status": "done", "results": results}
    except Exception as e:
        sample["status"] = "Ошибка"
        logger.exception("Analysis failed for %s", sample_id)
        raise HTTPException(500, detail=str(e))


# ── Экспорт ──────────────────────────────────────────────────────────────────
_CSV_HEAD = ("id;партия;класс модели;класс эксперта;вердикт;тальк фаза %;зоны оталькования %;"
             "сульфиды %;обычные %;тонкие %;уверенность %;статус;файл\n")


def _csv_row(s: dict) -> str:
    r = s.get("results") or {}
    return ";".join(str(x if x is not None else "") for x in [
        s["id"], s["batch_id"], r.get("ore_class", ""), s.get("expert_ore_class") or "",
        s.get("expert_verdict") or "", r.get("talc_pct", ""), r.get("talc_zone_pct", ""),
        r.get("sulfide_pct", ""), r.get("phase_fractions", {}).get("Обычные срастания", ""),
        r.get("phase_fractions", {}).get("Тонкие срастания", ""), r.get("confidence", ""),
        s.get("status", ""), s.get("source", ""),
    ]) + "\n"


def _csv_response(rows: list[dict], filename: str) -> Response:
    text = _CSV_HEAD + "".join(_csv_row(s) for s in rows)
    return Response(content="﻿" + text, media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/samples/{sample_id}/export.csv")
async def export_sample_csv(sample_id: str):
    if sample_id not in SAMPLES:
        raise HTTPException(404, "Sample not found")
    return _csv_response([SAMPLES[sample_id]], f"{sample_id}.csv")


@app.get("/api/batches/{batch_id}/export.csv")
async def export_batch_csv(batch_id: str):
    if batch_id not in BATCHES:
        raise HTTPException(404, "Batch not found")
    rows = [s for s in SAMPLES.values() if s["batch_id"] == batch_id]
    return _csv_response(rows, f"{batch_id}.csv")


def _sample_pdf(sample: dict) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas as pdfcanvas

    font = "Helvetica"
    for cand in (r"C:\Windows\Fonts\arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(cand).exists():
            pdfmetrics.registerFont(TTFont("Body", cand))
            font = "Body"
            break

    r = sample.get("results") or {}
    buf = io.BytesIO()
    c = pdfcanvas.Canvas(buf, pagesize=A4)
    W, H = A4
    y = H - 20 * mm
    c.setFont(font, 16)
    c.drawString(20 * mm, y, f"OreScope — отчёт по образцу {sample['id']}")
    y -= 8 * mm
    c.setFont(font, 10)
    c.drawString(20 * mm, y, f"Партия: {sample['batch_id']} · файл: {sample.get('source', '—')} · "
                             f"статус: {sample.get('status')}")
    y -= 10 * mm
    c.setFont(font, 12)
    c.drawString(20 * mm, y, f"Класс: {r.get('ore_class', '—')}"
                 + (f" (эксперт: {sample['expert_ore_class']})" if sample.get("expert_ore_class") else ""))
    y -= 8 * mm
    c.setFont(font, 9)
    for line in ([r.get("conclusion", "")] + (r.get("report", {}).get("image", []))
                 + (r.get("report", {}).get("classification", []))):
        for chunk in [line[i:i + 110] for i in range(0, len(line), 110)] or [""]:
            c.drawString(20 * mm, y, chunk)
            y -= 5 * mm
    ov = FILES_DIR / f"{sample['id']}_overlay.jpg"
    if ov.exists():
        try:
            c.drawImage(str(ov), 20 * mm, max(y - 110 * mm, 15 * mm),
                        width=170 * mm, height=105 * mm, preserveAspectRatio=True, anchor="sw")
        except Exception:
            pass
    c.showPage()
    c.save()
    return buf.getvalue()


@app.get("/api/samples/{sample_id}/report.pdf")
async def export_sample_pdf(sample_id: str):
    if sample_id not in SAMPLES:
        raise HTTPException(404, "Sample not found")
    if not SAMPLES[sample_id].get("results"):
        raise HTTPException(400, "Sample not analyzed yet")
    pdf = await asyncio.to_thread(_sample_pdf, SAMPLES[sample_id])
    return Response(content=pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{sample_id}.pdf"'})


@app.get("/api/batches/{batch_id}/report.pdf")
async def export_batch_pdf(batch_id: str):
    if batch_id not in BATCHES:
        raise HTTPException(404, "Batch not found")
    rows = [s for s in SAMPLES.values() if s["batch_id"] == batch_id and s.get("results")]
    if not rows:
        raise HTTPException(400, "No analyzed samples in batch")
    pdf = await asyncio.to_thread(_sample_pdf, rows[0])  # титульный по первому + CSV основной путь
    return Response(content=pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{batch_id}.pdf"'})


# ── Посев реального датасета: папка = ground-truth класс ────────────────────
ORE_DATA = ROOT / "ore_data"
_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_ORE_BATCHES = [
    ("RYAD", "Рядовые руды", "Рядовая",
     ["Фото руд по сортам. ч1/Рядовые руды", "Фото руд по сортам. ч2/рядовые"]),
    ("HARD", "Труднообогатимые руды", "Труднообогатимая",
     ["Фото руд по сортам. ч1/Труднообогатимые руды", "Фото руд по сортам. ч2/тонкие"]),
    ("TALC", "Оталькованные руды", "Оталькованная",
     ["Фото руд по сортам. ч1/Оталькованные руды", "Фото руд по сортам. ч2/оталькованные"]),
]


def _seed_ore_data() -> None:
    """Все фото датасета: класс из имени папки; панорамы отдельной партией (в конец очереди)."""
    if not ORE_DATA.exists():
        logger.warning("Датасет руд не найден: %s", ORE_DATA)
        return
    total = 0
    for bid, blabel, gt_class, subdirs in _ORE_BATCHES:
        imgs: list[Path] = []
        for sd in subdirs:
            d = ORE_DATA / sd
            if d.exists():
                imgs += [p for p in sorted(d.iterdir())
                         if p.is_file() and p.suffix.lower() in _IMG_EXT]
        if not imgs:
            continue
        BATCHES[bid] = {"id": bid, "label": blabel, "sub": f"эталон: {gt_class} · {len(imgs)} фото",
                        "created": str(datetime.now().date())}
        for i, p in enumerate(imgs):
            sid = f"{bid}-{i:03d}"
            SAMPLES[sid] = {
                "id": sid, "batch_id": bid, "status": "Загружено",
                "files": {"om": str(p)}, "results": None,
                "created": datetime.now().strftime("%H:%M"), "modalities": ["OM"],
                "source": p.name, "ground_truth": gt_class,
                "expert_verdict": None, "expert_ore_class": None, "um_per_px": None,
            }
            _restore(SAMPLES[sid])
            total += 1
        logger.info("Партия %s (%s): %d образцов", bid, gt_class, len(imgs))

    pano_dir = ORE_DATA / "Панорамы"
    if pano_dir.exists():
        panos = [p for p in sorted(pano_dir.iterdir())
                 if p.is_file() and p.suffix.lower() in _IMG_EXT]
        if panos:
            BATCHES["PANO"] = {"id": "PANO", "label": "Панорамы",
                               "sub": f"панорамные шлифы · {len(panos)}",
                               "created": str(datetime.now().date())}
            for i, p in enumerate(panos):
                sid = f"PANO-{i:03d}"
                SAMPLES[sid] = {
                    "id": sid, "batch_id": "PANO", "status": "Загружено",
                    "files": {"om": str(p)}, "results": None,
                    "created": datetime.now().strftime("%H:%M"), "modalities": ["OM"],
                    "source": p.name, "ground_truth": None,
                    "expert_verdict": None, "expert_ore_class": None, "um_per_px": None,
                }
                _restore(SAMPLES[sid])
                total += 1
    logger.info("Засеяно %d образцов руд из ore_data", total)


_seed_ore_data()


# ── Фоновый воркер: последовательная автообработка очереди ──────────────────
def _worker_loop() -> None:
    time.sleep(3.0)
    while True:
        try:
            with _LOCK:
                pending = next(
                    (s for s in SAMPLES.values()
                     if s.get("status") in ("Загружено", "Новый")
                     and (s.get("files", {}).get("om") or s.get("files", {}).get("sem"))),
                    None,
                )
                if pending is not None:
                    pending["status"] = "Обработка"
            if pending is None:
                time.sleep(2.0)
                continue
            try:
                results = _run_analysis_sync(pending)
                pending["results"] = results
                pending["status"] = "Проверка" if results.get("review_needed") else "Готово"
                _persist(pending)
                logger.info("auto-processed %s -> %s", pending["id"], pending["status"])
            except Exception:
                pending["status"] = "Ошибка"
                logger.exception("auto-process failed for %s", pending["id"])
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception:
            logger.exception("worker loop error")
            time.sleep(2.0)


threading.Thread(target=_worker_loop, daemon=True, name="orescope-worker").start()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("ORESCOPE_PORT", "7860")))
