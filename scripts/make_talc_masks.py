"""Конвертация экспертной разметки талька (синие линии от руки) в бинарные маски.

Алгоритм:
  1. Выделяем синие пиксели линии (HSV + доминирование B-канала).
  2. Замыкаем разрывы morphological closing.
  3. Барьер = замкнутые линии + рамка кадра (1 px): контуры, упирающиеся в край,
     образуют области вместе с границей кадра.
  4. Связные компоненты дополнения барьера: крупнейшая = не-тальк,
     остальные = тальк-ЗОНЫ (обведённые области). Для инвертированных случаев —
     override "bg_points": явные точки фоновых компонент.
  5. Уточнение зоны до ФАЗЫ: тальк = тёмная рассеянная фаза внутри зоны
     (темнее робастной статистики чистой матрицы вне зон), минус сульфиды.
     Зоны эксперта содержат и матрицу, и сульфиды — заливка целиком завышала долю.
     Тёмная фаза ВНЕ зон (потенциально неразмеченный тальк) -> data/talc_ignore
     (исключается из лосса при обучении).
  6. Sanity-check долей + contact-sheet с оверлеем для визуальной проверки всех пар.

Запуск:  py -3.11 scripts/make_talc_masks.py [--close-k 25] [--k-dark 1.0] [--no-refine]
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
from src.sulfides import filter_small, sulfide_mask  # noqa: E402
from utils import load_config  # noqa: E402

ANNOT_DIR = ROOT / "ore_data" / "Фото руд по сортам. ч1" / "Оталькованные руды" / "Области оталькования"
ORIG_DIR = ANNOT_DIR.parent
OUT_DIR = ROOT / "data" / "talc_masks"
IGNORE_DIR = ROOT / "data" / "talc_ignore"
REVIEW_DIR = ROOT / "outputs" / "talc_review"
# {stem: {"extra_lines": [[x1,y1,x2,y2],...], "pair_dist": N, "border_ext": N,
#         "bg_points": [[x,y],...]  — точки фоновых компонент (инвертированные зоны),
#         "exclude": true           — разметка неоднозначна, файл не используется}}
OVERRIDES = ROOT / "data" / "talc_masks_overrides.json"


def extract_blue_line(rgb: np.ndarray) -> np.ndarray:
    """Маска пикселей синей линии аннотации."""
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    dominance = (b - np.maximum(r, g)) > 40

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hsv_blue = (h > 100) & (h < 140) & (s > 100) & (v > 60)

    return (dominance & hsv_blue).astype(np.uint8)


def _skeleton_endpoints(line: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Скелет линии и список его концевых точек (y, x)."""
    from skimage.morphology import skeletonize

    skel = skeletonize(line.astype(bool)).astype(np.uint8)
    kernel = np.ones((3, 3), np.float32)
    neigh = cv2.filter2D(skel.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    endpoints = np.argwhere((skel == 1) & (np.round(neigh) == 2))  # сам пиксель + 1 сосед
    return skel, [tuple(p) for p in endpoints]


def _walk_direction(skel: np.ndarray, start: tuple[int, int], steps: int = 40) -> np.ndarray:
    """Направление линии у концевой точки: от точки в steps шагах назад к концу."""
    h, w = skel.shape
    prev, cur = None, start
    path = [start]
    for _ in range(steps):
        nxt = None
        y, x = cur
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx_ < w and skel[ny, nx_] and (ny, nx_) != prev and (ny, nx_) not in path[-3:]:
                    nxt = (ny, nx_)
                    break
            if nxt:
                break
        if nxt is None:
            break
        prev, cur = cur, nxt
        path.append(cur)
    far = np.array(path[-1], dtype=np.float32)
    end = np.array(start, dtype=np.float32)
    d = end - far
    n = np.linalg.norm(d)
    return d / n if n > 1e-6 else np.array([0.0, 0.0], dtype=np.float32)


def bridge_and_extend(line: np.ndarray, pair_dist: int = 350, border_ext: int = 600,
                      thickness: int = 7) -> np.ndarray:
    """Замыкает разрывы разметки: мостики между ближайшими концами линий,
    затем дотяжка оставшихся концов до рамки вдоль направления штриха."""
    h, w = line.shape
    skel, eps = _skeleton_endpoints(line)
    out = line.copy()
    if not eps:
        return out

    # идентификатор штриха для каждого конца
    _, stroke_labels = cv2.connectedComponents(line, connectivity=8)
    ep_stroke = [int(stroke_labels[y, x]) for (y, x) in eps]

    used = set()
    # 1) мостики: жадно ближайшие пары концов (разных или одного штриха)
    dists = []
    for i in range(len(eps)):
        for j in range(i + 1, len(eps)):
            d = np.hypot(eps[i][0] - eps[j][0], eps[i][1] - eps[j][1])
            if d <= pair_dist:
                dists.append((d, i, j))
    for d, i, j in sorted(dists):
        if i in used or j in used:
            continue
        p1, p2 = eps[i], eps[j]
        cv2.line(out, (p1[1], p1[0]), (p2[1], p2[0]), 1, thickness)
        used.add(i)
        used.add(j)

    # 2) дотяжка свободных концов до рамки вдоль направления штриха
    for i, (y, x) in enumerate(eps):
        if i in used:
            continue
        if min(y, x, h - 1 - y, w - 1 - x) < 10:  # уже на рамке
            used.add(i)
            continue
        dy, dx = _walk_direction(skel, (y, x))
        if abs(dy) < 1e-6 and abs(dx) < 1e-6:
            continue
        end_pt = None
        for t in range(1, border_ext + 1):
            ny, nx_ = int(round(y + dy * t)), int(round(x + dx * t))
            if not (0 <= ny < h and 0 <= nx_ < w):
                end_pt = (int(np.clip(ny, 0, h - 1)), int(np.clip(nx_, 0, w - 1)))
                break
        if end_pt is not None:
            cv2.line(out, (x, y), (end_pt[1], end_pt[0]), 1, thickness)
            used.add(i)

    # 3) штрих, у которого оба конца так и остались свободными, замыкаем его же хордой
    #    (C-образные контуры с большим разрывом) — строго добавляет области, не меняя удачные
    from collections import defaultdict
    free_by_stroke: dict[int, list[int]] = defaultdict(list)
    for i in range(len(eps)):
        if i not in used:
            free_by_stroke[ep_stroke[i]].append(i)
    for stroke_id, idxs in free_by_stroke.items():
        if len(idxs) == 2:
            p1, p2 = eps[idxs[0]], eps[idxs[1]]
            cv2.line(out, (p1[1], p1[0]), (p2[1], p2[0]), 1, thickness)
    return out


def line_to_regions(line: np.ndarray, close_k: int, pair_dist: int = 350,
                    border_ext: int = 600,
                    bg_points: list[list[int]] | None = None) -> tuple[np.ndarray, int, float]:
    """Синяя линия -> заполненная маска зон. Возвращает (маска 0/1, число областей, ambiguity)."""
    line = bridge_and_extend(line, pair_dist=pair_dist, border_ext=border_ext)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    closed = cv2.morphologyEx(line, cv2.MORPH_CLOSE, k)

    barrier = closed.copy()
    barrier[0, :] = 1
    barrier[-1, :] = 1
    barrier[:, 0] = 1
    barrier[:, -1] = 1

    free = (barrier == 0).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(free, connectivity=4)
    if n_labels <= 1:
        return np.zeros_like(line), 0, 0.0

    counts = np.bincount(labels.ravel(), minlength=n_labels)
    counts[0] = 0  # label 0 = барьер
    if bg_points:
        # явное указание фоновых компонент (для инвертированных зон)
        bg_labels = {int(labels[int(y), int(x)]) for x, y in bg_points}
        bg_labels.discard(0)
        talc = (labels != 0) & ~np.isin(labels, sorted(bg_labels))
        ambiguity = 0.0
    else:
        # крупнейшая компонента = фон (не-тальк), остальные = тальк
        order = np.argsort(counts)[::-1]
        bg_label = int(order[0])
        # ambiguity: близость 2-й компоненты к 1-й => выбор фона ненадёжен
        ambiguity = float(counts[order[1]] / counts[bg_label]) if len(order) > 1 and counts[bg_label] > 0 else 0.0
        talc = (labels != 0) & (labels != bg_label)
    talc = talc.astype(np.uint8)

    # линию присоединяем к тальку только там, где она граничит с талевой областью
    grown = cv2.dilate(talc, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k + 2, close_k + 2)))
    talc = talc | (closed & grown)

    n_regions = int(cv2.connectedComponents(talc, connectivity=8)[0]) - 1
    return talc, n_regions, ambiguity


def refine_to_phase(zone: np.ndarray, rgb_orig: np.ndarray, cfg,
                    k_dark: float = 1.0, min_obj: int = 12
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Зона эксперта -> тальковая ФАЗА: тёмная рассеянная составляющая внутри зоны.

    Матрица моделируется робастно (median/MAD) по валидным не-сульфидным пикселям
    вне зон (с защитным отступом); тальк = пиксели зоны темнее матрицы на k_dark*σ.
    Возвращает (тальк bool, ignore bool — тёмная фаза вне зон, сульфиды bool)."""
    pre = preprocess(rgb_orig, cfg)
    sulf, _ = sulfide_mask(pre.gray, pre.valid, cfg)
    gray = pre.gray.astype(np.float32)
    zone_b = zone.astype(bool)

    guard = cv2.dilate(zone.astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))).astype(bool)
    matrix = pre.valid & ~guard & ~sulf
    if matrix.sum() < 0.05 * matrix.size:
        matrix = pre.valid & ~zone_b & ~sulf
    if matrix.sum() < 1000:
        matrix = pre.valid & ~sulf

    vals = gray[matrix]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826
    sigma = max(mad, 4.0)
    dark = (gray < med - k_dark * sigma) & (gray > 12)  # >12: не поры/выколки

    talc = zone_b & pre.valid & ~sulf & dark
    talc = cv2.morphologyEx(talc.astype(np.uint8), cv2.MORPH_CLOSE,
                            np.ones((3, 3), np.uint8)).astype(bool)
    talc = filter_small(talc, min_obj)

    ignore = (~zone_b) & pre.valid & ~sulf & dark
    ignore = filter_small(ignore, min_obj)
    return talc, ignore, sulf


def make_overlay(rgb: np.ndarray, talc: np.ndarray, zone: np.ndarray | None = None) -> np.ndarray:
    """Полупрозрачная заливка тальковой фазы + жёлтый контур зон эксперта."""
    overlay = rgb.copy()
    color = np.array([40, 90, 255], dtype=np.uint8)
    m = talc.astype(bool)
    overlay[m] = (0.45 * overlay[m] + 0.55 * color).astype(np.uint8)
    outline_src = zone if zone is not None else talc
    contours, _ = cv2.findContours(outline_src.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 0), 3)
    return overlay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--close-k", type=int, default=25, help="ядро замыкания разрывов линии")
    parser.add_argument("--pair-dist", type=int, default=350, help="макс. дистанция мостика между концами")
    parser.add_argument("--border-ext", type=int, default=600, help="макс. дотяжка конца до рамки")
    parser.add_argument("--k-dark", type=float, default=1.0, help="порог тёмной фазы, сигм ниже матрицы")
    parser.add_argument("--no-refine", action="store_true", help="сохранить заливку зон без уточнения фазы")
    parser.add_argument("--flag-min", type=float, default=0.5, help="флаг: тальк меньше, %")
    parser.add_argument("--flag-max", type=float, default=45.0, help="флаг: тальк больше, %")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IGNORE_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(ROOT / "config" / "config.yaml"))

    overrides: dict = {}
    if OVERRIDES.exists():
        import json
        overrides = json.loads(OVERRIDES.read_text(encoding="utf-8"))
        print(f"Загружены overrides для {len(overrides)} файлов")

    pairs = sorted({p.resolve() for p in ANNOT_DIR.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")})
    rows = []
    thumbs = []
    for i, annot_path in enumerate(pairs):
        orig_path = ORIG_DIR / annot_path.name
        if not orig_path.exists():
            print(f"[SKIP] нет оригинала для {annot_path.name}")
            continue

        stem = annot_path.stem
        ov = overrides.get(stem, {})
        if ov.get("exclude"):
            for stale in (OUT_DIR / f"{stem}.png", IGNORE_DIR / f"{stem}.png"):
                if stale.exists():
                    stale.unlink()
            print(f"[{i:02d}] {annot_path.name}: EXCLUDED (разметка неоднозначна)")
            continue

        rgb = imread_rgb(annot_path)
        line = extract_blue_line(rgb)

        for x1, y1, x2, y2 in ov.get("extra_lines", []):
            cv2.line(line, (int(x1), int(y1)), (int(x2), int(y2)), 1, 7)
        pair_dist = int(ov.get("pair_dist", args.pair_dist))
        border_ext = int(ov.get("border_ext", args.border_ext))

        zone, n_zones, ambiguity = line_to_regions(line, args.close_k, pair_dist, border_ext,
                                                   bg_points=ov.get("bg_points"))
        zone_frac = 100.0 * zone.sum() / zone.size

        rgb_orig = imread_rgb(orig_path)
        if args.no_refine:
            talc = zone.astype(bool)
            ignore = np.zeros_like(talc)
        else:
            talc, ignore, _ = refine_to_phase(zone, rgb_orig, cfg, k_dark=args.k_dark)
        frac = 100.0 * talc.sum() / talc.size
        n_regions = int(cv2.connectedComponents(talc.astype(np.uint8), connectivity=8)[0]) - 1

        flag = ""
        if frac < args.flag_min:
            flag = "TOO_SMALL"
        elif frac > args.flag_max:
            flag = "TOO_LARGE"
        elif ambiguity > 0.75:
            flag = "AMBIGUOUS_BG"

        imwrite(OUT_DIR / f"{stem}.png", (talc.astype(np.uint8) * 255))
        imwrite(IGNORE_DIR / f"{stem}.png", (ignore.astype(np.uint8) * 255))
        overlay = make_overlay(rgb_orig, talc, zone)
        imwrite(REVIEW_DIR / f"{stem}_overlay.jpg", overlay)

        th = cv2.resize(overlay, (500, 375))
        label = f"{i:02d} {frac:.1f}% (z{zone_frac:.0f}%) {flag}"
        cv2.putText(th, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(th, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        thumbs.append(th)

        rows.append({"idx": i, "file": annot_path.name, "talc_percent": round(frac, 2),
                     "zone_percent": round(zone_frac, 2), "n_regions": n_regions,
                     "ambiguity": round(ambiguity, 2), "flag": flag})
        status = flag if flag else "ok"
        print(f"[{i:02d}] {annot_path.name}: тальк {frac:.1f}% (зона {zone_frac:.1f}%) | "
              f"областей {n_regions} | {status}", flush=True)

    # contact sheets 4 x N
    cols = 4
    for sheet_i in range(0, len(thumbs), 16):
        chunk = thumbs[sheet_i:sheet_i + 16]
        while len(chunk) % cols:
            chunk.append(np.full_like(chunk[0], 255))
        rows_img = [np.hstack(chunk[r:r + cols]) for r in range(0, len(chunk), cols)]
        sheet = np.vstack(rows_img)
        imwrite(REVIEW_DIR / f"contact_sheet_{sheet_i // 16}.jpg", sheet)

    with open(REVIEW_DIR / "report.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "file", "talc_percent", "zone_percent",
                                               "n_regions", "ambiguity", "flag"])
        writer.writeheader()
        writer.writerows(rows)

    flagged = [r for r in rows if r["flag"]]
    print(f"\nИтого: {len(rows)} масок, флагов: {len(flagged)}")
    for r in flagged:
        print(f"  !! [{r['idx']:02d}] {r['file']} — {r['flag']} ({r['talc_percent']}%)")
    print(f"Contact-sheets: {REVIEW_DIR}")


if __name__ == "__main__":
    main()
