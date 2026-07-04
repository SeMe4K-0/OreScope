"""Тайловый инференс U-Net по большим изображениям со взвешенной сборкой.

Вероятности тайлов смешиваются косинусным окном (вес -> 0 к краю тайла),
бинаризация только после смешивания — швов на стыках нет. Аккумуляторы
float16: для панорамы 300 Мп это ~1.2 ГБ вместо 2.4 ГБ.
"""
from __future__ import annotations

import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _cosine_window(size: int) -> np.ndarray:
    r = np.hanning(size + 2)[1:-1].astype(np.float32)
    w = np.outer(r, r)
    return np.maximum(w, 1e-3)


def _positions(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    pos = list(range(0, length - tile + 1, stride))
    if pos[-1] != length - tile:
        pos.append(length - tile)
    return pos


@torch.no_grad()
def predict_tiled(model: torch.nn.Module, rgb: np.ndarray, device: torch.device,
                  tile: int = 1024, overlap: int = 128, batch: int = 4,
                  tta: bool = False) -> np.ndarray:
    """Возвращает карту вероятности класса 1 (float32 HxW) по RGB uint8."""
    model.eval()
    H, W = rgb.shape[:2]
    stride = tile - overlap
    win = _cosine_window(tile)

    prob_acc = np.zeros((H, W), np.float16)
    w_acc = np.zeros((H, W), np.float16)

    coords = [(y, x) for y in _positions(H, tile, stride) for x in _positions(W, tile, stride)]

    def run_batch(batch_tiles: list[np.ndarray], batch_coords: list[tuple[int, int]]) -> None:
        arr = np.stack(batch_tiles).astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        t = torch.from_numpy(arr.transpose(0, 3, 1, 2)).to(device)
        views = [t]
        if tta:
            views += [torch.rot90(t, k, dims=[-2, -1]) for k in (1, 2, 3)]
        probs = None
        for k, v in enumerate(views):
            logits = model(v)
            p = torch.softmax(logits, dim=1)[:, 1]
            if k:
                p = torch.rot90(p, -k, dims=[-2, -1])
            probs = p if probs is None else probs + p
        probs = (probs / len(views)).float().cpu().numpy()
        for p, (y, x) in zip(probs, batch_coords):
            th, tw = min(tile, H - y), min(tile, W - x)
            prob_acc[y:y + th, x:x + tw] += (p[:th, :tw] * win[:th, :tw]).astype(np.float16)
            w_acc[y:y + th, x:x + tw] += win[:th, :tw].astype(np.float16)

    tiles, cs = [], []
    for (y, x) in coords:
        patch = rgb[y:y + tile, x:x + tile]
        if patch.shape[:2] != (tile, tile):
            padded = np.zeros((tile, tile, 3), np.uint8)
            padded[:patch.shape[0], :patch.shape[1]] = patch
            patch = padded
        tiles.append(patch)
        cs.append((y, x))
        if len(tiles) == batch:
            run_batch(tiles, cs)
            tiles, cs = [], []
    if tiles:
        run_batch(tiles, cs)

    return (prob_acc.astype(np.float32) / np.maximum(w_acc.astype(np.float32), 1e-3))
