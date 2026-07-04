from __future__ import annotations

import logging

import cv2
import numpy as np
from omegaconf import DictConfig
from skimage.morphology import remove_small_objects

logger = logging.getLogger(__name__)


def postprocess_mask(mask: np.ndarray, cfg: DictConfig) -> np.ndarray:
    pp = cfg.postprocessing
    num_classes = cfg.model.num_classes

    result = np.zeros_like(mask)
    for class_idx in range(num_classes):
        binary = (mask == class_idx)
        if class_idx == 0:
            result[binary] = class_idx
            continue

        cleaned = remove_small_objects(binary, min_size=pp.min_object_size)

        open_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (pp.opening_kernel_size, pp.opening_kernel_size),
        )
        close_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (pp.closing_kernel_size, pp.closing_kernel_size),
        )
        opened = cv2.morphologyEx(
            cleaned.astype(np.uint8), cv2.MORPH_OPEN, open_k
        ).astype(bool)
        closed = cv2.morphologyEx(
            opened.astype(np.uint8), cv2.MORPH_CLOSE, close_k
        ).astype(bool)

        if pp.fill_holes:
            closed = _fill_holes(closed)

        result[closed] = class_idx

    _resolve_conflicts(result, mask)
    return result


def _fill_holes(binary: np.ndarray) -> np.ndarray:
    flooded = binary.astype(np.uint8).copy()
    h, w = flooded.shape
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flooded, canvas, (0, 0), 1)
    holes = (flooded == 0) & (~binary)
    return binary | holes


def _resolve_conflicts(result: np.ndarray, original: np.ndarray) -> None:
    conflict = (result == 0) & (original != 0)
    result[conflict] = original[conflict]
