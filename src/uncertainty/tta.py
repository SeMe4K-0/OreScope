from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class TTAResult:
    final_mask: np.ndarray
    uncertainty_map: np.ndarray
    confidence_score: float
    mean_probs: np.ndarray


def _rotate_tensor(x: torch.Tensor, k: int) -> torch.Tensor:
    return torch.rot90(x, k, dims=[-2, -1])


def _rotate_probs_back(probs: torch.Tensor, k: int) -> torch.Tensor:
    return torch.rot90(probs, -k, dims=[-2, -1])


@torch.no_grad()
def tta_predict(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    num_classes: int,
) -> TTAResult:
    model.eval()
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    all_probs: list[torch.Tensor] = []
    rotations = [0, 1, 2, 3]  # 0°, 90°, 180°, 270°

    for k in rotations:
        rotated = _rotate_tensor(image_tensor, k)
        logits = model(rotated)
        probs = F.softmax(logits, dim=1)
        probs_aligned = _rotate_probs_back(probs, k)
        all_probs.append(probs_aligned.cpu())

    stacked = torch.stack(all_probs, dim=0)  # (4, 1, C, H, W)
    mean_probs = stacked.mean(dim=0).squeeze(0)   # (C, H, W)
    std_probs = stacked.std(dim=0).squeeze(0)     # (C, H, W)

    final_mask = mean_probs.argmax(dim=0).numpy().astype(np.uint8)
    uncertainty_map = std_probs.mean(dim=0).numpy().astype(np.float32)

    max_probs = mean_probs.max(dim=0).values.numpy()
    confidence_score = float(max_probs.mean())

    logger.debug(
        "TTA done: confidence=%.3f, uncertainty_mean=%.4f",
        confidence_score,
        uncertainty_map.mean(),
    )

    return TTAResult(
        final_mask=final_mask,
        uncertainty_map=uncertainty_map,
        confidence_score=confidence_score,
        mean_probs=mean_probs.numpy(),
    )
