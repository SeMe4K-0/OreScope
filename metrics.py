from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _to_onehot(tensor: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(tensor.long(), num_classes).permute(0, 3, 1, 2).float()


class SegmentationMetrics:
    def __init__(self, num_classes: int, ignore_index: int = -1) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self._tp = torch.zeros(self.num_classes)
        self._fp = torch.zeros(self.num_classes)
        self._fn = torch.zeros(self.num_classes)
        self._tn = torch.zeros(self.num_classes)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        if preds.ndim == 4:
            preds = preds.argmax(dim=1)
        preds = preds.cpu()
        targets = targets.cpu()

        if self.ignore_index >= 0:
            mask = targets != self.ignore_index
            preds = preds[mask]
            targets = targets[mask]

        for c in range(self.num_classes):
            pred_c = preds == c
            true_c = targets == c
            self._tp[c] += (pred_c & true_c).sum().float()
            self._fp[c] += (pred_c & ~true_c).sum().float()
            self._fn[c] += (~pred_c & true_c).sum().float()
            self._tn[c] += (~pred_c & ~true_c).sum().float()

    def dice(self) -> torch.Tensor:
        denom = 2 * self._tp + self._fp + self._fn
        return torch.where(denom > 0, 2 * self._tp / denom, torch.zeros_like(self._tp))

    def iou(self) -> torch.Tensor:
        denom = self._tp + self._fp + self._fn
        return torch.where(denom > 0, self._tp / denom, torch.zeros_like(self._tp))

    def precision(self) -> torch.Tensor:
        denom = self._tp + self._fp
        return torch.where(denom > 0, self._tp / denom, torch.zeros_like(self._tp))

    def recall(self) -> torch.Tensor:
        denom = self._tp + self._fn
        return torch.where(denom > 0, self._tp / denom, torch.zeros_like(self._tp))

    def compute(self) -> dict[str, float]:
        dice = self.dice()
        iou = self.iou()
        prec = self.precision()
        rec = self.recall()
        return {
            "dice_mean": dice.mean().item(),
            "iou_mean": iou.mean().item(),
            "precision_mean": prec.mean().item(),
            "recall_mean": rec.mean().item(),
            **{f"dice_class{c}": dice[c].item() for c in range(self.num_classes)},
            **{f"iou_class{c}": iou[c].item() for c in range(self.num_classes)},
        }


class DiceLoss(torch.nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets_oh = _to_onehot(targets, self.num_classes).to(logits.device)
        dims = (0, 2, 3)
        intersection = (probs * targets_oh).sum(dim=dims)
        cardinality = (probs + targets_oh).sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.dice = DiceLoss(num_classes)
        self.ce = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.dice_w = dice_weight
        self.ce_w = ce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice_w * self.dice(logits, targets) + self.ce_w * self.ce(logits, targets)


def compute_class_weights(
    mask_dir: str,
    num_classes: int,
    max_samples: int = 500,
) -> torch.Tensor:
    """Inverse-frequency class weights computed from a sample of masks."""
    import cv2, os, random
    from pathlib import Path
    paths = list(Path(mask_dir).glob("*.png"))
    if len(paths) > max_samples:
        paths = random.sample(paths, max_samples)
    counts = torch.zeros(num_classes)
    for p in paths:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        for c in range(num_classes):
            counts[c] += float((m == c).sum())
    counts = counts.clamp(min=1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes  # normalise so mean=1
    return weights
