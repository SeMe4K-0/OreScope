from __future__ import annotations

import logging
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def build_model(cfg: DictConfig) -> nn.Module:
    model_cfg = cfg.model
    model = smp.Unet(
        encoder_name=model_cfg.encoder,
        encoder_weights=model_cfg.encoder_weights,
        in_channels=model_cfg.in_channels,
        classes=model_cfg.num_classes,
        activation=None,
    )
    logger.info(
        "Built UNet with encoder=%s, classes=%d",
        model_cfg.encoder,
        model_cfg.num_classes,
    )
    return model


def load_model(cfg: DictConfig, checkpoint_path: str | Path, device: torch.device) -> nn.Module:
    model = build_model(cfg)
    state = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    logger.info("Loaded model weights from %s", checkpoint_path)
    return model


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: str | Path,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )
    logger.info("Saved checkpoint -> %s  (epoch=%d)", path, epoch)
