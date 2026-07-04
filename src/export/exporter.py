from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from src.analysis.sem_analyzer import SEMAnalysisResult, sem_result_to_dict
from src.analysis.xrd_analyzer import XRDAnalysisResult, xrd_result_to_dict
from src.fusion.fusion import FusionResult, fusion_to_dataframe
from src.uncertainty.tta import TTAResult

logger = logging.getLogger(__name__)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def export_csv(
    fusion_result: FusionResult,
    sem_result: SEMAnalysisResult,
    xrd_result: XRDAnalysisResult,
    cfg: DictConfig,
    output_dir: str | Path | None = None,
) -> Path:
    out_dir = Path(output_dir or cfg.export.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"analysis_{_timestamp()}.csv"
    delimiter = cfg.export.csv_delimiter

    df = fusion_to_dataframe(fusion_result)
    df.to_csv(path, sep=delimiter, index=False)
    logger.info("CSV exported -> %s", path)
    return path


def export_json(
    sem_result: SEMAnalysisResult,
    xrd_result: XRDAnalysisResult,
    fusion_result: FusionResult,
    tta_result: TTAResult | None,
    cfg: DictConfig,
    output_dir: str | Path | None = None,
) -> Path:
    out_dir = Path(output_dir or cfg.export.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"analysis_{_timestamp()}.json"

    payload: dict = {
        "timestamp": _timestamp(),
        "sem_analysis": sem_result_to_dict(sem_result),
        "xrd_analysis": xrd_result_to_dict(xrd_result),
        "fusion": {
            "overall_agreement": fusion_result.overall_agreement,
            "review_recommended": fusion_result.review_recommended,
            "n_phases_sem": fusion_result.n_phases_sem,
            "n_phases_xrd": fusion_result.n_phases_xrd,
            "table": [
                {
                    "rank": r.rank,
                    "sem_phase": r.sem_phase,
                    "sem_area_%": r.sem_area_fraction,
                    "xrd_phase": r.xrd_phase,
                    "xrd_volume_%": r.xrd_volume_fraction,
                    "difference_%": r.difference,
                    "comment": r.comment,
                }
                for r in fusion_result.rows
            ],
        },
    }
    if tta_result is not None:
        payload["uncertainty"] = {
            "confidence_score": tta_result.confidence_score,
            "uncertainty_mean": float(tta_result.uncertainty_map.mean()),
            "uncertainty_max": float(tta_result.uncertainty_map.max()),
        }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=int(cfg.export.json_indent), ensure_ascii=False)

    logger.info("JSON exported -> %s", path)
    return path


