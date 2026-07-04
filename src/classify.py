"""Экспертное правило классификации руды + текстовое заключение."""
from __future__ import annotations

from dataclasses import dataclass, field

from omegaconf import DictConfig


@dataclass
class OreVerdict:
    ore_class: str                 # "оталькованная" | "рядовая" | "труднообогатимая" | "не классифицирована"
    talc_percent: float            # тальковая ФАЗА, % валидной площади
    sulfide_percent: float
    ordinary_percent: float
    fine_percent: float
    fine_share: float              # доля тонких среди сульфидов, %
    talc_zone_percent: float = 0.0  # ЗОНЫ оталькования (реконструкция), % — для правила 10%
    warnings: list[str] = field(default_factory=list)
    review_needed: bool = False
    conclusion: str = ""


def classify_ore(talc_px: int, ordinary_px: int, fine_px: int, valid_px: int,
                 cfg: DictConfig, talc_zone_px: int | None = None) -> OreVerdict:
    """Экспертное правило. Порог 10% эксперт вывел для ЗОН оталькования (его
    контуры обводят зоны), поэтому правило применяется к talc_zone_px;
    talc_px — рассеянная тальковая фаза, репортится как количественная метрика."""
    r = cfg.rule
    valid_px = max(valid_px, 1)

    talc_pct = 100.0 * talc_px / valid_px
    zone_pct = 100.0 * (talc_zone_px if talc_zone_px is not None else talc_px) / valid_px
    ord_pct = 100.0 * ordinary_px / valid_px
    fine_pct = 100.0 * fine_px / valid_px
    sulf_px = ordinary_px + fine_px
    sulf_pct = ord_pct + fine_pct
    fine_share = 100.0 * fine_px / sulf_px if sulf_px > 0 else 0.0

    warnings: list[str] = []
    review = False

    if abs(zone_pct - float(r.talc_percent)) < float(r.talc_border_percent):
        warnings.append(
            f"Зоны оталькования ({zone_pct:.1f}%) на границе порога {r.talc_percent:.0f}% — "
            "рекомендуется ручная верификация.")
        review = True

    min_phase = float(getattr(r, "min_talc_phase_percent", 0.0))
    if zone_pct > float(r.talc_percent) and talc_pct > min_phase:
        ore_class = "оталькованная"
    elif sulf_pct < float(r.min_sulfide_percent):
        ore_class = "не классифицирована"
        warnings.append(
            f"Суммарная площадь сульфидов ({sulf_pct:.2f}%) ниже значимого порога "
            f"{r.min_sulfide_percent}% — пустая порода либо недостаточно данных.")
        review = True
    else:
        if abs(fine_pct - ord_pct) < float(r.near_tie_percent) * sulf_pct / 100.0:
            warnings.append(
                "Площади обычных и тонких срастаний практически равны — "
                "рекомендуется ручная верификация.")
            review = True
        ore_class = "труднообогатимая" if fine_px >= ordinary_px else "рядовая"

    verdict = OreVerdict(
        ore_class=ore_class,
        talc_percent=round(talc_pct, 2),
        sulfide_percent=round(sulf_pct, 2),
        ordinary_percent=round(ord_pct, 2),
        fine_percent=round(fine_pct, 2),
        fine_share=round(fine_share, 1),
        talc_zone_percent=round(zone_pct, 2),
        warnings=warnings,
        review_needed=review,
    )
    verdict.conclusion = _build_conclusion(verdict)
    return verdict


def _build_conclusion(v: OreVerdict) -> str:
    if v.ore_class == "оталькованная":
        text = (f"Руда классифицирована как оталькованная: зоны оталькования — "
                f"{v.talc_zone_percent:.1f}% (тальковая фаза — {v.talc_percent:.1f}%), "
                f"преобладание тонких срастаний — {v.fine_share:.0f}%.")
    elif v.ore_class == "рядовая":
        text = (f"Руда классифицирована как рядовая: зоны оталькования — "
                f"{v.talc_zone_percent:.1f}% (≤ 10%), "
                f"преобладают обычные срастания ({100 - v.fine_share:.0f}% площади сульфидов).")
    elif v.ore_class == "труднообогатимая":
        text = (f"Руда классифицирована как труднообогатимая: зоны оталькования — "
                f"{v.talc_zone_percent:.1f}% (≤ 10%), "
                f"преобладают тонкие срастания ({v.fine_share:.0f}% площади сульфидов).")
    else:
        text = ("Руда не классифицирована: содержание полезных и вредных компонентов "
                "ниже значимых порогов (пустая порода).")
    if v.warnings:
        text += " " + " ".join(v.warnings)
    return text
