from __future__ import annotations

import random
import re
from typing import Dict, List

from incubation_rules import IncubationRule, build_rules
from ore_state import OreState


DEFAULT_MAJOR_STRATEGIES = [
    "Macro Metallogenic System",
    "Plate Tectonics And Regional Distribution",
    "Element Geochemistry And Enrichment",
    "Ore-Forming Fluid Source",
    "Complex Transport Of Metals",
    "Physicochemical Trigger And Precipitation",
    "Orebody Geometry And Rock Mechanics",
    "Magmatic Specificity And Differentiation",
    "Skarn Dynamic Evolution",
    "Hydrothermal Network And Alteration",
    "Volcanic Exhalative And Sedimentary Coupling",
    "Pegmatite Late-Stage Enrichment",
    "Ore Texture And Micro-Structure",
    "Metamorphic Remobilization",
    "Metallogenic Specificity And Prediction",
    "Key Supplemental Details",
    "High-Order System Extension",
    "Deposit-Type Identification Rules",
    "Micro Dynamic Kinetic Effects",
    "Integrated Exploration Model",
]


def load_major_strategy_names(rule_file: str) -> List[str]:
    names: List[str] = []
    heading_pattern = re.compile(r"^\s*[一二三四五六七八九十百]+、\s*(.+?)\s*$")

    try:
        with open(rule_file, "r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()
                match = heading_pattern.match(line)
                if match:
                    title = match.group(1)
                    names.append(title)
    except OSError:
        return DEFAULT_MAJOR_STRATEGIES.copy()

    if len(names) < 20:
        names.extend(DEFAULT_MAJOR_STRATEGIES[len(names):])
    return names[:20]


def _compute_effective_weight(
    rank: int,
    total: int,
    rng: random.Random,
) -> tuple[float, float, float]:
    if total <= 1:
        base_importance = 1.0
    else:
        # Earlier rules are more important; later ones are weaker but still impactful.
        base_importance = 1.0 - ((rank - 1) / (total - 1)) * 0.65

    if rank <= 5:
        compliance = 1.0
    elif rank <= 12:
        compliance = rng.uniform(0.7, 1.0)
    else:
        compliance = rng.uniform(0.35, 0.85)

    effective = base_importance * compliance
    return base_importance, compliance, effective


def incubate_seed(
    state: OreState,
    rng: random.Random,
    rule_file: str,
) -> List[Dict[str, float]]:
    names = load_major_strategy_names(rule_file)
    rules: List[IncubationRule] = build_rules(names)

    logs: List[Dict[str, float]] = []
    total = len(rules)
    for rank, rule in enumerate(rules, start=1):
        before = sum(state.potential) / max(1, state.voxel_count)
        base_importance, compliance, effective = _compute_effective_weight(rank, total, rng)
        rule.apply(state, rng, effective, compliance)
        state.clamp_all()
        after = sum(state.potential) / max(1, state.voxel_count)

        logs.append(
            {
                "rank": float(rank),
                "base_importance": base_importance,
                "compliance": compliance,
                "effective_weight": effective,
                "mean_before": before,
                "mean_after": after,
                "delta": after - before,
                "name": rule.name,
            }
        )

    return logs
