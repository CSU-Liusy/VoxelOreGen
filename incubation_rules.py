from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List

from ore_state import OreState, clamp01


RuleApplyFn = Callable[[OreState, random.Random, float, float], None]


@dataclass
class IncubationRule:
    rule_id: int
    name: str
    apply: RuleApplyFn


def _local_mean(state: OreState, values: List[float]) -> List[float]:
    output = values.copy()
    for idx in range(state.voxel_count):
        x, y, z = state.xyz(idx)
        total = values[idx]
        count = 1
        for nidx in state.iter_neighbors6(x, y, z):
            total += values[nidx]
            count += 1
        output[idx] = total / count
    return output


def _edge_strength(state: OreState, values: List[float]) -> List[float]:
    edges = [0.0] * state.voxel_count
    for idx in range(state.voxel_count):
        x, y, z = state.xyz(idx)
        v = values[idx]
        diff = 0.0
        count = 0
        for nidx in state.iter_neighbors6(x, y, z):
            diff += abs(v - values[nidx])
            count += 1
        edges[idx] = diff / max(1, count)
    return edges


def _blend(old: float, target: float, amount: float) -> float:
    return old + (target - old) * amount


def _rule_1_system(state: OreState, rng: random.Random, influence: float, compliance: float) -> None:
    k = 0.42 * influence
    for i in range(state.voxel_count):
        target = (
            0.32 * state.structure[i]
            + 0.2 * state.fluid_flux[i]
            + 0.2 * state.permeability[i]
            + 0.18 * state.temperature[i]
            + 0.1 * state.preservation[i]
        )
        state.potential[i] = clamp01(_blend(state.potential[i], target, k))


def _rule_2_plate_control(state: OreState, rng: random.Random, influence: float, compliance: float) -> None:
    x_size, y_size, z_size = state.grid_size
    major_axis = rng.choice((0, 1))
    for idx in range(state.voxel_count):
        x, y, z = state.xyz(idx)
        axis_ratio = (x / max(1, x_size - 1)) if major_axis == 0 else (y / max(1, y_size - 1))
        edge = abs(axis_ratio - 0.5) * 2.0
        depth = z / max(1, z_size - 1)
        belt = clamp01(1.0 - edge + 0.25 * depth)
        state.potential[idx] = clamp01(state.potential[idx] + 0.15 * influence * belt * state.structure[idx])


def _rule_3_geochemical_enrich(state: OreState, rng: random.Random, influence: float, compliance: float) -> None:
    gain = 0.2 * influence
    for i in range(state.voxel_count):
        p = state.potential[i]
        if p > 0.12:
            p = p + gain * (0.2 + p)
        else:
            p = p * (1.0 - 0.08 * influence)
        state.potential[i] = clamp01(p)


def _rule_4_fluid_source(state: OreState, rng: random.Random, influence: float, compliance: float) -> None:
    for i in range(state.voxel_count):
        carrier = state.fluid_flux[i] * state.permeability[i] * state.temperature[i]
        state.potential[i] = clamp01(state.potential[i] + 0.18 * influence * carrier)


def _rule_5_complex_transport(state: OreState, rng: random.Random, influence: float, compliance: float) -> None:
    x_size, y_size, z_size = state.grid_size
    moved = state.potential.copy()
    for z in range(1, z_size):
        for y in range(y_size):
            for x in range(x_size):
                src = state.index(x, y, z)
                dst = state.index(x, y, z - 1)
                transport = 0.06 * influence * state.fluid_flux[src] * state.structure[src]
                transfer = state.potential[src] * transport
                moved[src] -= transfer * 0.6
                moved[dst] += transfer
    state.potential = [clamp01(v) for v in moved]


def _rule_6_trigger_precipitation(state: OreState, rng: random.Random, influence: float, compliance: float) -> None:
    temp_edge = _edge_strength(state, state.temperature)
    pressure_edge = _edge_strength(state, state.pressure)
    for i in range(state.voxel_count):
        trigger = clamp01(0.5 * temp_edge[i] + 0.5 * pressure_edge[i])
        state.potential[i] = clamp01(state.potential[i] + 0.2 * influence * trigger * state.fluid_flux[i])


def _rule_7_geometry_mechanics(state: OreState, rng: random.Random, influence: float, compliance: float) -> None:
    smoothed = _local_mean(state, state.potential)
    for i in range(state.voxel_count):
        continuity = 0.5 + 0.5 * state.structure[i]
        amount = 0.35 * influence * continuity
        state.potential[i] = clamp01(_blend(state.potential[i], smoothed[i], amount))


def _rule_8_magmatic_specificity(state: OreState, rng: random.Random, influence: float, compliance: float) -> None:
    _, _, z_size = state.grid_size
    for idx in range(state.voxel_count):
        _, _, z = state.xyz(idx)
        depth = z / max(1, z_size - 1)
        magmatic = depth * state.temperature[idx]
        state.potential[idx] = clamp01(state.potential[idx] + 0.14 * influence * magmatic)


def _rule_9_skarn_contact(state: OreState, rng: random.Random, influence: float, compliance: float) -> None:
    reactivity_edge = _edge_strength(state, state.reactivity)
    for i in range(state.voxel_count):
        contact = clamp01(0.55 * state.reactivity[i] + 0.45 * reactivity_edge[i])
        state.potential[i] = clamp01(state.potential[i] + 0.16 * influence * contact * state.fluid_flux[i])


def _rule_10_hydrothermal_network(state: OreState, rng: random.Random, influence: float, compliance: float) -> None:
    halo = _local_mean(state, state.potential)
    for i in range(state.voxel_count):
        if halo[i] > state.potential[i]:
            state.potential[i] = clamp01(state.potential[i] + 0.12 * influence * (halo[i] - state.potential[i]))


def _rule_11_volcanic_sedimentary(state: OreState, rng: random.Random, influence: float, compliance: float) -> None:
    x_size, y_size, z_size = state.grid_size
    horizon_count = rng.randint(1, 3)
    horizons = [rng.randint(z_size // 4, max(z_size // 4, (z_size * 3) // 4)) for _ in range(horizon_count)]
    for z0 in horizons:
        for z in range(max(0, z0 - 1), min(z_size, z0 + 2)):
            layer_factor = 1.0 - abs(z - z0) * 0.4
            for y in range(y_size):
                for x in range(x_size):
                    idx = state.index(x, y, z)
                    state.potential[idx] = clamp01(
                        state.potential[idx] + 0.07 * influence * layer_factor * state.permeability[idx]
                    )


def _rule_12_pegmatite_late_stage(state: OreState, rng: random.Random, influence: float, compliance: float) -> None:
    x_size, y_size, z_size = state.grid_size
    pod_count = rng.randint(1, 4)
    for _ in range(pod_count):
        cx = rng.randint(0, x_size - 1)
        cy = rng.randint(0, y_size - 1)
        cz = rng.randint(0, z_size - 1)
        radius = rng.randint(1, max(1, min(state.grid_size) // 10))
        for z in range(max(0, cz - radius), min(z_size, cz + radius + 1)):
            for y in range(max(0, cy - radius), min(y_size, cy + radius + 1)):
                for x in range(max(0, cx - radius), min(x_size, cx + radius + 1)):
                    idx = state.index(x, y, z)
                    dx = x - cx
                    dy = y - cy
                    dz = z - cz
                    d2 = dx * dx + dy * dy + dz * dz
                    if d2 <= radius * radius:
                        bump = 0.22 * influence * (1.0 - (d2 / max(1, radius * radius + 1)))
                        state.potential[idx] = clamp01(state.potential[idx] + bump)


def _rule_13_microtexture(state: OreState, rng: random.Random, influence: float, compliance: float) -> None:
    for i in range(state.voxel_count):
        perturb = rng.gauss(0.0, 0.015 + 0.02 * influence)
        state.potential[i] = clamp01(state.potential[i] + perturb)


def _rule_14_metamorphic_remobilization(state: OreState, rng: random.Random, influence: float, compliance: float) -> None:
    shifted = state.potential.copy()
    direction = rng.choice(((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)))
    dx, dy, dz = direction
    x_size, y_size, z_size = state.grid_size
    for z in range(z_size):
        for y in range(y_size):
            for x in range(x_size):
                src = state.index(x, y, z)
                nx = x + dx
                ny = y + dy
                nz = z + dz
                if nx < 0 or ny < 0 or nz < 0 or nx >= x_size or ny >= y_size or nz >= z_size:
                    continue
                dst = state.index(nx, ny, nz)
                flow = 0.05 * influence * state.structure[src]
                transfer = state.potential[src] * flow
                shifted[src] -= transfer * 0.5
                shifted[dst] += transfer
    state.potential = [clamp01(v) for v in shifted]


def _rule_15_predictive_specificity(state: OreState, rng: random.Random, influence: float, compliance: float) -> None:
    avg = sum(state.potential) / max(1, state.voxel_count)
    for i in range(state.voxel_count):
        if state.potential[i] > avg:
            state.potential[i] = clamp01(state.potential[i] + 0.08 * influence * state.preservation[i])
        else:
            state.potential[i] = clamp01(state.potential[i] * (1.0 - 0.04 * influence))


def _rule_16_detail_balance(state: OreState, rng: random.Random, influence: float, compliance: float) -> None:
    for i in range(state.voxel_count):
        chloride_proxy = 0.5 * state.fluid_flux[i] + 0.5 * state.temperature[i]
        state.potential[i] = clamp01(state.potential[i] + 0.09 * influence * chloride_proxy)


def _rule_17_high_order_extension(state: OreState, rng: random.Random, influence: float, compliance: float) -> None:
    for _ in range(2):
        smoothed = _local_mean(state, state.potential)
        for i in range(state.voxel_count):
            pulse = rng.uniform(0.0, 0.05) * state.structure[i]
            state.potential[i] = clamp01(
                _blend(state.potential[i], smoothed[i] + pulse, 0.18 * influence)
            )


def _rule_18_deposit_type_id(state: OreState, rng: random.Random, influence: float, compliance: float) -> None:
    _, _, z_size = state.grid_size
    for idx in range(state.voxel_count):
        _, _, z = state.xyz(idx)
        depth = z / max(1, z_size - 1)
        zoning = 0.65 + 0.35 * depth
        state.potential[idx] = clamp01(state.potential[idx] * (1.0 - 0.08 * influence) + 0.08 * influence * zoning)


def _rule_19_micro_dynamics(state: OreState, rng: random.Random, influence: float, compliance: float) -> None:
    for i in range(state.voxel_count):
        if rng.random() < (0.005 + 0.02 * influence * state.potential[i]):
            nugget = rng.uniform(0.06, 0.22)
            state.potential[i] = clamp01(state.potential[i] + nugget)


def _rule_20_integrated_summary(state: OreState, rng: random.Random, influence: float, compliance: float) -> None:
    smooth = _local_mean(state, state.potential)
    for idx in range(state.voxel_count):
        x, y, z = state.xyz(idx)
        nearby_rich = 0
        for nidx in state.iter_neighbors6(x, y, z):
            if state.potential[nidx] > 0.12:
                nearby_rich += 1

        if nearby_rich <= 1:
            state.potential[idx] = clamp01(state.potential[idx] * (1.0 - 0.25 * influence))
        else:
            state.potential[idx] = clamp01(_blend(state.potential[idx], smooth[idx], 0.2 * influence))


def build_rules(names: List[str]) -> List[IncubationRule]:
    apply_functions: List[RuleApplyFn] = [
        _rule_1_system,
        _rule_2_plate_control,
        _rule_3_geochemical_enrich,
        _rule_4_fluid_source,
        _rule_5_complex_transport,
        _rule_6_trigger_precipitation,
        _rule_7_geometry_mechanics,
        _rule_8_magmatic_specificity,
        _rule_9_skarn_contact,
        _rule_10_hydrothermal_network,
        _rule_11_volcanic_sedimentary,
        _rule_12_pegmatite_late_stage,
        _rule_13_microtexture,
        _rule_14_metamorphic_remobilization,
        _rule_15_predictive_specificity,
        _rule_16_detail_balance,
        _rule_17_high_order_extension,
        _rule_18_deposit_type_id,
        _rule_19_micro_dynamics,
        _rule_20_integrated_summary,
    ]

    rules: List[IncubationRule] = []
    for idx, apply_func in enumerate(apply_functions, start=1):
        name = names[idx - 1] if idx - 1 < len(names) else f"Rule {idx}"
        rules.append(IncubationRule(rule_id=idx, name=name, apply=apply_func))
    return rules
