from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

from ore_state import OreState, clamp01


METAL_KEYS = ("cu", "pb", "zn", "au", "ag", "w", "mo")


def _norm_depth_from_bottom(z: int, z_size: int) -> float:
    """Depth ratio where z=0 is deepest and z=max is shallowest."""
    if z_size <= 1:
        return 1.0
    return 1.0 - (z / (z_size - 1))


def _mean(values: List[float]) -> float:
    return sum(values) / max(1, len(values))


def _gaussian_2d(x: float, y: float, cx: float, cy: float, sigma: float) -> float:
    sigma2 = max(1e-6, sigma * sigma)
    d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
    return math.exp(-d2 / (2.0 * sigma2))


def _fade(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _hash_noise3(ix: int, iy: int, iz: int, seed: int) -> float:
    # Integer hash to deterministic pseudo-random value in [-1, 1].
    n = ix * 73856093 ^ iy * 19349663 ^ iz * 83492791 ^ seed * 2654435761
    n = (n << 13) ^ n
    raw = (n * (n * n * 15731 + 789221) + 1376312589) & 0x7FFFFFFF
    return 1.0 - raw / 1073741824.0


def _value_noise3(x: float, y: float, z: float, seed: int) -> float:
    x0 = math.floor(x)
    y0 = math.floor(y)
    z0 = math.floor(z)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    tx = _fade(x - x0)
    ty = _fade(y - y0)
    tz = _fade(z - z0)

    n000 = _hash_noise3(x0, y0, z0, seed)
    n100 = _hash_noise3(x1, y0, z0, seed)
    n010 = _hash_noise3(x0, y1, z0, seed)
    n110 = _hash_noise3(x1, y1, z0, seed)
    n001 = _hash_noise3(x0, y0, z1, seed)
    n101 = _hash_noise3(x1, y0, z1, seed)
    n011 = _hash_noise3(x0, y1, z1, seed)
    n111 = _hash_noise3(x1, y1, z1, seed)

    nx00 = _lerp(n000, n100, tx)
    nx10 = _lerp(n010, n110, tx)
    nx01 = _lerp(n001, n101, tx)
    nx11 = _lerp(n011, n111, tx)
    nxy0 = _lerp(nx00, nx10, ty)
    nxy1 = _lerp(nx01, nx11, ty)
    nxyz = _lerp(nxy0, nxy1, tz)
    return 0.5 + 0.5 * nxyz


def _fractal_noise3(
    x: float,
    y: float,
    z: float,
    seed: int,
    base_freq: float = 0.2,
    octaves: int = 3,
    gain: float = 0.5,
) -> float:
    total = 0.0
    amp = 1.0
    freq = base_freq
    norm = 0.0

    for octave in range(max(1, octaves)):
        total += amp * _value_noise3(x * freq, y * freq, z * freq, seed + octave * 9176)
        norm += amp
        amp *= gain
        freq *= 2.0

    return total / max(1e-6, norm)


def _normalize3(vx: float, vy: float, vz: float) -> Tuple[float, float, float]:
    norm = max(1e-8, math.sqrt(vx * vx + vy * vy + vz * vz))
    return vx / norm, vy / norm, vz / norm


def _sample_nearest(field: List[float], state: OreState, x: float, y: float, z: float) -> float:
    x_size, y_size, z_size = state.grid_size
    xi = int(round(x))
    yi = int(round(y))
    zi = int(round(z))
    xi = max(0, min(x_size - 1, xi))
    yi = max(0, min(y_size - 1, yi))
    zi = max(0, min(z_size - 1, zi))
    return field[state.index(xi, yi, zi)]


def _make_empty_state(grid_size: Tuple[int, int, int]) -> OreState:
    x_size, y_size, z_size = grid_size
    n = x_size * y_size * z_size
    return OreState(
        grid_size=grid_size,
        potential=[0.0] * n,
        temperature=[0.0] * n,
        pressure=[0.0] * n,
        permeability=[0.0] * n,
        structure=[0.0] * n,
        fluid_flux=[0.0] * n,
        reactivity=[0.0] * n,
        preservation=[0.0] * n,
        porosity=[0.0] * n,
        ph=[0.0] * n,
        eh=[0.0] * n,
        metal_channels={key: [0.0] * n for key in METAL_KEYS},
        complex_channels={
            "chloride_complex": [0.0] * n,
            "bisulfide_complex": [0.0] * n,
            "tungsten_complex": [0.0] * n,
        },
        metadata={
            "chloride": [0.0] * n,
            "bisulfide": [0.0] * n,
            "fluid_saturation": [0.0] * n,
            "oxygen_water": [0.0] * n,
            "alteration_index": [0.0] * n,
            "lithology_limestone": [0.0] * n,
            "stockwork_index": [0.0] * n,
            "center_focus": [0.0] * n,
            "boundary_cooling": [0.0] * n,
            "anisotropy": [0.0] * n,
            "stratabound_mask": [0.0] * n,
            "fracture_seal": [0.0] * n,
        },
    )


def stage0_initialize_background(state: OreState, rng: random.Random) -> None:
    """Step 1: create background tensor channels with low Clark-like metal baseline."""
    x_size, y_size, z_size = state.grid_size

    for z in range(z_size):
        depth = _norm_depth_from_bottom(z, z_size)
        for y in range(y_size):
            y_norm = (2.0 * y / max(1, y_size - 1)) - 1.0
            for x in range(x_size):
                x_norm = (2.0 * x / max(1, x_size - 1)) - 1.0
                idx = state.index(x, y, z)

                state.temperature[idx] = clamp01(0.18 + 0.52 * depth + 0.03 * rng.random())
                state.pressure[idx] = clamp01(0.2 + 0.72 * depth + 0.03 * rng.random())
                state.porosity[idx] = clamp01(0.035 + 0.03 * (1.0 - depth) + 0.02 * rng.random())
                state.permeability[idx] = clamp01(0.05 + 0.35 * state.porosity[idx] + 0.02 * rng.random())

                # pH and Eh are normalized to [0,1] internally.
                state.ph[idx] = clamp01((6.4 + rng.uniform(-0.25, 0.25)) / 14.0)
                state.eh[idx] = clamp01(0.42 + 0.2 * (1.0 - depth) + 0.08 * rng.random())

                state.structure[idx] = clamp01(0.12 + 0.06 * rng.random())
                state.fluid_flux[idx] = clamp01(0.04 + 0.05 * rng.random())
                state.reactivity[idx] = clamp01(0.2 + 0.2 * rng.random())
                state.preservation[idx] = clamp01(0.38 + 0.48 * depth)

                # Clark-like low background concentrations.
                state.metal_channels["cu"][idx] = 0.003 + 0.003 * rng.random()
                state.metal_channels["pb"][idx] = 0.0015 + 0.002 * rng.random()
                state.metal_channels["zn"][idx] = 0.002 + 0.0025 * rng.random()
                state.metal_channels["au"][idx] = 0.0002 + 0.0002 * rng.random()
                state.metal_channels["ag"][idx] = 0.0004 + 0.0004 * rng.random()
                state.metal_channels["w"][idx] = 0.0009 + 0.0007 * rng.random()
                state.metal_channels["mo"][idx] = 0.0007 + 0.0006 * rng.random()

                state.metadata["chloride"][idx] = 0.02 + 0.03 * rng.random()
                state.metadata["bisulfide"][idx] = 0.015 + 0.025 * rng.random()
                state.metadata["fluid_saturation"][idx] = 0.03 + 0.03 * rng.random()
                state.metadata["oxygen_water"][idx] = clamp01(0.5 * (1.0 - depth) + 0.1 * rng.random())

                # Build patchy limestone domains for wall-rock neutralization in step 8.
                limestone_seed = (
                    0.5
                    + 0.35 * (1.0 - abs(x_norm))
                    + 0.25 * (1.0 - abs(y_norm))
                    + rng.uniform(-0.45, 0.2)
                )
                state.metadata["lithology_limestone"][idx] = 1.0 if limestone_seed > 0.78 else 0.0


def stage0_structural_weakness(state: OreState, rng: random.Random) -> int:
    """Step 2: generate fault zones as high-permeability transport corridors."""
    x_size, y_size, z_size = state.grid_size
    triggered = 0

    fault_count = rng.randint(2, 4)
    for _ in range(fault_count):
        nx = rng.uniform(-1.0, 1.0)
        ny = rng.uniform(-1.0, 1.0)
        nz = rng.uniform(-0.6, 0.6)
        norm = max(1e-6, (nx * nx + ny * ny + nz * nz) ** 0.5)
        nx /= norm
        ny /= norm
        nz /= norm
        offset = rng.uniform(-0.55, 0.55)
        width = rng.uniform(0.08, 0.18)

        for z in range(z_size):
            zn = (2.0 * z / max(1, z_size - 1)) - 1.0
            for y in range(y_size):
                yn = (2.0 * y / max(1, y_size - 1)) - 1.0
                for x in range(x_size):
                    xn = (2.0 * x / max(1, x_size - 1)) - 1.0
                    idx = state.index(x, y, z)

                    plane_dist = abs(nx * xn + ny * yn + nz * zn + offset)
                    fault_intensity = max(0.0, 1.0 - plane_dist / width)
                    if fault_intensity <= 0.0:
                        continue

                    triggered += 1
                    state.structure[idx] = clamp01(max(state.structure[idx], 0.55 + 0.45 * fault_intensity))
                    state.permeability[idx] = clamp01(state.permeability[idx] + 0.35 * fault_intensity)
                    state.porosity[idx] = clamp01(state.porosity[idx] + 0.18 * fault_intensity)
                    state.fluid_flux[idx] = clamp01(state.fluid_flux[idx] + 0.2 * fault_intensity)

    return triggered


def stage0_apply_centripetal_permeability(state: OreState) -> None:
    """Build a center-high permeability gradient so fluids naturally converge inward."""
    x_size, y_size, z_size = state.grid_size
    cx = (x_size - 1) / 2.0
    cy = (y_size - 1) / 2.0
    sigma = max(1.2, min(x_size, y_size) * 0.22)

    center_focus = state.metadata["center_focus"]
    for z in range(z_size):
        depth = _norm_depth_from_bottom(z, z_size)
        depth_boost = 0.75 + 0.25 * depth
        for y in range(y_size):
            for x in range(x_size):
                idx = state.index(x, y, z)
                g = _gaussian_2d(x, y, cx, cy, sigma)
                focus = clamp01(g * depth_boost)
                center_focus[idx] = focus

                state.permeability[idx] = clamp01(max(state.permeability[idx], 0.08 + 0.78 * focus))
                state.porosity[idx] = clamp01(max(state.porosity[idx], 0.04 + 0.32 * focus))
                state.structure[idx] = clamp01(max(state.structure[idx], 0.12 + 0.55 * focus))
                state.fluid_flux[idx] = clamp01(max(state.fluid_flux[idx], 0.05 + 0.45 * focus))


def stage0_apply_boundary_cooling_barrier(state: OreState, edge_width: int = 3) -> int:
    """Apply cold/low-pressure side boundaries to force earlier unloading away from edges."""
    x_size, y_size, z_size = state.grid_size
    boundary_cooling = state.metadata["boundary_cooling"]
    touched = 0

    for z in range(z_size):
        for y in range(y_size):
            for x in range(x_size):
                dx = min(x, x_size - 1 - x)
                dy = min(y, y_size - 1 - y)
                d = min(dx, dy)
                if d >= edge_width:
                    continue

                # 1.0 at edge, tapering to 0.0 inside the interior.
                edge_intensity = clamp01((edge_width - d) / max(1e-6, edge_width))
                idx = state.index(x, y, z)
                touched += 1
                boundary_cooling[idx] = max(boundary_cooling[idx], edge_intensity)

                # Dirichlet-like cooling and depressurization near side boundaries.
                state.temperature[idx] = min(state.temperature[idx], 0.2 + 0.12 * (1.0 - edge_intensity))
                state.pressure[idx] = min(state.pressure[idx], 0.24 + 0.16 * (1.0 - edge_intensity))
                state.permeability[idx] = clamp01(state.permeability[idx] * (1.0 - 0.6 * edge_intensity))
                state.fluid_flux[idx] = clamp01(state.fluid_flux[idx] * (1.0 - 0.5 * edge_intensity))
                state.metadata["fluid_saturation"][idx] = clamp01(
                    state.metadata["fluid_saturation"][idx] * (1.0 - 0.45 * edge_intensity)
                )

    return touched


def stage0_apply_stockwork_anisotropy(state: OreState, rng: random.Random) -> Dict[str, float]:
    """Build a branch-prone anisotropic permeability field for sharp stockwork veins."""
    x_size, y_size, z_size = state.grid_size
    anisotropy = state.metadata["anisotropy"]

    # Prefer an oblique-upward direction (~45 degrees up) with slight azimuth randomness.
    azimuth = math.radians(35.0 + rng.uniform(-12.0, 12.0))
    dip = math.radians(42.0 + rng.uniform(-10.0, 10.0))
    dx = math.cos(dip) * math.cos(azimuth)
    dy = math.cos(dip) * math.sin(azimuth)
    dz = math.sin(dip)
    dx, dy, dz = _normalize3(dx, dy, dz)

    noise_seed = rng.randrange(1, 2**31)
    hits = 0
    cx = (x_size - 1) / 2.0
    cy = (y_size - 1) / 2.0
    cz = (z_size - 1) / 2.0

    for z in range(z_size):
        zn = (z - cz) / max(1.0, cz)
        for y in range(y_size):
            yn = (y - cy) / max(1.0, cy)
            for x in range(x_size):
                xn = (x - cx) / max(1.0, cx)
                idx = state.index(x, y, z)

                proj = xn * dx + yn * dy + zn * dz
                noise = _fractal_noise3(x, y, z, noise_seed, base_freq=0.18, octaves=3, gain=0.58)
                ridge = 0.5 + 0.5 * math.sin(16.0 * proj + 5.0 * noise)

                # Blend directional ridge and noise to create textured fracture corridors.
                a = clamp01(0.62 * ridge + 0.38 * noise)
                anisotropy[idx] = a

                if a > 0.58:
                    hits += 1
                    state.permeability[idx] = clamp01(max(state.permeability[idx], 0.12 + 0.86 * a))
                    state.porosity[idx] = clamp01(max(state.porosity[idx], 0.04 + 0.46 * a))
                    state.structure[idx] = clamp01(max(state.structure[idx], 0.2 + 0.72 * a))
                    state.fluid_flux[idx] = clamp01(max(state.fluid_flux[idx], 0.08 + 0.62 * a))

    state.metadata["principal_direction"] = (dx, dy, dz)
    return {
        "anisotropy_hits": float(hits),
        "principal_dx": dx,
        "principal_dy": dy,
        "principal_dz": dz,
    }


def stage0_initialize_stratabound_channels(state: OreState, rng: random.Random) -> Dict[str, float]:
    """Initialize thick, wavy stratabound high-permeability channels for lens-like ore bodies."""
    x_size, y_size, z_size = state.grid_size
    mask = state.metadata["stratabound_mask"]

    noise_seed = rng.randrange(1, 2**31)
    layers = rng.randint(1, 2)
    base_levels = [int((0.36 + i * 0.22 + rng.uniform(-0.05, 0.05)) * (z_size - 1)) for i in range(layers)]
    phase_offsets = [rng.uniform(-0.2, 0.2) for _ in range(layers)]
    wave_amp = max(1.0, z_size * rng.uniform(0.08, 0.16))
    wave_freq_x = rng.uniform(1.3, 2.3)
    wave_freq_y = rng.uniform(0.9, 1.8)
    thickness = max(1.2, z_size * rng.uniform(0.08, 0.14))

    hits = 0
    for z in range(z_size):
        for y in range(y_size):
            y_phase = (2.0 * math.pi * y) / max(1.0, y_size - 1)
            for x in range(x_size):
                x_phase = (2.0 * math.pi * x) / max(1.0, x_size - 1)
                idx = state.index(x, y, z)

                low_freq_noise = _fractal_noise3(x, y, z, noise_seed, base_freq=0.08, octaves=2, gain=0.6)
                layer_score = 0.0
                for layer_idx, base_z in enumerate(base_levels):
                    wave_z = (
                        base_z
                        + wave_amp * math.sin(wave_freq_x * x_phase + phase_offsets[layer_idx])
                        + 0.55 * wave_amp * math.sin(wave_freq_y * y_phase + 0.8)
                        + (low_freq_noise - 0.5) * 2.2
                    )
                    local_thickness = thickness * (0.82 + 0.44 * low_freq_noise)
                    dist = abs(z - wave_z)
                    score = clamp01(1.0 - dist / max(1e-6, local_thickness))
                    layer_score = max(layer_score, score)

                if layer_score <= 0.0:
                    continue

                mask[idx] = layer_score
                if layer_score > 0.28:
                    hits += 1
                    state.permeability[idx] = clamp01(max(state.permeability[idx], 0.15 + 0.72 * layer_score))
                    state.porosity[idx] = clamp01(max(state.porosity[idx], 0.05 + 0.45 * layer_score))
                    state.fluid_flux[idx] = clamp01(max(state.fluid_flux[idx], 0.08 + 0.56 * layer_score))
                    state.structure[idx] = clamp01(max(state.structure[idx], 0.16 + 0.48 * layer_score))

    state.metadata["stratabound_wave_amplitude"] = wave_amp
    state.metadata["stratabound_thickness"] = thickness
    return {
        "stratabound_hits": float(hits),
        "wave_amplitude": wave_amp,
        "layer_thickness": thickness,
    }


def stage1_activate_source(state: OreState, rng: random.Random) -> Dict[str, object]:
    """Step 3: inject deep heat and supercritical fluid source near bottom z=0."""
    x_size, y_size, _ = state.grid_size
    source_center = (x_size // 2, y_size // 2)
    sigma = max(1.2, min(x_size, y_size) * 0.16)

    changed = 0
    z = 0
    cx, cy = source_center
    for y in range(y_size):
        for x in range(x_size):
            idx = state.index(x, y, z)
            radial = _gaussian_2d(x, y, cx, cy, sigma)
            if radial <= 0.0:
                continue

            changed += 1
            state.temperature[idx] = clamp01(max(state.temperature[idx], 0.8 + 0.15 * radial))
            state.pressure[idx] = clamp01(max(state.pressure[idx], 0.86 + 0.12 * radial))
            state.metadata["fluid_saturation"][idx] = clamp01(0.78 + 0.2 * radial)
            state.metadata["chloride"][idx] = clamp01(0.62 + 0.3 * radial)
            state.metadata["bisulfide"][idx] = clamp01(0.48 + 0.35 * radial)
            state.fluid_flux[idx] = clamp01(max(state.fluid_flux[idx], 0.72 + 0.2 * radial))

            state.metal_channels["cu"][idx] = clamp01(state.metal_channels["cu"][idx] + 0.22 * radial)
            state.metal_channels["pb"][idx] = clamp01(state.metal_channels["pb"][idx] + 0.12 * radial)
            state.metal_channels["zn"][idx] = clamp01(state.metal_channels["zn"][idx] + 0.16 * radial)
            state.metal_channels["au"][idx] = clamp01(state.metal_channels["au"][idx] + 0.08 * radial)
            state.metal_channels["ag"][idx] = clamp01(state.metal_channels["ag"][idx] + 0.09 * radial)
            state.metal_channels["w"][idx] = clamp01(state.metal_channels["w"][idx] + 0.1 * radial)
            state.metal_channels["mo"][idx] = clamp01(state.metal_channels["mo"][idx] + 0.09 * radial)

    state.metadata["source_center_xy"] = source_center
    state.metadata["source_radius"] = sigma
    return {"source_center_xy": source_center, "source_radius": sigma, "source_cells": changed}


def stage1_complexation(state: OreState) -> None:
    """Step 4: metals move as complexes, not as free ions."""
    chloride = state.metadata["chloride"]
    bisulfide = state.metadata["bisulfide"]

    chloride_complex = state.complex_channels["chloride_complex"]
    bisulfide_complex = state.complex_channels["bisulfide_complex"]
    tungsten_complex = state.complex_channels["tungsten_complex"]

    for idx in range(state.voxel_count):
        temp_factor = 0.45 + 0.55 * state.temperature[idx]
        salinity_factor = chloride[idx]
        sulfur_factor = bisulfide[idx]

        base_metals = (
            state.metal_channels["cu"][idx] + state.metal_channels["pb"][idx] + state.metal_channels["zn"][idx]
        ) / 3.0
        precious_metals = (
            state.metal_channels["au"][idx] + state.metal_channels["ag"][idx]
        ) / 2.0

        chloride_complex[idx] = clamp01(base_metals * salinity_factor * temp_factor * 1.8)
        bisulfide_complex[idx] = clamp01(precious_metals * sulfur_factor * temp_factor * 2.2)
        tungsten_complex[idx] = clamp01(state.metal_channels["w"][idx] * (0.3 + 0.7 * salinity_factor) * temp_factor)


def stage2_darcy_advection(state: OreState, cycles: int = 8) -> None:
    """Step 5: advection along pressure gradients and permeability pathways."""
    x_size, y_size, z_size = state.grid_size

    fluid = state.metadata["fluid_saturation"]
    center_focus = state.metadata["center_focus"]
    boundary_cooling = state.metadata["boundary_cooling"]
    chloride_complex = state.complex_channels["chloride_complex"]
    bisulfide_complex = state.complex_channels["bisulfide_complex"]
    tungsten_complex = state.complex_channels["tungsten_complex"]

    for _ in range(cycles):
        next_fluid = fluid.copy()
        next_chl = chloride_complex.copy()
        next_bis = bisulfide_complex.copy()
        next_w = tungsten_complex.copy()
        next_temp = state.temperature.copy()
        next_pressure = state.pressure.copy()

        for z in range(z_size):
            for y in range(y_size):
                for x in range(x_size):
                    idx = state.index(x, y, z)
                    if fluid[idx] <= 1e-6:
                        continue

                    candidate_neighbors: List[Tuple[int, float]] = []
                    for nidx in state.iter_neighbors6(x, y, z):
                        nx, ny, nz = state.xyz(nidx)
                        pressure_grad = state.pressure[idx] - state.pressure[nidx]
                        perm_drive = 0.35 * max(0.0, state.permeability[nidx] - state.permeability[idx])
                        upward_bonus = 0.07 if nz > z else 0.0

                        inward_bonus = 0.09 * max(0.0, center_focus[nidx] - center_focus[idx])
                        boundary_penalty = 0.08 * max(0.0, boundary_cooling[nidx] - boundary_cooling[idx])
                        drive = pressure_grad + perm_drive + upward_bonus + inward_bonus - boundary_penalty
                        if drive > 0.0:
                            candidate_neighbors.append((nidx, drive))

                    if not candidate_neighbors:
                        continue

                    total_drive = sum(d for _, d in candidate_neighbors)
                    mobile_fraction = min(0.35, 0.08 + 0.42 * state.permeability[idx])

                    for nidx, drive in candidate_neighbors:
                        share = drive / max(1e-6, total_drive)
                        transfer = fluid[idx] * mobile_fraction * share

                        next_fluid[idx] = max(0.0, next_fluid[idx] - transfer)
                        next_fluid[nidx] = clamp01(next_fluid[nidx] + transfer)

                        for old_array, new_array in (
                            (chloride_complex, next_chl),
                            (bisulfide_complex, next_bis),
                            (tungsten_complex, next_w),
                        ):
                            moved = old_array[idx] * mobile_fraction * share
                            new_array[idx] = max(0.0, new_array[idx] - moved)
                            new_array[nidx] = clamp01(new_array[nidx] + moved)

                        temp_mix = 0.14 * transfer
                        next_temp[nidx] = clamp01(next_temp[nidx] + temp_mix * (state.temperature[idx] - state.temperature[nidx]))
                        next_pressure[idx] = clamp01(next_pressure[idx] - 0.06 * transfer)
                        next_pressure[nidx] = clamp01(next_pressure[nidx] + 0.04 * transfer)

        fluid[:] = [clamp01(v) for v in next_fluid]
        chloride_complex[:] = [clamp01(v) for v in next_chl]
        bisulfide_complex[:] = [clamp01(v) for v in next_bis]
        tungsten_complex[:] = [clamp01(v) for v in next_w]
        state.temperature[:] = [clamp01(v) for v in next_temp]
        state.pressure[:] = [clamp01(v) for v in next_pressure]


def stage2_hydraulic_fracturing(state: OreState) -> int:
    """Step 6: trigger stockwork-like permeability surge when pressure exceeds threshold."""
    fluid = state.metadata["fluid_saturation"]
    stockwork = state.metadata["stockwork_index"]

    triggered = 0
    updates: Dict[int, float] = {}

    for idx in range(state.voxel_count):
        x, y, z = state.xyz(idx)
        depth = _norm_depth_from_bottom(z, state.grid_size[2])

        lithostatic = 0.28 + 0.66 * depth
        tensile_strength = 0.26 - 0.16 * state.structure[idx]
        fracture_pressure = lithostatic + tensile_strength

        if state.pressure[idx] + 0.72 * fluid[idx] <= fracture_pressure:
            continue

        triggered += 1
        updates[idx] = max(updates.get(idx, 0.0), 1.0)
        for nidx in state.iter_neighbors6(x, y, z):
            updates[nidx] = max(updates.get(nidx, 0.0), 0.55)

    for idx, intensity in updates.items():
        state.porosity[idx] = clamp01(state.porosity[idx] + 0.24 * intensity)
        state.permeability[idx] = clamp01(state.permeability[idx] + 0.3 * intensity)
        state.structure[idx] = clamp01(state.structure[idx] + 0.22 * intensity)
        state.fluid_flux[idx] = clamp01(state.fluid_flux[idx] + 0.2 * intensity)
        stockwork[idx] = clamp01(stockwork[idx] + 0.7 * intensity)

    return triggered


def _pick_directional_neighbors(state: OreState, idx: int, direction: Tuple[float, float, float]) -> List[int]:
    x, y, z = state.xyz(idx)
    dx, dy, dz = direction
    scored: List[Tuple[float, int]] = []
    for nidx in state.iter_neighbors6(x, y, z):
        nx, ny, nz = state.xyz(nidx)
        vx = nx - x
        vy = ny - y
        vz = nz - z
        score = vx * dx + vy * dy + vz * dz
        scored.append((score, nidx))

    scored.sort(key=lambda item: item[0], reverse=True)
    best = [n for _, n in scored[:2]]
    if len(scored) >= 2:
        best.append(scored[-1][1])
    return best


def stage2_hydraulic_fracturing_stockwork(state: OreState) -> int:
    """Pressure-threshold fracturing with directional branching for sharp stockwork geometry."""
    fluid = state.metadata["fluid_saturation"]
    anisotropy = state.metadata["anisotropy"]
    stockwork = state.metadata["stockwork_index"]
    fracture_seal = state.metadata["fracture_seal"]

    principal = state.metadata.get("principal_direction", (0.62, 0.2, 0.76))
    direction = _normalize3(float(principal[0]), float(principal[1]), float(principal[2]))

    triggered = 0
    updates: Dict[int, float] = {}

    for idx in range(state.voxel_count):
        _, _, z = state.xyz(idx)
        depth = _norm_depth_from_bottom(z, state.grid_size[2])

        overburden = 0.22 + 0.7 * depth
        tensile_strength = 0.22 - 0.16 * state.structure[idx] + 0.12 * (1.0 - anisotropy[idx])
        fluid_pressure = state.pressure[idx] + 0.88 * fluid[idx]

        if fluid_pressure <= overburden + tensile_strength:
            continue

        triggered += 1
        updates[idx] = max(updates.get(idx, 0.0), 1.0)
        for nidx in _pick_directional_neighbors(state, idx, direction):
            updates[nidx] = max(updates.get(nidx, 0.0), 0.82)

    for idx, intensity in updates.items():
        state.porosity[idx] = clamp01(state.porosity[idx] + 0.28 * intensity)
        # Equivalent to "instant opening" in normalized space.
        state.permeability[idx] = max(state.permeability[idx], 0.96 if intensity > 0.9 else 0.88)
        state.structure[idx] = clamp01(state.structure[idx] + 0.26 * intensity)
        state.fluid_flux[idx] = clamp01(state.fluid_flux[idx] + 0.22 * intensity)
        stockwork[idx] = clamp01(stockwork[idx] + 0.85 * intensity)
        fracture_seal[idx] = clamp01(fracture_seal[idx] * (1.0 - 0.3 * intensity))

        # Rapid pressure unloading after fracture opening.
        state.pressure[idx] = clamp01(state.pressure[idx] - 0.22 * intensity)

    return triggered


def stage2_stratabound_flow_focus(state: OreState, cycles: int = 5) -> int:
    """Confine and elongate flow inside wave-like stratabound channels."""
    mask = state.metadata["stratabound_mask"]
    fluid = state.metadata["fluid_saturation"]
    x_size, y_size, z_size = state.grid_size

    moved_events = 0
    for _ in range(max(1, cycles)):
        next_fluid = fluid.copy()
        for z in range(z_size):
            for y in range(y_size):
                for x in range(x_size):
                    idx = state.index(x, y, z)
                    if mask[idx] < 0.2 or fluid[idx] <= 1e-6:
                        continue

                    candidates: List[Tuple[int, float]] = []
                    for nidx in state.iter_neighbors6(x, y, z):
                        if mask[nidx] < 0.2:
                            continue
                        drive = 0.5 * (mask[nidx] - mask[idx]) + 0.35 * (state.permeability[nidx] - state.permeability[idx])
                        if drive > 0.0:
                            candidates.append((nidx, drive))

                    if not candidates:
                        continue

                    total = sum(v for _, v in candidates)
                    mobile = min(0.24, 0.06 + 0.28 * mask[idx])
                    for nidx, drive in candidates:
                        share = drive / max(1e-6, total)
                        transfer = fluid[idx] * mobile * share
                        if transfer <= 1e-7:
                            continue
                        moved_events += 1

                        next_fluid[idx] = max(0.0, next_fluid[idx] - transfer)
                        next_fluid[nidx] = clamp01(next_fluid[nidx] + transfer)

        fluid[:] = [clamp01(v) for v in next_fluid]

    return moved_events


def stage3_stockwork_pulse_precipitation(state: OreState) -> int:
    """Pressure-drop boiling with fast precipitation and channel sealing (pinch-out/restart behavior)."""
    fluid = state.metadata["fluid_saturation"]
    stockwork = state.metadata["stockwork_index"]
    anisotropy = state.metadata["anisotropy"]
    fracture_seal = state.metadata["fracture_seal"]

    bisulfide_complex = state.complex_channels["bisulfide_complex"]
    chloride_complex = state.complex_channels["chloride_complex"]

    triggered = 0
    for idx in range(state.voxel_count):
        if stockwork[idx] < 0.28 or fluid[idx] < 0.05:
            continue

        pressure_drop = clamp01(0.68 - state.pressure[idx])
        if pressure_drop < 0.08:
            continue

        triggered += 1
        solubility_collapse = clamp01(0.55 * pressure_drop + 0.25 * anisotropy[idx] + 0.2 * state.temperature[idx])

        precious_precip = bisulfide_complex[idx] * (0.7 * solubility_collapse)
        base_precip = chloride_complex[idx] * (0.32 * solubility_collapse)

        bisulfide_complex[idx] = max(0.0, bisulfide_complex[idx] - precious_precip)
        chloride_complex[idx] = max(0.0, chloride_complex[idx] - base_precip)
        state.potential[idx] = clamp01(state.potential[idx] + 1.05 * precious_precip + 0.42 * base_precip)

        # Sealing blocks old channel and pushes later pulses into new weak paths.
        seal = clamp01(fracture_seal[idx] + 0.72 * solubility_collapse)
        fracture_seal[idx] = seal
        state.permeability[idx] = clamp01(state.permeability[idx] * (1.0 - 0.72 * seal))
        state.fluid_flux[idx] = clamp01(state.fluid_flux[idx] * (1.0 - 0.56 * seal))

    return triggered


def stage3_metasomatic_dilation(state: OreState, rng: random.Random, rounds: int = 2) -> int:
    """Reactively dilate stratabound channels to form thick, irregular lens-like boundaries."""
    mask = state.metadata["stratabound_mask"]
    fluid = state.metadata["fluid_saturation"]
    limestone = state.metadata["lithology_limestone"]
    x_size, y_size, z_size = state.grid_size

    expanded_total = 0
    noise_seed = rng.randrange(1, 2**31)

    for step in range(max(1, rounds)):
        to_expand: Dict[int, float] = {}

        for idx in range(state.voxel_count):
            if mask[idx] < 0.25 or fluid[idx] < 0.05:
                continue

            x, y, z = state.xyz(idx)
            dwell = clamp01(fluid[idx] * (0.45 + 0.55 * state.permeability[idx]))
            for nidx in state.iter_neighbors6(x, y, z):
                if mask[nidx] > 0.75:
                    continue

                nx, ny, nz = state.xyz(nidx)
                edge_noise = _fractal_noise3(
                    nx,
                    ny,
                    nz,
                    noise_seed + step * 113,
                    base_freq=0.22,
                    octaves=2,
                    gain=0.6,
                )
                chemical_boost = 0.25 + 0.45 * limestone[nidx] + 0.3 * state.reactivity[nidx]
                growth = clamp01(dwell * chemical_boost * (0.7 + 0.6 * edge_noise))
                if growth < 0.08:
                    continue

                to_expand[nidx] = max(to_expand.get(nidx, 0.0), growth)

        for idx, growth in to_expand.items():
            expanded_total += 1
            mask[idx] = clamp01(max(mask[idx], 0.35 + 0.65 * growth))
            state.permeability[idx] = clamp01(state.permeability[idx] + 0.3 * growth)
            state.porosity[idx] = clamp01(state.porosity[idx] + 0.22 * growth)
            state.structure[idx] = clamp01(state.structure[idx] + 0.18 * growth)
            state.potential[idx] = clamp01(state.potential[idx] + 0.08 * growth)

    return expanded_total


def stage5_ductile_shear_fold_deformation(state: OreState, rng: random.Random) -> Dict[str, float]:
    """Apply non-linear post-ore displacement to mimic fold/shear boudinage deformation."""
    x_size, y_size, z_size = state.grid_size
    cx = (x_size - 1) / 2.0
    cy = (y_size - 1) / 2.0
    cz = (z_size - 1) / 2.0

    phase = rng.uniform(0.0, 2.0 * math.pi)
    compress = rng.uniform(0.2, 0.38)
    shear = rng.uniform(0.18, 0.32)
    fold_amp = rng.uniform(0.8, 1.8)

    old_potential = state.potential[:]
    old_structure = state.structure[:]
    old_permeability = state.permeability[:]

    new_potential = [0.0] * state.voxel_count
    new_structure = [0.0] * state.voxel_count
    new_permeability = [0.0] * state.voxel_count

    for z in range(z_size):
        zn = (z - cz) / max(1.0, cz)
        for y in range(y_size):
            yn = (y - cy) / max(1.0, cy)
            for x in range(x_size):
                xn = (x - cx) / max(1.0, cx)
                idx = state.index(x, y, z)

                # X-side compression + Y shear + Z folding.
                sx = x - compress * xn * abs(xn) * (0.55 * x_size) + shear * yn * (0.18 * x_size)
                sy = y - shear * xn * (0.34 * y_size)
                sz = z - fold_amp * math.sin(2.0 * math.pi * xn + phase) * (0.14 * z_size)
                sz += 0.5 * math.sin(2.0 * math.pi * yn + 0.6 * phase) * (0.08 * z_size) * (1.0 - abs(zn))

                new_potential[idx] = _sample_nearest(old_potential, state, sx, sy, sz)
                new_structure[idx] = _sample_nearest(old_structure, state, sx, sy, sz)
                new_permeability[idx] = _sample_nearest(old_permeability, state, sx, sy, sz)

    state.potential = [clamp01(v) for v in new_potential]
    state.structure = [clamp01(v) for v in new_structure]
    state.permeability = [clamp01(v) for v in new_permeability]

    return {
        "compress_strength": compress,
        "shear_strength": shear,
        "fold_amplitude": fold_amp,
    }


def stage3_boiling_precipitation(state: OreState) -> int:
    """Step 7: shallow decompression boiling and precious-metal precipitation."""
    z_size = state.grid_size[2]
    shallow_cut = int(0.62 * (z_size - 1))

    fluid = state.metadata["fluid_saturation"]
    chloride = state.metadata["chloride"]
    bisulfide = state.metadata["bisulfide"]
    bisulfide_complex = state.complex_channels["bisulfide_complex"]
    chloride_complex = state.complex_channels["chloride_complex"]

    triggered = 0
    for idx in range(state.voxel_count):
        x, y, z = state.xyz(idx)
        if z < shallow_cut:
            continue
        if (
            state.temperature[idx] < 0.3
            or fluid[idx] < 0.045
            or (state.pressure[idx] > 0.72 and state.structure[idx] < 0.55)
        ):
            continue

        triggered += 1
        volatile_escape = clamp01((0.62 - state.pressure[idx]) * 1.3 + 0.18)

        state.temperature[idx] = clamp01(state.temperature[idx] - 0.16 * volatile_escape)
        chloride[idx] = clamp01(chloride[idx] * (1.0 - 0.35 * volatile_escape))
        bisulfide[idx] = clamp01(bisulfide[idx] * (1.0 - 0.5 * volatile_escape))
        state.ph[idx] = clamp01(state.ph[idx] + 0.1 * volatile_escape)

        precious_precip = bisulfide_complex[idx] * (0.55 * volatile_escape)
        base_precip = chloride_complex[idx] * (0.2 * volatile_escape)

        bisulfide_complex[idx] = max(0.0, bisulfide_complex[idx] - precious_precip)
        chloride_complex[idx] = max(0.0, chloride_complex[idx] - base_precip)

        state.potential[idx] = clamp01(state.potential[idx] + 0.95 * precious_precip + 0.35 * base_precip)

    return triggered


def stage3_mixing_and_neutralization(state: OreState) -> int:
    """Step 8: meteoric-water mixing and acid-neutralization wall-rock reaction."""
    x_size, y_size, z_size = state.grid_size

    oxygen = state.metadata["oxygen_water"]
    fluid = state.metadata["fluid_saturation"]
    limestone = state.metadata["lithology_limestone"]

    chloride_complex = state.complex_channels["chloride_complex"]
    tungsten_complex = state.complex_channels["tungsten_complex"]

    # Simple top-down meteoric recharge.
    for z in range(max(0, z_size - 3), z_size):
        for y in range(y_size):
            for x in range(x_size):
                idx = state.index(x, y, z)
                oxygen[idx] = clamp01(oxygen[idx] + 0.12)
                fluid[idx] = clamp01(fluid[idx] + 0.06)

    triggered = 0
    for idx in range(state.voxel_count):
        mix = fluid[idx] * oxygen[idx]
        if mix < 0.06:
            continue

        triggered += 1
        state.temperature[idx] = clamp01(state.temperature[idx] - 0.12 * mix)
        state.eh[idx] = clamp01(state.eh[idx] + 0.1 * mix)
        state.ph[idx] = clamp01(state.ph[idx] + 0.03 * mix)

        precip = chloride_complex[idx] * (0.32 * mix)
        chloride_complex[idx] = max(0.0, chloride_complex[idx] - precip)
        state.potential[idx] = clamp01(state.potential[idx] + 0.52 * precip)

        # Skarn-like neutralization around limestone under acidic fluid.
        if limestone[idx] > 0.5 and state.ph[idx] < (6.0 / 14.0):
            neutralization = clamp01((6.0 / 14.0 - state.ph[idx]) * 2.5)
            tungsten_precip = tungsten_complex[idx] * (0.45 * neutralization)
            tungsten_complex[idx] = max(0.0, tungsten_complex[idx] - tungsten_precip)
            state.ph[idx] = clamp01(state.ph[idx] + 0.12 * neutralization)
            state.potential[idx] = clamp01(state.potential[idx] + 0.7 * tungsten_precip)

    return triggered


def stage4_zonation(state: OreState) -> None:
    """Step 9: central high-T precipitation, peripheral low-T Pb/Zn/Ag zoning."""
    source_center = state.metadata.get("source_center_xy", (state.grid_size[0] // 2, state.grid_size[1] // 2))
    radius = max(1, int(state.metadata.get("source_radius", max(2, min(state.grid_size) // 6))))
    x_size, y_size, z_size = state.grid_size
    cx, cy = source_center

    for z in range(z_size):
        depth = _norm_depth_from_bottom(z, z_size)
        for y in range(y_size):
            for x in range(x_size):
                idx = state.index(x, y, z)
                dist = ((x - cx) ** 2 + (y - cy) ** 2 + (z * 0.65) ** 2) ** 0.5
                near_center = clamp01(1.0 - dist / max(1.0, radius * 2.8))
                edge_zone = clamp01(1.0 - near_center)

                high_t = state.temperature[idx]
                low_t = clamp01(1.0 - high_t)

                core_enrich = near_center * high_t * (0.5 + 0.5 * depth)
                edge_enrich = edge_zone * low_t * (0.5 + 0.5 * (1.0 - depth))

                state.potential[idx] = clamp01(
                    state.potential[idx]
                    + 0.14 * core_enrich * (state.metal_channels["cu"][idx] + state.metal_channels["au"][idx])
                    + 0.12 * core_enrich * (state.metal_channels["w"][idx] + state.metal_channels["mo"][idx])
                    + 0.11 * edge_enrich * (state.metal_channels["pb"][idx] + state.metal_channels["zn"][idx] + state.metal_channels["ag"][idx])
                )


def stage4_alteration_halo(state: OreState) -> None:
    """Step 10: generate alteration halos (potassic -> phyllic -> propylitic)."""
    alteration = state.metadata["alteration_index"]
    for idx in range(state.voxel_count):
        intensity = state.fluid_flux[idx] * state.permeability[idx]
        if intensity < 0.04:
            alteration[idx] = 0.0
            continue

        temp = state.temperature[idx]
        if temp >= 0.68:
            zone = 1.0  # potassic-like
            state.reactivity[idx] = clamp01(state.reactivity[idx] + 0.1)
        elif temp >= 0.48:
            zone = 0.66  # phyllic-like
        else:
            zone = 0.33  # propylitic-like

        alteration[idx] = zone
        state.potential[idx] = clamp01(state.potential[idx] + 0.06 * zone * intensity)


def stage5_supergene_enrichment(state: OreState, rng: random.Random) -> int:
    """Step 11: top-down oxidation leaching and water-table secondary enrichment."""
    x_size, y_size, z_size = state.grid_size
    if z_size <= 2:
        return 0

    oxygen = state.metadata["oxygen_water"]
    water_table = int(0.68 * (z_size - 1))
    triggered = 0

    for y in range(y_size):
        for x in range(x_size):
            leached = 0.0
            for z in range(z_size - 1, water_table, -1):
                idx = state.index(x, y, z)
                oxygen[idx] = clamp01(oxygen[idx] + 0.08)
                state.eh[idx] = clamp01(state.eh[idx] + 0.08)

                # Oxidize and leach shallow low-grade sulfide-like material.
                if 0.01 <= state.potential[idx] <= 0.42 and (state.eh[idx] > 0.44 or oxygen[idx] > 0.52):
                    amount = state.potential[idx] * 0.22
                    state.potential[idx] = clamp01(state.potential[idx] - amount)
                    leached += amount
                    triggered += 1
                elif state.eh[idx] > 0.5 and oxygen[idx] > 0.58:
                    # Even weakly mineralized zones can release dispersed metals during oxidation.
                    dispersed = (
                        state.metal_channels["cu"][idx]
                        + state.metal_channels["pb"][idx]
                        + state.metal_channels["zn"][idx]
                    ) * 0.08
                    if dispersed > 0.0:
                        leached += dispersed
                        triggered += 1

            if leached <= 0.0:
                continue

            target_z = min(z_size - 1, max(0, water_table + rng.choice((-1, 0, 1))))
            target_idx = state.index(x, y, target_z)
            reducing_factor = clamp01(1.0 - state.eh[target_idx])
            secondary = leached * (0.65 + 0.35 * reducing_factor)

            state.potential[target_idx] = clamp01(state.potential[target_idx] + secondary)
            state.eh[target_idx] = clamp01(state.eh[target_idx] - 0.1)
            state.porosity[target_idx] = clamp01(state.porosity[target_idx] + 0.05)

    return triggered


def run_staged_metallogenesis(
    grid_size: Tuple[int, int, int],
    rng: random.Random,
    style: str = "default",
) -> Tuple[OreState, List[Dict[str, float]]]:
    """Execute stage-0 to stage-5 geological simulation and return incubated state."""
    if style not in {"default", "stockwork", "lens"}:
        raise ValueError("style must be one of: default, stockwork, lens")

    state = _make_empty_state(grid_size)
    stage_logs: List[Dict[str, float]] = []

    def log(
        stage_id: int,
        name: str,
        keyword: str,
        before: float,
        extra: Optional[Dict[str, float]] = None,
    ) -> None:
        after = _mean(state.potential)
        record: Dict[str, float] = {
            "rank": float(stage_id),
            "mean_before": before,
            "mean_after": after,
            "delta": after - before,
            "name": name,
            "keyword": keyword,
            "effective_weight": 1.0,
        }
        if extra:
            record.update(extra)
        stage_logs.append(record)

    before = _mean(state.potential)
    stage0_initialize_background(state, rng)
    hits = stage0_structural_weakness(state, rng)
    stage0_apply_centripetal_permeability(state)
    cooled = stage0_apply_boundary_cooling_barrier(state, edge_width=3)
    style_extra: Dict[str, float] = {}
    style_keyword = ""
    style_name = ""
    if style == "stockwork":
        style_extra = stage0_apply_stockwork_anisotropy(state, rng)
        style_keyword = "|Anisotropic_Permeability"
        style_name = " + Stockwork Anisotropy"
    elif style == "lens":
        style_extra = stage0_initialize_stratabound_channels(state, rng)
        style_keyword = "|Stratabound_Channel_Mask"
        style_name = " + Stratabound Channels"

    stage0_extra: Dict[str, float] = {
        "fault_hits": float(hits),
        "boundary_cells": float(cooled),
    }
    stage0_extra.update(style_extra)
    stage0_extra["staged_style"] = style

    log(
        1,
        f"Background + Structural + Permeability Focus + Boundary Cooling{style_name}",
        f"Tensor_Definition|Fault_Zone_Generation|Permeability_Gradient|Dirichlet_Boundary{style_keyword}",
        before,
        stage0_extra,
    )

    before = _mean(state.potential)
    source_info = stage1_activate_source(state, rng)
    stage1_complexation(state)
    log(
        2,
        "Source Activation + Complexation",
        "Heat_Source_Boundary|Supercritical_Fluid_Injection|Complexation",
        before,
        {
            "source_cells": float(source_info["source_cells"]),
            "source_radius": float(source_info["source_radius"]),
        },
    )

    before = _mean(state.potential)
    stage2_darcy_advection(state, cycles=8)
    transport_extra: Dict[str, float] = {}
    if style == "stockwork":
        fractures = stage2_hydraulic_fracturing_stockwork(state)
        transport_extra["fracture_events"] = float(fractures)
        transport_name = "Transport + Hydraulic Fracturing (Stockwork)"
        transport_keyword = "Advection_Diffusion|Hydraulic_Fracture|Directional_Branching"
    elif style == "lens":
        channel_moves = stage2_stratabound_flow_focus(state, cycles=5)
        fractures = stage2_hydraulic_fracturing(state)
        transport_extra["channel_moves"] = float(channel_moves)
        transport_extra["fracture_events"] = float(fractures)
        transport_name = "Transport + Stratabound Channel Flow"
        transport_keyword = "Advection_Diffusion|Stratabound_Flow|Hydraulic_Fracture"
    else:
        fractures = stage2_hydraulic_fracturing(state)
        transport_extra["fracture_events"] = float(fractures)
        transport_name = "Transport + Hydraulic Fracturing"
        transport_keyword = "Advection_Diffusion|Hydraulic_Fracture"

    log(3, transport_name, transport_keyword, before, transport_extra)

    before = _mean(state.potential)
    boiling_hits = stage3_boiling_precipitation(state)
    mixing_hits = stage3_mixing_and_neutralization(state)
    stage3_extra: Dict[str, float] = {
        "boiling_hits": float(boiling_hits),
        "mixing_hits": float(mixing_hits),
    }
    stage3_name = "Boiling + Fluid Mixing Precipitation"
    stage3_keyword = "Depressurization_Boiling|Fluid_Mixing|pH_Neutralization"

    if style == "stockwork":
        pulse_hits = stage3_stockwork_pulse_precipitation(state)
        stage3_extra["pulse_precip_hits"] = float(pulse_hits)
        stage3_name = "Boiling + Pulse Unloading (Stockwork)"
        stage3_keyword = "Depressurization_Boiling|Pulse_Unloading|Channel_Sealing"
    elif style == "lens":
        dilation_hits = stage3_metasomatic_dilation(state, rng, rounds=2)
        stage3_extra["dilation_hits"] = float(dilation_hits)
        stage3_name = "Boiling + Metasomatic Dilation"
        stage3_keyword = "Depressurization_Boiling|Fluid_Mixing|Metasomatic_Dilation"

    log(
        4,
        stage3_name,
        stage3_keyword,
        before,
        stage3_extra,
    )

    before = _mean(state.potential)
    stage4_zonation(state)
    stage4_alteration_halo(state)
    log(5, "Zonation + Alteration Halo", "Spatial_Metal_Zonation|Alteration_Index", before)

    before = _mean(state.potential)
    enrich_hits = stage5_supergene_enrichment(state, rng)
    log(6, "Supergene Enrichment", "Meteoric_Water_Percolation|Secondary_Enrichment", before, {"supergene_hits": float(enrich_hits)})

    if style == "lens":
        before = _mean(state.potential)
        deform_info = stage5_ductile_shear_fold_deformation(state, rng)
        log(
            7,
            "Post-ore Ductile Shear/Folding",
            "Vector_Displacement_Field|Ductile_Deformation|Boudinage",
            before,
            deform_info,
        )

    state.clamp_all()
    return state, stage_logs
