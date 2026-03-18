from __future__ import annotations

import math
import random
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, map_coordinates, shift as nd_shift
from skimage.measure import marching_cubes

from ore_state import OreState, clamp01


@dataclass
class PhysicsConfig:
    grid_size: Tuple[int, int, int]
    boundary_layers: int = 3
    base_permeability: float = 0.1
    boundary_permeability: float = 1e-10
    seed_size: int = 4
    seed_fluid: float = 100.0
    seed_temperature: float = 800.0
    perlin_gain: float = 0.45
    perlin_frequency: float = 0.11
    time_steps: int = 50
    temperature_sigma: float = 1.0
    max_mobile_fraction: float = 0.35
    temperature_threshold: float = 300.0
    dilation_iterations: int = 1
    dilation_boost: float = 0.18
    cutoff_grade: float = 5.0
    apply_shear: bool = True
    shear_compress: float = 0.28
    shear_z: float = 0.22
    shear_fold: float = 1.0
    peclet_number: float = 1.0
    damkohler_number: float = 1.0


def _save_timestep_snapshot(
    snapshot_dir: Path,
    step: int,
    ore_grade: np.ndarray,
    fluid_metal: np.ndarray,
    temperature: np.ndarray,
    permeability: np.ndarray,
    precip_mask: np.ndarray,
    precip_amount: float,
) -> str:
    file_name = f"step_{step:04d}.npz"
    file_path = snapshot_dir / file_name

    np.savez_compressed(
        file_path,
        step=np.int32(step),
        ore_grade=ore_grade.astype(np.float32),
        fluid_metal=fluid_metal.astype(np.float32),
        temperature=temperature.astype(np.float32),
        permeability=permeability.astype(np.float32),
        precip_mask=precip_mask.astype(np.uint8),
        precip_amount=np.float32(precip_amount),
    )
    return file_name


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def _hash_noise(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, seed: int) -> np.ndarray:
    n = ix * 73856093 ^ iy * 19349663 ^ iz * 83492791 ^ seed * 2654435761
    n = (n << 13) ^ n
    raw = (n * (n * n * 15731 + 789221) + 1376312589) & 0x7FFFFFFF
    return 1.0 - raw.astype(np.float64) / 1073741824.0


def _value_noise3(x: np.ndarray, y: np.ndarray, z: np.ndarray, seed: int) -> np.ndarray:
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    z0 = np.floor(z).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    tx = _fade(x - x0)
    ty = _fade(y - y0)
    tz = _fade(z - z0)

    n000 = _hash_noise(x0, y0, z0, seed)
    n100 = _hash_noise(x1, y0, z0, seed)
    n010 = _hash_noise(x0, y1, z0, seed)
    n110 = _hash_noise(x1, y1, z0, seed)
    n001 = _hash_noise(x0, y0, z1, seed)
    n101 = _hash_noise(x1, y0, z1, seed)
    n011 = _hash_noise(x0, y1, z1, seed)
    n111 = _hash_noise(x1, y1, z1, seed)

    nx00 = n000 + (n100 - n000) * tx
    nx10 = n010 + (n110 - n010) * tx
    nx01 = n001 + (n101 - n001) * tx
    nx11 = n011 + (n111 - n011) * tx
    nxy0 = nx00 + (nx10 - nx00) * ty
    nxy1 = nx01 + (nx11 - nx01) * ty
    nxyz = nxy0 + (nxy1 - nxy0) * tz
    return 0.5 + 0.5 * nxyz


def _fractal_noise3(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    seed: int,
    base_freq: float,
    octaves: int = 3,
    gain: float = 0.5,
) -> np.ndarray:
    total = np.zeros_like(x, dtype=np.float64)
    amp = 1.0
    freq = base_freq
    norm = 0.0

    for octave in range(max(1, octaves)):
        total += amp * _value_noise3(x * freq, y * freq, z * freq, seed + octave * 7919)
        norm += amp
        amp *= gain
        freq *= 2.0

    return total / max(1e-6, norm)


def _boundary_mask(grid_size: Tuple[int, int, int], layers: int) -> np.ndarray:
    x_size, y_size, z_size = grid_size
    x, y, z = np.indices((x_size, y_size, z_size))
    edge_dist = np.minimum.reduce(
        [
            x,
            y,
            z,
            (x_size - 1) - x,
            (y_size - 1) - y,
            (z_size - 1) - z,
        ]
    )
    return edge_dist < layers


def _apply_directional_perlin(
    permeability: np.ndarray,
    grid_size: Tuple[int, int, int],
    rng: random.Random,
    cfg: PhysicsConfig,
) -> None:
    x_size, y_size, z_size = grid_size
    x, y, z = np.indices((x_size, y_size, z_size)).astype(np.float64)

    cx = (x_size - 1) / 2.0
    cy = (y_size - 1) / 2.0
    cz = (z_size - 1) / 2.0

    xn = (x - cx) / max(1.0, cx)
    yn = (y - cy) / max(1.0, cy)
    zn = (z - cz) / max(1.0, cz)

    theta = math.radians(rng.uniform(20.0, 70.0))
    phi = math.radians(rng.uniform(10.0, 35.0))
    dx = math.cos(phi) * math.cos(theta)
    dy = math.cos(phi) * math.sin(theta)
    dz = math.sin(phi)

    seed = rng.randrange(1, 2**31)
    noise = _fractal_noise3(x, y, z, seed, base_freq=cfg.perlin_frequency, octaves=4, gain=0.58)

    proj = xn * dx + yn * dy + zn * dz
    ridge = 0.5 + 0.5 * np.sin(10.0 * proj + 4.5 * noise)

    r2 = xn * xn + yn * yn + zn * zn
    center_weight = np.exp(-r2 / (2.0 * 0.7 * 0.7))

    channel = np.clip(0.62 * noise + 0.38 * ridge, 0.0, 1.0)
    permeability += cfg.perlin_gain * channel * center_weight


def _seed_center_block(
    fluid: np.ndarray,
    temperature: np.ndarray,
    cfg: PhysicsConfig,
) -> Tuple[slice, slice, slice]:
    x_size, y_size, z_size = fluid.shape
    block = max(1, cfg.seed_size)

    sx = max(0, x_size // 2 - block // 2)
    sy = max(0, y_size // 2 - block // 2)
    sz = max(0, z_size // 2 - block // 2)

    ex = min(x_size, sx + block)
    ey = min(y_size, sy + block)
    ez = min(z_size, sz + block)

    seed_slice = (slice(sx, ex), slice(sy, ey), slice(sz, ez))
    fluid[seed_slice] = cfg.seed_fluid
    temperature[seed_slice] = cfg.seed_temperature
    return seed_slice


def _transport_step(
    fluid: np.ndarray,
    permeability: np.ndarray,
    cfg: PhysicsConfig,
) -> np.ndarray:
    directions = (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    )

    drives: List[np.ndarray] = []
    for d in directions:
        neighbor_perm = nd_shift(permeability, shift=d, order=0, mode="constant", cval=0.0, prefilter=False)
        drives.append(np.clip(neighbor_perm - permeability, 0.0, None))

    total_drive = np.zeros_like(fluid)
    for drive in drives:
        total_drive += drive

    pe_scale = np.clip(float(cfg.peclet_number), 0.1, 6.0)
    mobile_fraction = np.clip((0.06 + 0.55 * permeability) * pe_scale, 0.0, cfg.max_mobile_fraction)
    outflow_budget = fluid * mobile_fraction

    moved_out = np.zeros_like(fluid)
    moved_in = np.zeros_like(fluid)

    safe_total = np.maximum(total_drive, 1e-12)
    active = total_drive > 1e-12

    for d, drive in zip(directions, drives):
        transfer = np.where(active, outflow_budget * (drive / safe_total), 0.0)
        moved_out += transfer
        moved_in += nd_shift(transfer, shift=d, order=0, mode="constant", cval=0.0, prefilter=False)

    next_fluid = np.clip(fluid - moved_out + moved_in, 0.0, None)
    return next_fluid


def _apply_ductile_shear(
    ore_grade: np.ndarray,
    cfg: PhysicsConfig,
    rng: random.Random,
) -> np.ndarray:
    x_size, y_size, z_size = ore_grade.shape
    x, y, z = np.indices((x_size, y_size, z_size)).astype(np.float64)

    cx = (x_size - 1) / 2.0
    cy = (y_size - 1) / 2.0
    cz = (z_size - 1) / 2.0

    xn = (x - cx) / max(1.0, cx)
    yn = (y - cy) / max(1.0, cy)

    phase = rng.uniform(0.0, 2.0 * math.pi)
    disp_x = -cfg.shear_compress * xn * np.abs(xn) * (0.45 * x_size)
    disp_z = cfg.shear_z * xn * (0.3 * z_size)
    disp_z += cfg.shear_fold * np.sin(2.0 * math.pi * yn + phase) * (0.1 * z_size)

    sample_x = x - disp_x
    sample_y = y
    sample_z = z - disp_z

    deformed = map_coordinates(
        ore_grade,
        [sample_x, sample_y, sample_z],
        order=1,
        mode="constant",
        cval=0.0,
    )
    return np.clip(deformed, 0.0, None)


def _build_state(
    ore_grade: np.ndarray,
    temperature: np.ndarray,
    permeability: np.ndarray,
    fluid: np.ndarray,
    cfg: PhysicsConfig,
) -> OreState:
    max_grade = float(np.max(ore_grade))
    if max_grade <= 1e-12:
        potential = np.zeros_like(ore_grade, dtype=np.float64)
    else:
        potential = np.clip(ore_grade / max_grade, 0.0, 1.0)

    temp_norm = np.clip(temperature / max(1e-6, cfg.seed_temperature), 0.0, 1.0)
    fluid_norm = np.clip(fluid / max(1e-6, cfg.seed_fluid), 0.0, 1.0)
    perm_norm = np.clip(permeability, 0.0, 1.0)

    x_size, y_size, z_size = cfg.grid_size
    voxel_count = x_size * y_size * z_size

    return OreState(
        grid_size=cfg.grid_size,
        potential=potential.reshape(voxel_count).astype(np.float64).tolist(),
        temperature=temp_norm.reshape(voxel_count).astype(np.float64).tolist(),
        pressure=np.zeros(voxel_count, dtype=np.float64).tolist(),
        permeability=perm_norm.reshape(voxel_count).astype(np.float64).tolist(),
        structure=perm_norm.reshape(voxel_count).astype(np.float64).tolist(),
        fluid_flux=fluid_norm.reshape(voxel_count).astype(np.float64).tolist(),
        reactivity=np.zeros(voxel_count, dtype=np.float64).tolist(),
        preservation=np.ones(voxel_count, dtype=np.float64).tolist(),
        metadata={
            "physics_pipeline": True,
            "physics_raw_ore_grade": ore_grade.reshape(voxel_count).astype(np.float64).tolist(),
            "physics_cutoff_grade": cfg.cutoff_grade,
        },
    )


def export_obj_from_raw_grade(
    ore_grade: np.ndarray,
    file_path: Path,
    level: float,
) -> Tuple[int, int]:
    if ore_grade.ndim != 3:
        raise ValueError("ore_grade must be a 3D numpy array")

    if float(np.max(ore_grade)) < level:
        file_path.write_text("# Empty mesh: no voxels above level\n", encoding="utf-8")
        return 0, 0

    verts, faces, _normals, _values = marching_cubes(ore_grade.astype(np.float32), level=level)

    with file_path.open("w", encoding="utf-8") as obj_file:
        obj_file.write("# Marching Cubes mesh exported by VoxelOreGen physics pipeline\n")
        obj_file.write(f"# level={level:.4f}\n")

        # skimage returns coordinates in array-axis order; remap to x,y,z for OBJ viewers.
        for vx, vy, vz in verts:
            ox = vz
            oy = vy
            oz = vx
            obj_file.write(f"v {ox:.6f} {oy:.6f} {oz:.6f}\n")

        for i0, i1, i2 in faces:
            a = int(i0) + 1
            b = int(i1) + 1
            c = int(i2) + 1
            obj_file.write(f"f {a} {b} {c}\n")

    return int(verts.shape[0]), int(faces.shape[0])


def export_ply_from_raw_grade(
    ore_grade: np.ndarray,
    file_path: Path,
    level: float,
    color: Tuple[int, int, int] = (242, 92, 48),
) -> Tuple[int, int]:
    if ore_grade.ndim != 3:
        raise ValueError("ore_grade must be a 3D numpy array")

    if float(np.max(ore_grade)) < level:
        file_path.write_text(
            "ply\nformat ascii 1.0\ncomment Empty mesh: no voxels above level\n"
            "element vertex 0\nproperty float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "element face 0\nproperty list uchar int vertex_index\nend_header\n",
            encoding="utf-8",
        )
        return 0, 0

    verts, faces, _normals, _values = marching_cubes(ore_grade.astype(np.float32), level=level)
    red, green, blue = color

    with file_path.open("w", encoding="utf-8") as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("comment Marching Cubes mesh exported by VoxelOreGen physics pipeline\n")
        ply_file.write(f"comment level={level:.4f}\n")
        ply_file.write(f"element vertex {verts.shape[0]}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write(f"element face {faces.shape[0]}\n")
        ply_file.write("property list uchar int vertex_index\n")
        ply_file.write("end_header\n")

        # skimage returns coordinates in array-axis order; remap to x,y,z for viewer consistency.
        for vx, vy, vz in verts:
            ox = vz
            oy = vy
            oz = vx
            ply_file.write(f"{ox:.6f} {oy:.6f} {oz:.6f} {red} {green} {blue}\n")

        for i0, i1, i2 in faces:
            ply_file.write(f"3 {int(i0)} {int(i1)} {int(i2)}\n")

    return int(verts.shape[0]), int(faces.shape[0])


def run_physics_voxel_growth(
    grid_size: Tuple[int, int, int],
    rng: random.Random,
    time_steps: int = 50,
    temperature_threshold: float = 300.0,
    cutoff_grade: float = 5.0,
    boundary_layers: int = 3,
    seed_size: int = 4,
    apply_shear: bool = True,
    peclet_number: float = 1.0,
    damkohler_number: float = 1.0,
    snapshot_dir: Optional[Path] = None,
    snapshot_every: int = 1,
    snapshot_include_initial: bool = True,
) -> Tuple[OreState, List[Dict[str, float]]]:
    pe = max(0.05, float(peclet_number))
    da = max(0.05, float(damkohler_number))

    cfg = PhysicsConfig(
        grid_size=grid_size,
        time_steps=time_steps,
        temperature_threshold=temperature_threshold,
        cutoff_grade=cutoff_grade,
        boundary_layers=boundary_layers,
        seed_size=seed_size,
        apply_shear=apply_shear,
        peclet_number=pe,
        damkohler_number=da,
    )

    x_size, y_size, z_size = grid_size
    ore_grade = np.zeros((x_size, y_size, z_size), dtype=np.float64)
    temperature = np.zeros((x_size, y_size, z_size), dtype=np.float64)
    fluid_metal = np.zeros((x_size, y_size, z_size), dtype=np.float64)
    permeability = np.full((x_size, y_size, z_size), cfg.base_permeability, dtype=np.float64)

    logs: List[Dict[str, float]] = []
    snapshot_files: List[str] = []

    if snapshot_every < 1:
        raise ValueError("snapshot_every must be >= 1")

    if snapshot_dir is not None:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    boundary = _boundary_mask(grid_size, cfg.boundary_layers)
    permeability[boundary] = cfg.boundary_permeability
    logs.append(
        {
            "rank": 1.0,
            "name": "Initialization + Boundary Lock",
            "keyword": "Grid_Init|Permeability_Boundary_Lock",
            "effective_weight": 1.0,
            "mean_before": 0.0,
            "mean_after": float(np.mean(permeability)),
            "delta": float(np.mean(permeability)),
            "boundary_cells": float(np.sum(boundary)),
        }
    )

    if snapshot_dir is not None and snapshot_include_initial:
        initial_mask = np.zeros((x_size, y_size, z_size), dtype=bool)
        snapshot_files.append(
            _save_timestep_snapshot(
                snapshot_dir=snapshot_dir,
                step=0,
                ore_grade=ore_grade,
                fluid_metal=fluid_metal,
                temperature=temperature,
                permeability=permeability,
                precip_mask=initial_mask,
                precip_amount=0.0,
            )
        )

    seed_slice = _seed_center_block(fluid_metal, temperature, cfg)
    logs.append(
        {
            "rank": 2.0,
            "name": "Central Ore Seed",
            "keyword": "Central_Fluid_Seed|Central_Temperature_Seed",
            "effective_weight": 1.0,
            "mean_before": 0.0,
            "mean_after": float(np.mean(fluid_metal)),
            "delta": float(np.mean(fluid_metal)),
            "seed_voxels": float(np.prod([s.stop - s.start for s in seed_slice])),
        }
    )

    mean_before = float(np.mean(permeability))
    _apply_directional_perlin(permeability, grid_size, rng, cfg)
    if pe != 1.0:
        permeability *= np.clip(0.82 + 0.18 * pe, 0.6, 2.0)
        np.clip(permeability, 0.0, 1.0, out=permeability)
    permeability[boundary] = cfg.boundary_permeability
    logs.append(
        {
            "rank": 3.0,
            "name": "Directional High-Permeability Channels",
            "keyword": "Directional_Perlin|Fracture_Corridor",
            "effective_weight": 1.0,
            "mean_before": mean_before,
            "mean_after": float(np.mean(permeability)),
            "delta": float(np.mean(permeability) - mean_before),
            "peclet_number": pe,
        }
    )

    total_precipitated = 0.0
    dilation_events = 0
    for step in range(1, cfg.time_steps + 1):
        temperature = gaussian_filter(temperature, sigma=cfg.temperature_sigma, mode="constant", cval=0.0)
        temperature *= 0.995

        fluid_metal = _transport_step(fluid_metal, permeability, cfg)
        fluid_metal[boundary] = 0.0

        precip_mask = (fluid_metal > 1e-6) & (temperature < cfg.temperature_threshold)
        temp_rel = np.clip(
            (cfg.temperature_threshold - temperature) / max(1e-6, cfg.temperature_threshold),
            0.0,
            1.0,
        )
        reaction_eff = np.clip(da * (0.35 + 0.65 * temp_rel), 0.0, 1.0)
        precip = np.where(precip_mask, fluid_metal * reaction_eff, 0.0)
        precip_amount = float(np.sum(precip))
        fluid_metal -= precip
        ore_grade += precip
        total_precipitated += precip_amount

        if np.any(precip_mask):
            dilated = binary_dilation(precip_mask, iterations=cfg.dilation_iterations)
            neighbor_ring = dilated & (~precip_mask)
            if np.any(neighbor_ring):
                permeability[neighbor_ring] = np.clip(
                    permeability[neighbor_ring] + cfg.dilation_boost,
                    cfg.boundary_permeability,
                    1.0,
                )
                dilation_events += int(np.sum(neighbor_ring))

        permeability[boundary] = cfg.boundary_permeability

        if snapshot_dir is not None and (step % snapshot_every == 0 or step == cfg.time_steps):
            snapshot_files.append(
                _save_timestep_snapshot(
                    snapshot_dir=snapshot_dir,
                    step=step,
                    ore_grade=ore_grade,
                    fluid_metal=fluid_metal,
                    temperature=temperature,
                    permeability=permeability,
                    precip_mask=precip_mask,
                    precip_amount=precip_amount,
                )
            )

    logs.append(
        {
            "rank": 4.0,
            "name": "Temperature Diffusion",
            "keyword": "Gaussian_Diffusion|Core_Cooling",
            "effective_weight": 1.0,
            "mean_before": 0.0,
            "mean_after": float(np.mean(temperature)),
            "delta": float(np.mean(temperature)),
        }
    )
    logs.append(
        {
            "rank": 5.0,
            "name": "Fluid Seepage Transport",
            "keyword": "Permeability_Gradient_Advection|Boundary_Truncation_Shift",
            "effective_weight": 1.0,
            "mean_before": 0.0,
            "mean_after": float(np.mean(fluid_metal)),
            "delta": float(np.mean(fluid_metal)),
        }
    )
    logs.append(
        {
            "rank": 6.0,
            "name": "Precipitation Unloading",
            "keyword": "Temperature_Threshold_Precipitation",
            "effective_weight": 1.0,
            "mean_before": 0.0,
            "mean_after": float(np.mean(ore_grade)),
            "delta": float(np.mean(ore_grade)),
            "precipitated_total": total_precipitated,
            "damkohler_number": da,
        }
    )
    logs.append(
        {
            "rank": 7.0,
            "name": "Reactive Dilation",
            "keyword": "Binary_Dilation|Permeability_Expansion",
            "effective_weight": 1.0,
            "mean_before": 0.0,
            "mean_after": float(np.mean(permeability)),
            "delta": float(np.mean(permeability)),
            "dilation_events": float(dilation_events),
        }
    )

    if cfg.apply_shear:
        before_shear = float(np.mean(ore_grade))
        ore_grade = _apply_ductile_shear(ore_grade, cfg, rng)
        logs.append(
            {
                "rank": 8.0,
                "name": "Post-ore Ductile Shear",
                "keyword": "Map_Coordinates|Vector_Field_Deformation",
                "effective_weight": 1.0,
                "mean_before": before_shear,
                "mean_after": float(np.mean(ore_grade)),
                "delta": float(np.mean(ore_grade) - before_shear),
            }
        )
    else:
        logs.append(
            {
                "rank": 8.0,
                "name": "Post-ore Ductile Shear (Skipped)",
                "keyword": "Map_Coordinates|Optional",
                "effective_weight": 1.0,
                "mean_before": float(np.mean(ore_grade)),
                "mean_after": float(np.mean(ore_grade)),
                "delta": 0.0,
            }
        )

    before_cutoff = float(np.mean(ore_grade))
    ore_grade[ore_grade < cfg.cutoff_grade] = 0.0
    logs.append(
        {
            "rank": 9.0,
            "name": "Boundary Grade Cutoff",
            "keyword": "Cutoff_Grade_Threshold",
            "effective_weight": 1.0,
            "mean_before": before_cutoff,
            "mean_after": float(np.mean(ore_grade)),
            "delta": float(np.mean(ore_grade) - before_cutoff),
            "cutoff_grade": cfg.cutoff_grade,
        }
    )

    above = int(np.sum(ore_grade >= cfg.cutoff_grade))
    logs.append(
        {
            "rank": 10.0,
            "name": "Marching Cubes Ready",
            "keyword": "Marching_Cubes|OBJ_Export",
            "effective_weight": 1.0,
            "mean_before": float(np.mean(ore_grade)),
            "mean_after": float(np.mean(ore_grade)),
            "delta": 0.0,
            "cutoff_voxels": float(above),
        }
    )

    if snapshot_dir is not None:
        manifest = {
            "snapshot_count": len(snapshot_files),
            "time_steps": cfg.time_steps,
            "snapshot_every": snapshot_every,
            "include_initial": snapshot_include_initial,
            "grid_size": list(cfg.grid_size),
            "temperature_threshold": cfg.temperature_threshold,
            "cutoff_grade": cfg.cutoff_grade,
            "files": snapshot_files,
        }
        with (snapshot_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        logs.append(
            {
                "rank": 10.5,
                "name": "Timestep Snapshots Exported",
                "keyword": "NPZ_Snapshot|Per_Timestep_State",
                "effective_weight": 1.0,
                "mean_before": float(np.mean(ore_grade)),
                "mean_after": float(np.mean(ore_grade)),
                "delta": 0.0,
                "snapshot_count": float(len(snapshot_files)),
            }
        )

    state = _build_state(ore_grade, temperature, permeability, fluid_metal, cfg)
    state.metadata["peclet_number"] = pe
    state.metadata["damkohler_number"] = da
    if snapshot_dir is not None:
        state.metadata["physics_snapshot_dir"] = str(snapshot_dir)
        state.metadata["physics_snapshot_count"] = len(snapshot_files)
        state.metadata["physics_snapshot_every"] = snapshot_every
    state.clamp_all()
    return state, logs
