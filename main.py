from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from incubation_engine import incubate_seed
from ore_state import OreState, clamp01
from physics_pipeline import export_obj_from_raw_grade, export_ply_from_raw_grade, run_physics_voxel_growth
from seed_generator import generate_seed_state
from staged_metallogenesis import run_staged_metallogenesis
from workflow_generator import export_obj_isosurface, export_ply_isosurface, run_voxel_workflow


# 0 = 非矿体，1~4 = 从低到高的矿石品位类别。
MATERIAL_COLORS = {
    0: (60, 120, 255),   # 非矿体（蓝色）
    1: (255, 214, 102),  # 低品位矿体
    2: (255, 158, 70),   # 中品位矿体
    3: (242, 92, 48),    # 高品位矿体
    4: (204, 32, 32),    # 富矿体
}

NON_ORE_ID = 0
GRADE_THRESHOLDS = (0.12, 0.3, 0.55, 0.8)


def parse_grid_size(value: str) -> Tuple[int, int, int]:
    """从 '32*32*32'、'32x32x32' 或 '32,32,32' 解析网格尺寸。"""
    normalized = value.lower().replace("x", "*").replace(",", "*").replace(" ", "")
    parts = [part for part in normalized.split("*") if part]

    if len(parts) == 1:
        parts = [parts[0], parts[0], parts[0]]

    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "grid-size format must be like 32*32*32 (or 32x32x32 / 32,32,32)."
        )

    try:
        x, y, z = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("grid-size must contain integers only.") from exc

    if x <= 0 or y <= 0 or z <= 0:
        raise argparse.ArgumentTypeError("grid-size values must be positive.")

    return x, y, z


def parse_transparency(value: str) -> float:
    """解析 [0, 1] 范围透明度，0 表示不透明，1 表示完全透明。"""
    try:
        transparency = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("non-ore-transparency must be a float in [0, 1].") from exc

    if transparency < 0.0 or transparency > 1.0:
        raise argparse.ArgumentTypeError("non-ore-transparency must be in [0, 1].")

    return transparency


def parse_unit_interval(value: str) -> float:
    """解析 [0, 1] 范围内的通用比例值。"""
    try:
        ratio = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a float in [0, 1].") from exc

    if ratio < 0.0 or ratio > 1.0:
        raise argparse.ArgumentTypeError("value must be in [0, 1].")
    return ratio


def parse_positive_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive float.") from exc

    if result <= 0.0:
        raise argparse.ArgumentTypeError("value must be > 0.")
    return result


def transparency_to_alpha(transparency: float) -> int:
    """将透明度比例转换为 8 位 alpha（255 不透明，0 透明）。"""
    return int(round((1.0 - transparency) * 255.0))


def grade_from_potential(potential: float, rng: random.Random) -> float:
    """将孵化后的矿化势转换为品位，并加入块金效应扰动。"""
    local_noise = rng.gauss(0.0, 0.012 + 0.04 * potential)

    nugget_effect = 0.0
    if rng.random() < (0.01 + 0.08 * potential):
        nugget_effect = rng.uniform(0.08, 0.32) * (0.35 + potential)

    grade = clamp01(potential + local_noise + nugget_effect)
    if grade < 0.02:
        return 0.0
    return grade


def grade_to_material_id(grade: float) -> int:
    if grade < GRADE_THRESHOLDS[0]:
        return 0
    if grade < GRADE_THRESHOLDS[1]:
        return 1
    if grade < GRADE_THRESHOLDS[2]:
        return 2
    if grade < GRADE_THRESHOLDS[3]:
        return 3
    return 4


def rebalance_potential_after_incubation(state: OreState) -> None:
    """多阶段增强后重标定矿化势，保持更真实的非矿体背景。"""
    if state.voxel_count <= 2:
        return

    sorted_values = sorted(state.potential)
    low_idx = int(0.55 * (state.voxel_count - 1))
    high_idx = int(0.985 * (state.voxel_count - 1))
    low = sorted_values[low_idx]
    high = sorted_values[high_idx]
    span = max(1e-6, high - low)

    for idx, value in enumerate(state.potential):
        normalized = clamp01((value - low) / span)
        # gamma > 1 可保留富矿核心，同时抑制弱背景矿化。
        state.potential[idx] = normalized ** 1.6


def normalize_potential(values: List[float], low_q: float, high_q: float, gamma: float) -> List[float]:
    if not values:
        return []

    n = len(values)
    sorted_values = sorted(values)
    low_idx = int(max(0, min(n - 1, round((n - 1) * low_q))))
    high_idx = int(max(0, min(n - 1, round((n - 1) * high_q))))
    low = sorted_values[low_idx]
    high = sorted_values[high_idx]
    span = max(1e-6, high - low)

    normalized = [0.0] * n
    gamma = max(0.2, gamma)
    for idx, value in enumerate(values):
        normalized[idx] = clamp01((value - low) / span) ** gamma
    return normalized


def smooth_positive_voxels(state: OreState, values: List[float], passes: int) -> List[float]:
    if passes <= 0:
        return values

    current = values[:]
    for _ in range(passes):
        nxt = current[:]
        for idx, value in enumerate(current):
            if value <= 0.0:
                continue
            x, y, z = state.xyz(idx)
            total = value
            count = 1
            for nidx in state.iter_neighbors6(x, y, z):
                total += current[nidx]
                count += 1
            local_mean = total / count
            nxt[idx] = clamp01(0.72 * value + 0.28 * local_mean)
        current = nxt
    return current


def build_hybrid_state(
    staged_state: OreState,
    legacy_state: OreState,
    staged_weight: float,
    hard_threshold: float,
    smooth_passes: int,
    boundary_strength: float,
    boundary_power: float,
    boundary_floor: float,
) -> Tuple[OreState, List[Dict[str, float]]]:
    staged_norm = normalize_potential(staged_state.potential, low_q=0.5, high_q=0.99, gamma=1.25)
    legacy_norm = normalize_potential(legacy_state.potential, low_q=0.5, high_q=0.99, gamma=1.15)

    hybrid = [0.0] * staged_state.voxel_count
    for idx in range(staged_state.voxel_count):
        mixed = staged_weight * staged_norm[idx] + (1.0 - staged_weight) * legacy_norm[idx]
        structure_mix = 0.5 * (staged_state.structure[idx] + legacy_state.structure[idx])
        # 结构引导调制：保持条带几何特征，同时维持团块连续性。
        hybrid[idx] = clamp01(mixed * (0.82 + 0.36 * structure_mix))

    hybrid = smooth_positive_voxels(staged_state, hybrid, smooth_passes)

    # 边界衰减：靠近边界处降低值，中心区域保持较高值。
    x_size, y_size, z_size = staged_state.grid_size
    max_radius = max(1e-6, min((x_size - 1) / 2.0, (y_size - 1) / 2.0, (z_size - 1) / 2.0))
    for idx, value in enumerate(hybrid):
        if value <= 0.0:
            continue
        x, y, z = staged_state.xyz(idx)
        edge_dist = min(x, x_size - 1 - x, y, y_size - 1 - y, z, z_size - 1 - z)
        radial = clamp01(edge_dist / max_radius)
        center_weight = radial ** boundary_power
        edge_factor = boundary_floor + (1.0 - boundary_floor) * center_weight
        factor = (1.0 - boundary_strength) + boundary_strength * edge_factor
        hybrid[idx] = clamp01(value * factor)

    kept = 0
    for idx, value in enumerate(hybrid):
        if value < hard_threshold:
            hybrid[idx] = 0.0
        else:
            kept += 1

    staged_state.potential = hybrid

    mean_staged = sum(staged_norm) / max(1, len(staged_norm))
    mean_legacy = sum(legacy_norm) / max(1, len(legacy_norm))
    mean_hybrid = sum(hybrid) / max(1, len(hybrid))

    logs: List[Dict[str, float]] = [
        {
            "rank": 1.0,
            "name": "Hybrid Base (staged normalized)",
            "keyword": "Hybrid_Staged_Base",
            "effective_weight": staged_weight,
            "mean_before": 0.0,
            "mean_after": mean_staged,
            "delta": mean_staged,
        },
        {
            "rank": 2.0,
            "name": "Hybrid Base (legacy normalized)",
            "keyword": "Hybrid_Legacy_Base",
            "effective_weight": 1.0 - staged_weight,
            "mean_before": 0.0,
            "mean_after": mean_legacy,
            "delta": mean_legacy,
        },
        {
            "rank": 3.0,
            "name": "Hybrid Blend + Threshold",
            "keyword": "Hybrid_Blend|Hybrid_Threshold",
            "effective_weight": 1.0,
            "mean_before": 0.5 * (mean_staged + mean_legacy),
            "mean_after": mean_hybrid,
            "delta": mean_hybrid - 0.5 * (mean_staged + mean_legacy),
            "kept_voxels": float(kept),
            "kept_ratio": kept / max(1, staged_state.voxel_count),
            "threshold": hard_threshold,
            "boundary_strength": boundary_strength,
            "boundary_power": boundary_power,
            "boundary_floor": boundary_floor,
        },
    ]
    return staged_state, logs


def build_ensemble_state(
    member_states: List[OreState],
    combine_mode: str,
    threshold: float,
) -> Tuple[OreState, List[Dict[str, float]]]:
    if not member_states:
        raise ValueError("member_states cannot be empty")

    base_state = member_states[0]
    voxel_count = base_state.voxel_count
    merged = [0.0] * voxel_count

    if combine_mode == "max":
        for idx in range(voxel_count):
            merged[idx] = max(state.potential[idx] for state in member_states)
    else:
        scale = 1.0 / len(member_states)
        for idx in range(voxel_count):
            merged[idx] = sum(state.potential[idx] for state in member_states) * scale

    before_mean = sum(merged) / max(1, voxel_count)

    kept = 0
    if threshold > 0.0:
        for idx, value in enumerate(merged):
            if value < threshold:
                merged[idx] = 0.0
            else:
                kept += 1
    else:
        kept = sum(1 for value in merged if value > 0.0)

    after_mean = sum(merged) / max(1, voxel_count)
    base_state.potential = merged

    logs: List[Dict[str, float]] = [
        {
            "rank": 1.0,
            "name": "Ensemble Combine",
            "keyword": "Ensemble_Combine",
            "effective_weight": 1.0,
            "mean_before": 0.0,
            "mean_after": before_mean,
            "delta": before_mean,
            "member_count": float(len(member_states)),
            "combine_mode": combine_mode,
        },
        {
            "rank": 2.0,
            "name": "Ensemble Threshold",
            "keyword": "Ensemble_Threshold",
            "effective_weight": 1.0,
            "mean_before": before_mean,
            "mean_after": after_mean,
            "delta": after_mean - before_mean,
            "threshold": threshold,
            "kept_voxels": float(kept),
            "kept_ratio": kept / max(1, voxel_count),
        },
    ]
    return base_state, logs


def center_ore_body_in_grid(
    state: OreState,
    ore_threshold: float = GRADE_THRESHOLDS[0],
) -> Tuple[int, int, int]:
    """平移矿化势分布，使矿体质心靠近网格中心。"""
    x_size, y_size, z_size = state.grid_size

    weighted_indices = []
    total_weight = 0.0
    for idx, potential in enumerate(state.potential):
        weight = max(0.0, potential - ore_threshold)
        if weight <= 0.0:
            continue
        weighted_indices.append((idx, weight))
        total_weight += weight

    # 若阈值筛选后无体素，则回退为使用所有非零矿化势体素。
    if total_weight <= 1e-12:
        weighted_indices = []
        total_weight = 0.0
        for idx, potential in enumerate(state.potential):
            if potential <= 1e-8:
                continue
            weighted_indices.append((idx, potential))
            total_weight += potential

    if total_weight <= 1e-12:
        return (0, 0, 0)

    centroid_x = 0.0
    centroid_y = 0.0
    centroid_z = 0.0

    for idx, weight in weighted_indices:
        x, y, z = state.xyz(idx)
        centroid_x += x * weight
        centroid_y += y * weight
        centroid_z += z * weight

    centroid_x /= total_weight
    centroid_y /= total_weight
    centroid_z /= total_weight

    target_x = (x_size - 1) / 2.0
    target_y = (y_size - 1) / 2.0
    target_z = (z_size - 1) / 2.0

    shift_x = int(round(target_x - centroid_x))
    shift_y = int(round(target_y - centroid_y))
    shift_z = int(round(target_z - centroid_z))

    if shift_x == 0 and shift_y == 0 and shift_z == 0:
        return (0, 0, 0)

    shifted = [0.0] * state.voxel_count
    for idx, value in enumerate(state.potential):
        if value <= 0.0:
            continue
        x, y, z = state.xyz(idx)
        nx = x + shift_x
        ny = y + shift_y
        nz = z + shift_z
        if nx < 0 or ny < 0 or nz < 0 or nx >= x_size or ny >= y_size or nz >= z_size:
            continue
        nidx = state.index(nx, ny, nz)
        shifted[nidx] = value

    state.potential = shifted
    return (shift_x, shift_y, shift_z)


def write_graded_ply(
    file_path: Path,
    state: OreState,
    grades: List[float],
    material_ids: List[int],
    alphas: List[int],
) -> Dict[int, int]:
    x_size, y_size, z_size = state.grid_size
    total_points = x_size * y_size * z_size
    class_counts = {material_id: 0 for material_id in MATERIAL_COLORS}

    with file_path.open("w", encoding="utf-8") as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("comment Virtual ore body generated by VoxelOreGen\n")
        ply_file.write(f"element vertex {total_points}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write("property uchar alpha\n")
        ply_file.write("property uchar material\n")
        ply_file.write("property float grade\n")
        ply_file.write("end_header\n")

        for z in range(z_size):
            for y in range(y_size):
                for x in range(x_size):
                    idx = state.index(x, y, z)
                    grade = grades[idx]
                    material_id = material_ids[idx]
                    alpha = alphas[idx]

                    red, green, blue = MATERIAL_COLORS[material_id]
                    class_counts[material_id] += 1
                    ply_file.write(
                        f"{x} {y} {z} {red} {green} {blue} {alpha} {material_id} {grade:.4f}\n"
                    )

    return class_counts


def write_gan_voxel_npz(
    file_path: Path,
    state: OreState,
    grades: List[float],
) -> Tuple[float, float]:
    """导出单样本体素张量，供 GAN 训练数据构建直接读取。"""
    grade_grid = np.asarray(grades, dtype=np.float32).reshape(state.grid_size)
    potential_grid = np.asarray(state.potential, dtype=np.float32).reshape(state.grid_size)
    tensor_norm = np.clip(grade_grid * 2.0 - 1.0, -1.0, 1.0).astype(np.float32)

    payload: Dict[str, np.ndarray] = {
        "ore_grade": grade_grid,
        "tensor_norm": tensor_norm,
        "raw_potential": potential_grid,
        "grid_size": np.asarray(state.grid_size, dtype=np.int32),
    }

    raw_grade = state.metadata.get("physics_raw_ore_grade")
    if isinstance(raw_grade, list) and len(raw_grade) == state.voxel_count:
        payload["raw_grade"] = np.asarray(raw_grade, dtype=np.float32).reshape(state.grid_size)

    np.savez_compressed(file_path, **payload)
    return float(np.min(grade_grid)), float(np.max(grade_grid))


def build_voxel_grade_model(
    state: OreState,
    rng: random.Random,
    non_ore_alpha: int,
) -> Tuple[List[float], List[int], List[int]]:
    grades = [0.0] * state.voxel_count
    material_ids = [NON_ORE_ID] * state.voxel_count
    alphas = [non_ore_alpha] * state.voxel_count

    for idx, potential in enumerate(state.potential):
        grade = grade_from_potential(potential, rng)
        material_id = grade_to_material_id(grade)
        grades[idx] = grade
        material_ids[idx] = material_id
        alphas[idx] = 255 if material_id != NON_ORE_ID else non_ore_alpha

    return grades, material_ids, alphas


def build_voxel_grade_model_direct(
    state: OreState,
    non_ore_alpha: int,
) -> Tuple[List[float], List[int], List[int]]:
    """将归一化矿化势直接映射为品位类别，不额外加入块金扰动。"""
    grades = [0.0] * state.voxel_count
    material_ids = [NON_ORE_ID] * state.voxel_count
    alphas = [non_ore_alpha] * state.voxel_count

    for idx, potential in enumerate(state.potential):
        grade = clamp01(potential)
        material_id = grade_to_material_id(grade)
        grades[idx] = grade
        material_ids[idx] = material_id
        alphas[idx] = 255 if material_id != NON_ORE_ID else non_ore_alpha

    return grades, material_ids, alphas


def build_voxel_grade_model_physics(
    state: OreState,
    non_ore_alpha: int,
    cutoff_grade: float,
) -> Tuple[List[float], List[int], List[int]]:
    """基于 physics 原始品位构建品位类别，使 PLY/OBJ 使用一致 cutoff 语义。"""
    raw_grade = state.metadata.get("physics_raw_ore_grade")
    if not isinstance(raw_grade, list) or len(raw_grade) != state.voxel_count:
        # 元数据不符合预期时回退到直接映射模式。
        return build_voxel_grade_model_direct(state, non_ore_alpha)

    cleaned_raw = [max(0.0, float(value)) for value in raw_grade]
    max_raw = max(cleaned_raw) if cleaned_raw else 0.0
    ore_span = max(1e-6, max_raw - cutoff_grade)

    grades = [0.0] * state.voxel_count
    material_ids = [NON_ORE_ID] * state.voxel_count
    alphas = [non_ore_alpha] * state.voxel_count

    for idx, raw_value in enumerate(cleaned_raw):
        if raw_value < cutoff_grade:
            continue

        normalized = clamp01((raw_value - cutoff_grade) / ore_span)
        # 即便接近 cutoff，也保持矿体体素在类别图中可见。
        grade = max(GRADE_THRESHOLDS[0], normalized)
        material_id = grade_to_material_id(grade)

        grades[idx] = grade
        material_ids[idx] = material_id
        alphas[idx] = 255

    return grades, material_ids, alphas


def extract_surface_indices(
    state: OreState,
    material_ids: List[int],
) -> List[int]:
    """返回位于矿体/非矿体边界（6 邻域）的矿体体素索引。"""
    surface_indices: List[int] = []

    x_size, y_size, z_size = state.grid_size
    for idx, material_id in enumerate(material_ids):
        if material_id == NON_ORE_ID:
            continue

        x, y, z = state.xyz(idx)
        is_surface = (
            x == 0
            or y == 0
            or z == 0
            or x == x_size - 1
            or y == y_size - 1
            or z == z_size - 1
        )

        if not is_surface:
            for nidx in state.iter_neighbors6(x, y, z):
                if material_ids[nidx] == NON_ORE_ID:
                    is_surface = True
                    break

        if is_surface:
            surface_indices.append(idx)

    return surface_indices


def write_ore_surface_ply(
    file_path: Path,
    state: OreState,
    material_ids: List[int],
    surface_indices: List[int],
) -> Tuple[int, int]:
    """将矿体表面写为三角网格 PLY（顶点 + 面），格式与 3dvae 类似。"""
    # 六个面方向及其在体素局部坐标下的四边形顶点。
    # 每个暴露面都会拆分为 2 个三角形。
    face_definitions = (
        # +X 方向
        (
            (1, 0, 0),
            ((0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5), (0.5, -0.5, 0.5)),
        ),
        # -X 方向
        (
            (-1, 0, 0),
            ((-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, 0.5), (-0.5, 0.5, -0.5)),
        ),
        # +Y 方向
        (
            (0, 1, 0),
            ((-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, -0.5)),
        ),
        # -Y 方向
        (
            (0, -1, 0),
            ((-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (-0.5, -0.5, 0.5)),
        ),
        # +Z 方向
        (
            (0, 0, 1),
            ((-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)),
        ),
        # -Z 方向
        (
            (0, 0, -1),
            ((-0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (0.5, 0.5, -0.5), (0.5, -0.5, -0.5)),
        ),
    )

    vertex_map: Dict[Tuple[float, float, float, int, int, int], int] = {}
    vertices: List[Tuple[float, float, float, int, int, int]] = []
    faces: List[Tuple[int, int, int]] = []

    x_size, y_size, z_size = state.grid_size

    def add_vertex(vx: float, vy: float, vz: float, red: int, green: int, blue: int) -> int:
        key = (vx, vy, vz, red, green, blue)
        existing = vertex_map.get(key)
        if existing is not None:
            return existing
        new_idx = len(vertices)
        vertex_map[key] = new_idx
        vertices.append((vx, vy, vz, red, green, blue))
        return new_idx

    for idx in surface_indices:
        x, y, z = state.xyz(idx)
        material_id = material_ids[idx]
        red, green, blue = MATERIAL_COLORS[material_id]

        for (dx, dy, dz), corners in face_definitions:
            nx = x + dx
            ny = y + dy
            nz = z + dz

            neighbor_is_ore = False
            if 0 <= nx < x_size and 0 <= ny < y_size and 0 <= nz < z_size:
                nidx = state.index(nx, ny, nz)
                neighbor_is_ore = material_ids[nidx] != NON_ORE_ID

            if neighbor_is_ore:
                continue

            quad_indices: List[int] = []
            for ox, oy, oz in corners:
                vx = x + ox
                vy = y + oy
                vz = z + oz
                quad_indices.append(add_vertex(vx, vy, vz, red, green, blue))

            faces.append((quad_indices[0], quad_indices[1], quad_indices[2]))
            faces.append((quad_indices[0], quad_indices[2], quad_indices[3]))

    with file_path.open("w", encoding="utf-8") as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("comment Ore surface mesh extracted from voxel boundary\n")
        ply_file.write(f"element vertex {len(vertices)}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write(f"element face {len(faces)}\n")
        ply_file.write("property list uchar int vertex_index\n")
        ply_file.write("end_header\n")

        for x, y, z, red, green, blue in vertices:
            ply_file.write(f"{x:.4f} {y:.4f} {z:.4f} {red} {green} {blue}\n")

        for i0, i1, i2 in faces:
            ply_file.write(f"3 {i0} {i1} {i2}\n")

    return len(vertices), len(faces)


def save_stage_logs(log_file: Path, stage_logs) -> None:
    payload = {
        "stage_count": len(stage_logs),
        "stages": stage_logs,
    }
    with log_file.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def slugify_stage_name(name: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower()).strip("_")
    return slug or "stage"


def clone_state_with_potential(template: OreState, potential: List[float]) -> OreState:
    return OreState(
        grid_size=template.grid_size,
        potential=potential,
        temperature=template.temperature,
        pressure=template.pressure,
        permeability=template.permeability,
        structure=template.structure,
        fluid_flux=template.fluid_flux,
        reactivity=template.reactivity,
        preservation=template.preservation,
        porosity=template.porosity,
        ph=template.ph,
        eh=template.eh,
        metal_channels=template.metal_channels,
        complex_channels=template.complex_channels,
        metadata=template.metadata,
    )


def export_workflow_stage_visuals(
    output_dir: Path,
    file_index: int,
    template_state: OreState,
    stage_snapshots: List[Dict[str, object]],
    non_ore_alpha: int,
    cutoff_grade: float,
    obj_smooth_iterations: int,
) -> List[Dict[str, float]]:
    reports: List[Dict[str, float]] = []

    sortable: List[Tuple[int, str, List[float]]] = []
    for stage in stage_snapshots:
        rank_raw = stage.get("rank", 0.0)
        name_raw = stage.get("name", "stage")
        potential_raw = stage.get("potential", [])

        if not isinstance(potential_raw, list):
            continue
        if len(potential_raw) != template_state.voxel_count:
            continue

        try:
            rank = int(round(float(rank_raw)))
        except (TypeError, ValueError):
            rank = 0

        name = str(name_raw)
        potential = [clamp01(float(value)) for value in potential_raw]
        sortable.append((rank, name, potential))

    sortable.sort(key=lambda item: item[0])

    for rank, name, potential in sortable:
        stage_state = clone_state_with_potential(template_state, potential)
        grades, material_ids, alphas = build_voxel_grade_model_direct(stage_state, non_ore_alpha)

        stage_slug = slugify_stage_name(name)
        stage_file_stem = f"ore_body_{file_index:03d}_stage_{rank:02d}_{stage_slug}"

        stage_ply_path = output_dir / f"{stage_file_stem}.ply"
        class_counts = write_graded_ply(stage_ply_path, stage_state, grades, material_ids, alphas)

        stage_obj_path = output_dir / f"{stage_file_stem}.obj"
        obj_vertices, obj_faces = export_obj_isosurface(
            stage_state,
            stage_obj_path,
            cutoff_grade=cutoff_grade,
            smooth_iterations=obj_smooth_iterations,
        )

        ore_voxels = class_counts[1] + class_counts[2] + class_counts[3] + class_counts[4]
        reports.append(
            {
                "stage_rank": float(rank),
                "ore_voxels": float(ore_voxels),
                "obj_vertices": float(obj_vertices),
                "obj_faces": float(obj_faces),
            }
        )

    return reports


def resolve_local_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def run_gan_mode(args: argparse.Namespace, base_dir: Path) -> None:
    """通过 main.py 统一入口执行 GAN 训练/生成流程。"""
    from gan_wgangp import generate as gan_generate
    from gan_wgangp import prepare_data as gan_prepare_data
    from gan_wgangp import train as gan_train

    if args.mode == "gan-train":
        dataset_path = resolve_local_path(base_dir, args.gan_dataset_path)
        source_dir = resolve_local_path(base_dir, args.gan_source_dir)

        if not args.gan_skip_prepare:
            prep_args = argparse.Namespace(
                source_dir=str(source_dir),
                output=str(dataset_path),
                pattern=args.gan_pattern,
                tensor_key=args.gan_tensor_key,
                cond_dim=args.gan_cond_dim,
                exclude_step_snapshots=args.gan_exclude_step_snapshots,
                skip_invalid=args.gan_skip_invalid,
                max_files=args.gan_max_files,
            )
            print("[GAN] 开始构建训练数据集（归一化到 [-1,1]）...")
            gan_prepare_data(prep_args)

        if args.gan_data.strip():
            train_data_path = resolve_local_path(base_dir, args.gan_data)
        else:
            train_data_path = dataset_path

        if not train_data_path.exists():
            raise FileNotFoundError(
                f"GAN 训练数据不存在: {train_data_path}。"
                "若使用 outputs 自动构建，请不要开启 --gan-skip-prepare。"
            )

        train_args = argparse.Namespace(
            data=str(train_data_path),
            out_dir=str(resolve_local_path(base_dir, args.gan_runs_dir)),
            auto_normalize=args.gan_auto_normalize,
            cond_dim=args.gan_cond_dim,
            epochs=args.gan_epochs,
            batch_size=args.gan_batch_size,
            num_workers=args.gan_num_workers,
            latent_dim=args.gan_latent_dim,
            lr_g=args.gan_lr_g,
            lr_d=args.gan_lr_d,
            n_critic=args.gan_n_critic,
            lambda_gp=args.gan_lambda_gp,
            log_every=args.gan_log_every,
            sample_every=args.gan_sample_every,
            save_every=args.gan_save_every,
            num_visualize=args.gan_num_visualize,
            resume=args.gan_resume,
            device=args.gan_device,
            seed=args.seed,
        )
        print("[GAN] 开始训练改进版条件 WGAN-GP...")
        gan_train(train_args)
        return

    if args.mode == "gan-generate":
        checkpoint_raw = args.gan_checkpoint.strip()
        if not checkpoint_raw:
            raise ValueError("gan-generate 模式需要提供 --gan-checkpoint。")

        checkpoint_path = resolve_local_path(base_dir, checkpoint_raw)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"GAN checkpoint 不存在: {checkpoint_path}")

        condition_file: str = ""
        if args.gan_condition_file.strip():
            condition_file = str(resolve_local_path(base_dir, args.gan_condition_file))

        generate_args = argparse.Namespace(
            checkpoint=str(checkpoint_path),
            out_dir=str(resolve_local_path(base_dir, args.gan_generated_dir)),
            num_samples=args.gan_num_samples,
            condition_vector=args.gan_condition_vector,
            condition_file=condition_file,
            grade_min=args.gan_grade_min,
            grade_max=args.gan_grade_max,
            cutoff_grade=args.gan_cutoff_grade,
            export_mesh=args.gan_export_mesh,
            device=args.gan_device,
            seed=args.seed,
        )
        print("[GAN] 开始生成新矿体样本...")
        gan_generate(generate_args)
        return

    raise ValueError(f"未知 mode: {args.mode}")


def add_oregen_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--grid-size",
        type=parse_grid_size,
        default=(32, 32, 32),
        help="点云网格尺寸，例如 32*32*32。",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=10000,
        help="要生成的随机矿体文件数量。",
    )
    parser.add_argument(
        "--non-ore-transparency",
        type=parse_transparency,
        default=0.9,
        help="非矿体透明度，范围 [0,1]。0.9 表示 90% 透明。",
    )
    parser.add_argument(
        "--rule-file",
        type=str,
        default="金属矿床成矿规律.txt",
        help="包含主要成矿策略标题的文本文件。",
    )
    parser.add_argument(
        "--save-stage-log",
        action="store_true",
        help="为每个文件保存阶段日志（.stages.json）。",
    )
    parser.add_argument(
        "--export-gan-voxels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="同步导出 GAN 训练体素 npz（ore_tensor_###.npz）。",
    )
    parser.add_argument(
        "--algorithm",
        choices=("workflow", "staged", "legacy", "hybrid", "physics"),
        default="workflow",
        help=(
            "生成算法：workflow（7 阶段体素地质流程）、"
            "staged、legacy、hybrid（staged+legacy 融合）或 physics（10 步运移/沉淀流程）。"
        ),
    )
    parser.add_argument(
        "--staged-style",
        choices=("default", "stockwork", "lens"),
        default="default",
        help="staged 模拟风格：default、stockwork（尖锐分支）或 lens（厚层状透镜体）。",
    )
    parser.add_argument(
        "--ensemble-count",
        type=int,
        default=1,
        help="每个输出先生成若干矿体成员，再合成为一个最终结果。",
    )
    parser.add_argument(
        "--ensemble-mode",
        choices=("mean", "max"),
        default="mean",
        help="集合成员融合方式：mean 或 max。",
    )
    parser.add_argument(
        "--ensemble-threshold",
        type=parse_unit_interval,
        default=0.1,
        help="集合融合后（品位映射前）应用的阈值。",
    )
    parser.add_argument(
        "--hybrid-staged-weight",
        type=parse_unit_interval,
        default=0.3,
        help="hybrid 模式下 staged 矿化势的融合权重，范围 [0,1]。",
    )
    parser.add_argument(
        "--hybrid-threshold",
        type=parse_unit_interval,
        default=0.1,
        help="hybrid 模式下，低于该阈值的值在品位映射前置零。",
    )
    parser.add_argument(
        "--hybrid-smooth-passes",
        type=int,
        default=2,
        help="hybrid 模式下，融合后的局部平滑迭代次数。",
    )
    parser.add_argument(
        "--hybrid-boundary-strength",
        type=parse_unit_interval,
        default=0.6,
        help="hybrid 模式下，边界附近衰减强度，范围 [0,1]。",
    )
    parser.add_argument(
        "--hybrid-boundary-power",
        type=parse_positive_float,
        default=1.8,
        help="hybrid 模式下中心偏置曲线幂次（>0），值越大边缘抑制越强。",
    )
    parser.add_argument(
        "--hybrid-boundary-floor",
        type=parse_unit_interval,
        default=0.05,
        help="hybrid 模式下边缘最小保留因子。0 为强抑制，>0 可保留低品位连续性。",
    )
    parser.add_argument(
        "--center-ore-body",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="将矿体平移到输出网格中心（适用于 legacy/hybrid；staged 使用地质中心策略）。",
    )
    parser.add_argument(
        "--export-ore-surface",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="额外导出仅包含矿体表面的 PLY 文件。",
    )
    parser.add_argument(
        "--ore-surface-ply-mode",
        choices=("isosurface", "voxel"),
        default="isosurface",
        help="表面 PLY 风格：isosurface（平滑，接近 OBJ）或 voxel（体素块状边界四边形）。",
    )
    parser.add_argument(
        "--export-obj-surface",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="导出基于 cutoff 的等值面 OBJ 网格。",
    )
    parser.add_argument(
        "--cutoff-grade",
        type=parse_unit_interval,
        default=0.5,
        help="OBJ 导出等值面提取所用 cutoff 品位。",
    )
    parser.add_argument(
        "--obj-smooth-iterations",
        type=int,
        default=1,
        help="OBJ 等值面网格的拉普拉斯平滑迭代次数。",
    )
    parser.add_argument(
        "--export-stage-visuals",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="workflow 算法下，导出每个阶段的体素 PLY 和阶段 OBJ 等值面。",
    )
    parser.add_argument(
        "--workflow-repeat-style",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "workflow 算法的单布尔开关。"
            "True：随机挑选风格驱动阶段（4/5/6）并随机重复若干次；"
            "False：所有阶段按顺序各执行一次。"
        ),
    )
    parser.add_argument(
        "--show-intermediate-steps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="统一中间步骤开关（完整阶段日志 + 算法相关中间导出）。",
    )
    parser.add_argument(
        "--physics-time-steps",
        type=int,
        default=50,
        help="physics 算法的运移/沉淀迭代步数。",
    )
    parser.add_argument(
        "--physics-temp-threshold",
        type=parse_positive_float,
        default=300.0,
        help="physics 算法的沉淀温度阈值。",
    )
    parser.add_argument(
        "--physics-cutoff-grade",
        type=parse_positive_float,
        default=5.0,
        help="physics 算法中 marching cubes 导出前应用的 cutoff 品位。",
    )
    parser.add_argument(
        "--physics-boundary-layers",
        type=int,
        default=3,
        help="physics 算法中边界锁定厚度（体素层数）。",
    )
    parser.add_argument(
        "--physics-seed-size",
        type=int,
        default=4,
        help="physics 算法中中心种子块边长。",
    )
    parser.add_argument(
        "--physics-shear",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="physics 算法中是否应用可选的成矿后韧性剪切形变。",
    )
    parser.add_argument(
        "--physics-export-snapshots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="physics 算法中是否导出每个时间步的中间体素快照。",
    )
    parser.add_argument(
        "--physics-snapshot-every",
        type=int,
        default=1,
        help="physics 算法中每 N 个时间步保存一帧快照。",
    )


def add_gan_train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--gan-source-dir",
        type=str,
        default="outputs/oregen",
        help="gan-train 模式的数据源目录（默认读取 oregen 结果）。",
    )
    parser.add_argument(
        "--gan-dataset-path",
        type=str,
        default="outputs/gan/gan_dataset.npz",
        help="gan-train 模式自动构建的训练数据集输出路径。",
    )
    parser.add_argument(
        "--gan-pattern",
        type=str,
        default="*.npz",
        help="gan-train 模式下在 source-dir 递归匹配文件模式。",
    )
    parser.add_argument(
        "--gan-tensor-key",
        type=str,
        default="",
        help="可选，强制指定 npz 中体素键名；留空则自动识别。",
    )
    parser.add_argument(
        "--gan-cond-dim",
        type=int,
        default=0,
        help="GAN 条件向量维度；无条件训练可设为 0。",
    )
    parser.add_argument(
        "--gan-exclude-step-snapshots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="构建 GAN 数据时是否跳过 step_*.npz 时间步快照。",
    )
    parser.add_argument(
        "--gan-skip-invalid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="构建 GAN 数据时是否跳过无法规整到 32^3 的文件。",
    )
    parser.add_argument(
        "--gan-max-files",
        type=int,
        default=0,
        help="构建 GAN 数据时最多读取文件数；0 表示不限制。",
    )
    parser.add_argument(
        "--gan-skip-prepare",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="gan-train 模式下跳过数据构建步骤，直接训练。",
    )
    parser.add_argument(
        "--gan-data",
        type=str,
        default="",
        help="gan-train 模式直接指定训练数据(.npz/.npy)；为空时使用 gan-dataset-path。",
    )
    parser.add_argument(
        "--gan-runs-dir",
        type=str,
        default="outputs/gan/runs",
        help="GAN 训练输出目录（checkpoint、日志、采样）。",
    )
    parser.add_argument(
        "--gan-auto-normalize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="训练时若输入不在 [-1,1]，可启用自动归一化。",
    )
    parser.add_argument("--gan-epochs", type=int, default=300, help="GAN 训练轮数。")
    parser.add_argument("--gan-batch-size", type=int, default=16, help="GAN 训练批大小。")
    parser.add_argument("--gan-num-workers", type=int, default=0, help="GAN 训练 DataLoader 进程数。")
    parser.add_argument("--gan-latent-dim", type=int, default=128, help="GAN 潜变量 z 维度。")
    parser.add_argument("--gan-lr-g", type=float, default=1e-4, help="GAN 生成器学习率。")
    parser.add_argument("--gan-lr-d", type=float, default=1e-4, help="GAN 判别器（critic）学习率。")
    parser.add_argument("--gan-n-critic", type=int, default=5, help="WGAN-GP 中每训练一次 G 前训练 D 的步数。")
    parser.add_argument("--gan-lambda-gp", type=float, default=10.0, help="WGAN-GP 梯度惩罚系数。")
    parser.add_argument("--gan-log-every", type=int, default=50, help="GAN 训练日志间隔（step）。")
    parser.add_argument("--gan-sample-every", type=int, default=200, help="GAN 训练中保存采样间隔（step）。")
    parser.add_argument("--gan-save-every", type=int, default=10, help="GAN 训练保存 checkpoint 间隔（epoch）。")
    parser.add_argument("--gan-num-visualize", type=int, default=8, help="GAN 训练中每次固定保存的样本数量。")
    parser.add_argument("--gan-resume", type=str, default="", help="GAN 继续训练时的 checkpoint 路径。")
    parser.add_argument("--gan-device", type=str, default="cuda", help="GAN 训练设备：cuda 或 cpu。")


def add_gan_generate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--gan-checkpoint",
        type=str,
        required=True,
        help="gan-generate 模式所需 checkpoint 路径。",
    )
    parser.add_argument(
        "--gan-generated-dir",
        type=str,
        default="outputs/oregen/gan_preview",
        help="gan-generate 模式输出目录（默认 outputs/oregen/gan_preview）。",
    )
    parser.add_argument(
        "--gan-num-samples",
        type=int,
        default=16,
        help="gan-generate 模式生成样本数。",
    )
    parser.add_argument(
        "--gan-condition-vector",
        type=str,
        default="",
        help="gan-generate 的单条件向量，例如 0,0.3,-0.6。",
    )
    parser.add_argument(
        "--gan-condition-file",
        type=str,
        default="",
        help="gan-generate 条件矩阵 .npy 路径（形状 [N, cond_dim]）。",
    )
    parser.add_argument("--gan-grade-min", type=float, default=0.0, help="GAN 输出反归一化的最小品位。")
    parser.add_argument("--gan-grade-max", type=float, default=1.0, help="GAN 输出反归一化的最大品位。")
    parser.add_argument("--gan-cutoff-grade", type=float, default=0.35, help="GAN 生成体素导出网格时的 cutoff 品位。")
    parser.add_argument(
        "--gan-export-mesh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="gan-generate 是否导出 OBJ/PLY 网格。",
    )
    parser.add_argument("--gan-device", type=str, default="cuda", help="GAN 生成设备：cuda 或 cpu。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="统一主入口：先验成矿生成、GAN 训练、GAN 生成。"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="通用随机种子。若为负数则自动生成。",
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    oregen_parser = subparsers.add_parser(
        "oregen",
        help="先验地质规则生成矿体样本",
    )
    add_oregen_arguments(oregen_parser)

    gan_train_parser = subparsers.add_parser(
        "gan-train",
        help="从 oregen 结果构建数据并训练改进版 WGAN-GP",
    )
    add_gan_train_arguments(gan_train_parser)

    gan_generate_parser = subparsers.add_parser(
        "gan-generate",
        help="使用训练好的 GAN 生成新矿体",
    )
    add_gan_generate_arguments(gan_generate_parser)

    # 开发期快捷开关：可在代码里固定运行模式，无需每次在终端传参。
    # 设为 True 时，main.py 将使用 code_mode_args 作为命令行参数。
    use_code_mode = True
    code_mode_args: List[str] = [
        # 示例 1：先验矿体生成
        # "oregen", "--num-files", "10000", "--algorithm", "workflow",

        # 示例 2：GAN 训练
        "gan-train", "--gan-epochs", "300", "--gan-batch-size", "16",

        # 示例 3：GAN 生成
        # "gan-generate", "--gan-checkpoint", "outputs/gan/runs/checkpoints/wgangp_epoch_0300.pt",
        # "--gan-num-samples", "128",
    ]

    args = parser.parse_args(code_mode_args if use_code_mode else None)

    base_dir = Path(__file__).resolve().parent

    if args.seed < 0:
        run_seed = random.SystemRandom().randrange(0, 2**32)
        args.seed = run_seed
    else:
        run_seed = args.seed

    if args.mode != "oregen":
        print(f"[Run] mode={args.mode}, seed={run_seed}")
        run_gan_mode(args, base_dir)
        return

    if args.num_files <= 0:
        parser.error("--num-files must be greater than 0.")
    if args.ensemble_count <= 0:
        parser.error("--ensemble-count must be greater than 0.")
    if args.hybrid_smooth_passes < 0:
        parser.error("--hybrid-smooth-passes must be >= 0.")
    if args.obj_smooth_iterations < 0:
        parser.error("--obj-smooth-iterations must be >= 0.")
    if args.physics_time_steps <= 0:
        parser.error("--physics-time-steps must be greater than 0.")
    if args.physics_boundary_layers < 1:
        parser.error("--physics-boundary-layers must be >= 1.")
    if args.physics_seed_size < 1:
        parser.error("--physics-seed-size must be >= 1.")
    if args.physics_snapshot_every < 1:
        parser.error("--physics-snapshot-every must be >= 1.")

    if args.show_intermediate_steps:
        args.save_stage_log = True
        if args.algorithm == "physics":
            args.physics_export_snapshots = True
            args.physics_snapshot_every = 1
        if args.algorithm == "workflow":
            args.export_stage_visuals = True

    rng = random.Random(run_seed)

    output_dir = base_dir / "outputs" / "oregen"
    output_dir.mkdir(parents=True, exist_ok=True)

    rule_file = Path(args.rule_file)
    if not rule_file.is_absolute():
        rule_file = base_dir / rule_file

    total_points = args.grid_size[0] * args.grid_size[1] * args.grid_size[2]
    non_ore_alpha = transparency_to_alpha(args.non_ore_transparency)

    def generate_state_once(
        local_rng: random.Random,
        physics_snapshot_dir: Path | None = None,
    ) -> Tuple[OreState, List[Dict[str, float]]]:
        if args.algorithm == "workflow":
            if args.workflow_repeat_style:
                # 单开关行为：随机步骤 + 随机重复次数。
                candidate_stages = (4, 5, 6)
                sampled_stages = [stage_id for stage_id in candidate_stages if local_rng.random() < 0.7]
                if not sampled_stages:
                    sampled_stages = [local_rng.choice(candidate_stages)]

                repeat_stages = tuple(sorted(set(sampled_stages)))
                repeat_min = 2
                repeat_max = 4
                repeat_random = True
                repeat_enabled = True
            else:
                # 顺序基线：各阶段只执行一次。
                repeat_stages = (4, 5, 6)
                repeat_min = 1
                repeat_max = 1
                repeat_random = False
                repeat_enabled = False

            return run_voxel_workflow(
                args.grid_size,
                local_rng,
                repeat_enabled=repeat_enabled,
                repeat_stages=repeat_stages,
                repeat_min=repeat_min,
                repeat_max=repeat_max,
                repeat_random=repeat_random,
            )

        if args.algorithm == "physics":
            return run_physics_voxel_growth(
                grid_size=args.grid_size,
                rng=local_rng,
                time_steps=args.physics_time_steps,
                temperature_threshold=args.physics_temp_threshold,
                cutoff_grade=args.physics_cutoff_grade,
                boundary_layers=args.physics_boundary_layers,
                seed_size=args.physics_seed_size,
                apply_shear=args.physics_shear,
                snapshot_dir=physics_snapshot_dir,
                snapshot_every=args.physics_snapshot_every,
            )

        if args.algorithm == "staged":
            return run_staged_metallogenesis(args.grid_size, local_rng, style=args.staged_style)

        if args.algorithm == "legacy":
            state = generate_seed_state(args.grid_size, local_rng)
            logs = incubate_seed(state, local_rng, str(rule_file))
            return state, logs

        file_seed = local_rng.randrange(0, 2**32)
        staged_rng = random.Random(file_seed)
        legacy_rng = random.Random(file_seed ^ 0x9E3779B9)

        staged_state, _ = run_staged_metallogenesis(args.grid_size, staged_rng, style=args.staged_style)
        legacy_state = generate_seed_state(args.grid_size, legacy_rng)
        _ = incubate_seed(legacy_state, legacy_rng, str(rule_file))

        return build_hybrid_state(
            staged_state=staged_state,
            legacy_state=legacy_state,
            staged_weight=args.hybrid_staged_weight,
            hard_threshold=args.hybrid_threshold,
            smooth_passes=args.hybrid_smooth_passes,
            boundary_strength=args.hybrid_boundary_strength,
            boundary_power=args.hybrid_boundary_power,
            boundary_floor=args.hybrid_boundary_floor,
        )

    for idx in range(1, args.num_files + 1):
        physics_snapshot_root: Path | None = None
        if args.algorithm == "physics" and args.physics_export_snapshots:
            physics_snapshot_root = output_dir / f"physics_snapshots_{idx:03d}"

        if args.ensemble_count == 1:
            state, stage_logs = generate_state_once(rng, physics_snapshot_root)
            if args.algorithm not in ("workflow", "physics"):
                rebalance_potential_after_incubation(state)
        else:
            members: List[OreState] = []
            for _member_idx in range(args.ensemble_count):
                member_seed = rng.randrange(0, 2**32)
                member_rng = random.Random(member_seed)
                member_snapshot_dir: Path | None = None
                if physics_snapshot_root is not None:
                    member_snapshot_dir = physics_snapshot_root / f"member_{_member_idx + 1:02d}"

                member_state, _member_logs = generate_state_once(member_rng, member_snapshot_dir)
                if args.algorithm not in ("workflow", "physics"):
                    rebalance_potential_after_incubation(member_state)
                members.append(member_state)

            state, stage_logs = build_ensemble_state(
                member_states=members,
                combine_mode=args.ensemble_mode,
                threshold=args.ensemble_threshold,
            )

        if args.center_ore_body and args.algorithm not in ("staged", "workflow"):
            shift_x, shift_y, shift_z = center_ore_body_in_grid(state)
        else:
            shift_x, shift_y, shift_z = (0, 0, 0)

        if args.algorithm == "workflow":
            grades, material_ids, alphas = build_voxel_grade_model_direct(state, non_ore_alpha)
        elif args.algorithm == "physics":
            grades, material_ids, alphas = build_voxel_grade_model_physics(
                state,
                non_ore_alpha,
                cutoff_grade=args.physics_cutoff_grade,
            )
        else:
            grades, material_ids, alphas = build_voxel_grade_model(state, rng, non_ore_alpha)

        output_path = output_dir / f"ore_body_{idx:03d}.ply"
        class_counts = write_graded_ply(output_path, state, grades, material_ids, alphas)

        gan_npz_path: Path | None = None
        gan_voxel_min = 0.0
        gan_voxel_max = 0.0
        if args.export_gan_voxels:
            gan_npz_path = output_dir / f"ore_tensor_{idx:03d}.npz"
            gan_voxel_min, gan_voxel_max = write_gan_voxel_npz(gan_npz_path, state, grades)

        surface_count = 0
        surface_faces = 0
        if args.export_ore_surface:
            surface_path = output_dir / f"ore_surface_{idx:03d}.ply"
            if args.ore_surface_ply_mode == "voxel":
                surface_indices = extract_surface_indices(state, material_ids)
                surface_count, surface_faces = write_ore_surface_ply(
                    surface_path,
                    state,
                    material_ids,
                    surface_indices,
                )
            else:
                if args.algorithm == "physics":
                    raw_grade = state.metadata.get("physics_raw_ore_grade")
                    if isinstance(raw_grade, list) and len(raw_grade) == state.voxel_count:
                        raw_grid = np.asarray(raw_grade, dtype=np.float64).reshape(args.grid_size)
                        surface_count, surface_faces = export_ply_from_raw_grade(
                            raw_grid,
                            surface_path,
                            level=args.physics_cutoff_grade,
                        )
                    else:
                        surface_count, surface_faces = export_ply_isosurface(
                            state,
                            surface_path,
                            cutoff_grade=args.cutoff_grade,
                            smooth_iterations=args.obj_smooth_iterations,
                        )
                else:
                    surface_count, surface_faces = export_ply_isosurface(
                        state,
                        surface_path,
                        cutoff_grade=args.cutoff_grade,
                        smooth_iterations=args.obj_smooth_iterations,
                    )

        obj_vertex_count = 0
        obj_face_count = 0
        if args.export_obj_surface:
            obj_path = output_dir / f"ore_surface_{idx:03d}.obj"
            if args.algorithm == "physics":
                raw_grade = state.metadata.get("physics_raw_ore_grade")
                if isinstance(raw_grade, list) and len(raw_grade) == state.voxel_count:
                    raw_grid = np.asarray(raw_grade, dtype=np.float64).reshape(args.grid_size)
                    obj_vertex_count, obj_face_count = export_obj_from_raw_grade(
                        raw_grid,
                        obj_path,
                        level=args.physics_cutoff_grade,
                    )
                else:
                    obj_vertex_count, obj_face_count = export_obj_isosurface(
                        state,
                        obj_path,
                        cutoff_grade=args.cutoff_grade,
                        smooth_iterations=args.obj_smooth_iterations,
                    )
            else:
                obj_vertex_count, obj_face_count = export_obj_isosurface(
                    state,
                    obj_path,
                    cutoff_grade=args.cutoff_grade,
                    smooth_iterations=args.obj_smooth_iterations,
                )

        stage_visual_reports: List[Dict[str, float]] = []
        if args.export_stage_visuals and args.algorithm == "workflow":
            snapshots = state.metadata.get("workflow_stage_snapshots", [])
            if isinstance(snapshots, list):
                stage_visual_reports = export_workflow_stage_visuals(
                    output_dir=output_dir,
                    file_index=idx,
                    template_state=state,
                    stage_snapshots=snapshots,
                    non_ore_alpha=non_ore_alpha,
                    cutoff_grade=args.cutoff_grade,
                    obj_smooth_iterations=args.obj_smooth_iterations,
                )

        if args.save_stage_log:
            log_path = output_dir / f"ore_body_{idx:03d}.stages.json"
            save_stage_logs(log_path, stage_logs)

        ore_count = class_counts[1] + class_counts[2] + class_counts[3] + class_counts[4]
        ore_ratio = (ore_count / total_points) * 100.0

        print(
            f"[{idx}/{args.num_files}] {output_path.name}: ore voxels {ore_count}/{total_points} "
            f"({ore_ratio:.2f}%) | grade classes L/M/H/R = "
            f"{class_counts[1]}/{class_counts[2]}/{class_counts[3]}/{class_counts[4]}"
        )
        if args.ensemble_count > 1:
            print(
                f"  Ensemble: members={args.ensemble_count}, mode={args.ensemble_mode}, "
                f"threshold={args.ensemble_threshold:.3f}"
            )
        print(f"  Center shift (x,y,z): ({shift_x}, {shift_y}, {shift_z})")
        if args.export_ore_surface:
            print(
                f"  Ore surface mesh: vertices={surface_count}, faces={surface_faces} "
                f"-> ore_surface_{idx:03d}.ply (mode={args.ore_surface_ply_mode})"
            )
        if gan_npz_path is not None:
            print(
                f"  GAN voxel npz: value_range=[{gan_voxel_min:.4f}, {gan_voxel_max:.4f}] "
                f"-> {gan_npz_path.name}"
            )
        if args.export_obj_surface:
            if args.algorithm == "physics":
                print(
                    f"  OBJ marching-cubes mesh: vertices={obj_vertex_count}, faces={obj_face_count} "
                    f"-> ore_surface_{idx:03d}.obj (level={args.physics_cutoff_grade:.2f})"
                )
            else:
                print(
                    f"  OBJ isosurface mesh: vertices={obj_vertex_count}, faces={obj_face_count} "
                    f"-> ore_surface_{idx:03d}.obj (cutoff={args.cutoff_grade:.2f})"
                )
        if args.algorithm == "physics" and args.physics_export_snapshots:
            snapshot_count = int(state.metadata.get("physics_snapshot_count", 0))
            snapshot_dir = state.metadata.get("physics_snapshot_dir", "")
            if isinstance(snapshot_dir, str) and snapshot_dir:
                print(
                    f"  Physics snapshots: count={snapshot_count}, every={args.physics_snapshot_every} "
                    f"-> {snapshot_dir}"
                )
            elif physics_snapshot_root is not None:
                print(
                    f"  Physics snapshots exported under: {physics_snapshot_root} "
                    f"(every={args.physics_snapshot_every})"
                )
        if stage_visual_reports:
            print(
                f"  Stage visuals: exported {len(stage_visual_reports)} stages "
                f"-> ore_body_{idx:03d}_stage_*.ply / ore_body_{idx:03d}_stage_*.obj"
            )
            for report in stage_visual_reports:
                print(
                    f"    Stage {int(report['stage_rank'])}: "
                    f"ore_voxels={int(report['ore_voxels'])}, "
                    f"obj_v={int(report['obj_vertices'])}, obj_f={int(report['obj_faces'])}"
                )

        if stage_logs:
            if args.show_intermediate_steps:
                shown = stage_logs
            else:
                top_logs = stage_logs[:3]
                tail_logs = stage_logs[-2:] if len(stage_logs) > 4 else []
                shown = top_logs + [log for log in tail_logs if log not in top_logs]
            for log in shown:
                rank = int(log.get("rank", 0))
                name = log.get("name", "unknown")
                weight = float(log.get("effective_weight", 1.0))
                delta = float(log.get("delta", 0.0))
                keyword = log.get("keyword", "")
                if keyword:
                    print(
                        f"  Stage {rank}: {name} keyword={keyword} "
                        f"weight={weight:.3f} delta={delta:.4f}"
                    )
                else:
                    print(
                        f"  Stage {rank}: {name} weight={weight:.3f} delta={delta:.4f}"
                    )

    x_size, y_size, z_size = args.grid_size
    print(
        f"Generated {args.num_files} file(s) in '{output_dir}'. "
        f"Grid size: {x_size}x{y_size}x{z_size}, seed: {run_seed}, "
        f"algorithm: {args.algorithm}, staged-style: {args.staged_style}, "
        f"non-ore transparency: {args.non_ore_transparency:.2f}, "
        f"cutoff-grade: {args.cutoff_grade:.2f}, physics-cutoff-grade: {args.physics_cutoff_grade:.2f}."
    )


if __name__ == "__main__":
    main()
