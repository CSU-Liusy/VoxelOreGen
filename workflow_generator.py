from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ore_state import OreState, clamp01


LITHOLOGY_DEEP_INTRUSIVE = 0
LITHOLOGY_CARBONATE = 1
ALTERATION_NONE = 0
ALTERATION_SKARN = 1


@dataclass
class WorkflowConfig:
    grid_size: Tuple[int, int, int]
    core_grade_raw: float = 10.0
    ellipsoid_threshold: float = 1.5
    fault_zone_half_width: float = 2.0
    hydraulic_trigger_prob: float = 0.6
    boiling_trigger_prob: float = 0.3
    skarn_trigger_prob: float = 0.8


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


def _percentile(values: Sequence[float], ratio: float) -> float:
    if not values:
        return 0.0
    ratio = max(0.0, min(1.0, ratio))
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * ratio))
    return ordered[idx]


def _safe_box_bounds(size: int) -> Tuple[int, int]:
    low = max(1, int(round(size * 0.25)))
    high = min(size - 2, int(round(size * 0.75)))
    if high <= low:
        low = max(0, size // 3)
        high = min(size - 1, max(low + 1, size - size // 3 - 1))
    return low, high


def _index(x: int, y: int, z: int, grid_size: Tuple[int, int, int]) -> int:
    x_size, y_size, _ = grid_size
    return z * (x_size * y_size) + y * x_size + x


def _xyz(idx: int, grid_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
    x_size, y_size, _ = grid_size
    layer = x_size * y_size
    z = idx // layer
    rem = idx % layer
    y = rem // x_size
    x = rem % x_size
    return x, y, z


def _fade(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _hash_noise3(ix: int, iy: int, iz: int, seed: int) -> float:
    n = ix * 73856093 ^ iy * 19349663 ^ iz * 83492791 ^ seed * 2654435761
    n = (n << 13) ^ n
    raw = (n * (n * n * 15731 + 789221) + 1376312589) & 0x7FFFFFFF
    return 1.0 - raw / 1073741824.0


def _value_noise3_signed(x: float, y: float, z: float, seed: int) -> float:
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
    return _lerp(nxy0, nxy1, tz)


def _normalize3(vx: float, vy: float, vz: float) -> Tuple[float, float, float]:
    norm = max(1e-8, math.sqrt(vx * vx + vy * vy + vz * vz))
    return vx / norm, vy / norm, vz / norm


def _distance_point_to_segment(
    p: Tuple[float, float, float],
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
) -> float:
    px, py, pz = p
    ax, ay, az = a
    bx, by, bz = b

    abx = bx - ax
    aby = by - ay
    abz = bz - az
    apx = px - ax
    apy = py - ay
    apz = pz - az

    ab2 = abx * abx + aby * aby + abz * abz
    if ab2 <= 1e-12:
        dx = px - ax
        dy = py - ay
        dz = pz - az
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    t = max(0.0, min(1.0, (apx * abx + apy * aby + apz * abz) / ab2))
    qx = ax + t * abx
    qy = ay + t * aby
    qz = az + t * abz

    dx = px - qx
    dy = py - qy
    dz = pz - qz
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _rotate_direction(
    direction: Tuple[float, float, float],
    yaw: float,
    pitch: float,
) -> Tuple[float, float, float]:
    dx, dy, dz = _normalize3(*direction)

    # Build a local orthonormal frame around current direction.
    if abs(dz) < 0.95:
        ref = (0.0, 0.0, 1.0)
    else:
        ref = (0.0, 1.0, 0.0)

    rx = dy * ref[2] - dz * ref[1]
    ry = dz * ref[0] - dx * ref[2]
    rz = dx * ref[1] - dy * ref[0]
    rx, ry, rz = _normalize3(rx, ry, rz)

    ux = ry * dz - rz * dy
    uy = rz * dx - rx * dz
    uz = rx * dy - ry * dx
    ux, uy, uz = _normalize3(ux, uy, uz)

    # First rotate around local up-axis (yaw), then tilt with pitch around right-axis.
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    d1x = cos_y * dx + sin_y * ux
    d1y = cos_y * dy + sin_y * uy
    d1z = cos_y * dz + sin_y * uz

    cos_p, sin_p = math.cos(pitch), math.sin(pitch)
    d2x = cos_p * d1x + sin_p * rx
    d2y = cos_p * d1y + sin_p * ry
    d2z = cos_p * d1z + sin_p * rz
    return _normalize3(d2x, d2y, d2z)


def _branch_l_system_segments(
    core: Tuple[int, int, int],
    grid_size: Tuple[int, int, int],
    rng: random.Random,
    iterations: int = 3,
) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    x_size, y_size, z_size = grid_size

    segments: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = []
    branches: List[Tuple[Tuple[float, float, float], Tuple[float, float, float], float]] = [
        ((float(core[0]), float(core[1]), float(core[2])), _normalize3(0.2, 0.15, 1.0), 2.2)
    ]

    for _ in range(max(1, iterations)):
        next_branches: List[Tuple[Tuple[float, float, float], Tuple[float, float, float], float]] = []
        for start, direction, length in branches:
            steps = rng.randint(2, 4)
            curr = start
            curr_dir = direction
            for _step in range(steps):
                seg_len = max(0.8, length * rng.uniform(0.75, 1.1))
                end = (
                    curr[0] + curr_dir[0] * seg_len,
                    curr[1] + curr_dir[1] * seg_len,
                    curr[2] + curr_dir[2] * seg_len,
                )

                ex = max(0.0, min(x_size - 1.0, end[0]))
                ey = max(0.0, min(y_size - 1.0, end[1]))
                ez = max(0.0, min(z_size - 1.0, end[2]))
                end = (ex, ey, ez)

                segments.append((curr, end))
                curr = end

                curr_dir = _rotate_direction(
                    curr_dir,
                    yaw=rng.uniform(-0.55, 0.55),
                    pitch=rng.uniform(-0.33, 0.48),
                )

                if rng.random() < 0.5:
                    child_dir = _rotate_direction(
                        curr_dir,
                        yaw=rng.uniform(-0.95, 0.95),
                        pitch=rng.uniform(-0.52, 0.52),
                    )
                    next_branches.append((curr, child_dir, length * rng.uniform(0.56, 0.76)))

            if rng.random() < 0.85:
                next_branches.append((curr, curr_dir, length * rng.uniform(0.62, 0.85)))

        branches = next_branches
        if not branches:
            break

    return segments


def _sample_scalar(values: Sequence[float], grid_size: Tuple[int, int, int], x: int, y: int, z: int) -> float:
    x_size, y_size, z_size = grid_size
    xi = max(0, min(x_size - 1, x))
    yi = max(0, min(y_size - 1, y))
    zi = max(0, min(z_size - 1, z))
    return values[_index(xi, yi, zi, grid_size)]


def _sample_scalar_trilinear(
    values: Sequence[float],
    grid_size: Tuple[int, int, int],
    x: float,
    y: float,
    z: float,
) -> float:
    x_size, y_size, z_size = grid_size
    x = max(0.0, min(float(x_size - 1), x))
    y = max(0.0, min(float(y_size - 1), y))
    z = max(0.0, min(float(z_size - 1), z))

    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    z0 = int(math.floor(z))
    x1 = min(x_size - 1, x0 + 1)
    y1 = min(y_size - 1, y0 + 1)
    z1 = min(z_size - 1, z0 + 1)

    tx = x - x0
    ty = y - y0
    tz = z - z0

    c000 = values[_index(x0, y0, z0, grid_size)]
    c100 = values[_index(x1, y0, z0, grid_size)]
    c010 = values[_index(x0, y1, z0, grid_size)]
    c110 = values[_index(x1, y1, z0, grid_size)]
    c001 = values[_index(x0, y0, z1, grid_size)]
    c101 = values[_index(x1, y0, z1, grid_size)]
    c011 = values[_index(x0, y1, z1, grid_size)]
    c111 = values[_index(x1, y1, z1, grid_size)]

    nx00 = _lerp(c000, c100, tx)
    nx10 = _lerp(c010, c110, tx)
    nx01 = _lerp(c001, c101, tx)
    nx11 = _lerp(c011, c111, tx)
    nxy0 = _lerp(nx00, nx10, ty)
    nxy1 = _lerp(nx01, nx11, ty)
    return _lerp(nxy0, nxy1, tz)


def _gradient_descending_normal(
    values: Sequence[float],
    grid_size: Tuple[int, int, int],
    x: float,
    y: float,
    z: float,
) -> Tuple[float, float, float]:
    gx = _sample_scalar_trilinear(values, grid_size, x + 1.0, y, z) - _sample_scalar_trilinear(values, grid_size, x - 1.0, y, z)
    gy = _sample_scalar_trilinear(values, grid_size, x, y + 1.0, z) - _sample_scalar_trilinear(values, grid_size, x, y - 1.0, z)
    gz = _sample_scalar_trilinear(values, grid_size, x, y, z + 1.0) - _sample_scalar_trilinear(values, grid_size, x, y, z - 1.0)
    return _normalize3(-gx, -gy, -gz)


def _interpolate_iso_vertex(
    p0: Tuple[float, float, float],
    p1: Tuple[float, float, float],
    v0: float,
    v1: float,
    iso: float,
) -> Tuple[float, float, float]:
    if abs(v1 - v0) < 1e-12:
        return ((p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5, (p0[2] + p1[2]) * 0.5)
    t = (iso - v0) / (v1 - v0)
    t = max(0.0, min(1.0, t))
    return (
        p0[0] + (p1[0] - p0[0]) * t,
        p0[1] + (p1[1] - p0[1]) * t,
        p0[2] + (p1[2] - p0[2]) * t,
    )


def _triangulate_tetra(
    tetra_positions: Sequence[Tuple[float, float, float]],
    tetra_values: Sequence[float],
    iso: float,
) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]:
    inside = [i for i, value in enumerate(tetra_values) if value >= iso]
    outside = [i for i, value in enumerate(tetra_values) if value < iso]

    if len(inside) == 0 or len(inside) == 4:
        return []

    def interp(i: int, j: int) -> Tuple[float, float, float]:
        return _interpolate_iso_vertex(
            tetra_positions[i],
            tetra_positions[j],
            tetra_values[i],
            tetra_values[j],
            iso,
        )

    triangles: List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]] = []

    if len(inside) == 1:
        i = inside[0]
        j0, j1, j2 = outside
        triangles.append((interp(i, j0), interp(i, j1), interp(i, j2)))
        return triangles

    if len(inside) == 3:
        o = outside[0]
        i0, i1, i2 = inside
        triangles.append((interp(o, i0), interp(o, i2), interp(o, i1)))
        return triangles

    # len(inside) == 2 -> quad split into two triangles
    i0, i1 = inside
    o0, o1 = outside
    p0 = interp(i0, o0)
    p1 = interp(i0, o1)
    p2 = interp(i1, o0)
    p3 = interp(i1, o1)

    triangles.append((p0, p1, p2))
    triangles.append((p1, p3, p2))
    return triangles


def _extract_isosurface(
    values: Sequence[float],
    grid_size: Tuple[int, int, int],
    iso: float,
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]]]:
    x_size, y_size, z_size = grid_size
    if x_size < 2 or y_size < 2 or z_size < 2:
        return [], []

    corner_offsets = (
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    )
    tetrahedra = (
        (0, 5, 1, 6),
        (0, 1, 2, 6),
        (0, 2, 3, 6),
        (0, 3, 7, 6),
        (0, 7, 4, 6),
        (0, 4, 5, 6),
    )

    vertices: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    vertex_map: Dict[Tuple[int, int, int], int] = {}

    def add_vertex(point: Tuple[float, float, float]) -> int:
        # Quantized key keeps mesh manifold while remaining stable.
        key = (int(round(point[0] * 10000)), int(round(point[1] * 10000)), int(round(point[2] * 10000)))
        cached = vertex_map.get(key)
        if cached is not None:
            return cached
        idx = len(vertices)
        vertices.append(point)
        vertex_map[key] = idx
        return idx

    for z in range(z_size - 1):
        for y in range(y_size - 1):
            for x in range(x_size - 1):
                cube_pos: List[Tuple[float, float, float]] = []
                cube_val: List[float] = []
                for ox, oy, oz in corner_offsets:
                    px = x + ox
                    py = y + oy
                    pz = z + oz
                    cube_pos.append((float(px), float(py), float(pz)))
                    cube_val.append(values[_index(px, py, pz, grid_size)])

                # Skip homogeneous blocks outside/inside cutoff.
                above = sum(1 for value in cube_val if value >= iso)
                if above == 0 or above == 8:
                    continue

                for t0, t1, t2, t3 in tetrahedra:
                    tpos = [cube_pos[t0], cube_pos[t1], cube_pos[t2], cube_pos[t3]]
                    tval = [cube_val[t0], cube_val[t1], cube_val[t2], cube_val[t3]]
                    triangles = _triangulate_tetra(tpos, tval, iso)
                    for p0, p1, p2 in triangles:
                        i0 = add_vertex(p0)
                        i1 = add_vertex(p1)
                        i2 = add_vertex(p2)
                        if i0 == i1 or i1 == i2 or i0 == i2:
                            continue
                        faces.append((i0, i1, i2))

    return vertices, faces


def _build_vertex_adjacency(vertex_count: int, faces: Sequence[Tuple[int, int, int]]) -> List[set[int]]:
    neighbors: List[set[int]] = [set() for _ in range(vertex_count)]
    for i0, i1, i2 in faces:
        neighbors[i0].add(i1)
        neighbors[i0].add(i2)
        neighbors[i1].add(i0)
        neighbors[i1].add(i2)
        neighbors[i2].add(i0)
        neighbors[i2].add(i1)
    return neighbors


def _laplacian_smooth(
    vertices: List[Tuple[float, float, float]],
    faces: Sequence[Tuple[int, int, int]],
    iterations: int,
    alpha: float = 0.25,
) -> List[Tuple[float, float, float]]:
    if iterations <= 0 or not vertices:
        return vertices

    curr = vertices[:]
    neighbors = _build_vertex_adjacency(len(vertices), faces)

    for _ in range(iterations):
        nxt = curr[:]
        for vidx, nset in enumerate(neighbors):
            if not nset:
                continue
            x, y, z = curr[vidx]
            mx = sum(curr[n][0] for n in nset) / len(nset)
            my = sum(curr[n][1] for n in nset) / len(nset)
            mz = sum(curr[n][2] for n in nset) / len(nset)

            nxt[vidx] = (
                (1.0 - alpha) * x + alpha * mx,
                (1.0 - alpha) * y + alpha * my,
                (1.0 - alpha) * z + alpha * mz,
            )
        curr = nxt

    return curr


def _compute_normals(
    values: Sequence[float],
    grid_size: Tuple[int, int, int],
    vertices: Sequence[Tuple[float, float, float]],
) -> List[Tuple[float, float, float]]:
    normals: List[Tuple[float, float, float]] = []
    for x, y, z in vertices:
        normals.append(_gradient_descending_normal(values, grid_size, x, y, z))
    return normals


def export_obj_isosurface(
    state: OreState,
    file_path: Path,
    cutoff_grade: float = 0.5,
    smooth_iterations: int = 1,
) -> Tuple[int, int]:
    vertices, faces = _extract_isosurface(state.potential, state.grid_size, cutoff_grade)
    if not vertices or not faces:
        file_path.write_text("# Empty mesh: no voxels above cutoff\n", encoding="utf-8")
        return 0, 0

    smoothed = _laplacian_smooth(vertices, faces, iterations=smooth_iterations, alpha=0.22)
    normals = _compute_normals(state.potential, state.grid_size, smoothed)

    with file_path.open("w", encoding="utf-8") as obj_file:
        obj_file.write("# VoxelOreGen isosurface mesh\n")
        obj_file.write(f"# cutoff_grade={cutoff_grade:.4f}\n")

        for x, y, z in smoothed:
            obj_file.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")

        for nx, ny, nz in normals:
            obj_file.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")

        for i0, i1, i2 in faces:
            # OBJ uses 1-based indices.
            a = i0 + 1
            b = i1 + 1
            c = i2 + 1
            obj_file.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")

    return len(smoothed), len(faces)


def export_ply_isosurface(
    state: OreState,
    file_path: Path,
    cutoff_grade: float = 0.5,
    smooth_iterations: int = 1,
    color: Tuple[int, int, int] = (242, 92, 48),
) -> Tuple[int, int]:
    """Export smooth isosurface mesh as PLY so surface style matches OBJ output."""
    vertices, faces = _extract_isosurface(state.potential, state.grid_size, cutoff_grade)
    if not vertices or not faces:
        file_path.write_text("ply\nformat ascii 1.0\ncomment Empty mesh: no voxels above cutoff\n"
                             "element vertex 0\nproperty float x\nproperty float y\nproperty float z\n"
                             "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                             "element face 0\nproperty list uchar int vertex_index\nend_header\n",
                             encoding="utf-8")
        return 0, 0

    smoothed = _laplacian_smooth(vertices, faces, iterations=smooth_iterations, alpha=0.22)
    red, green, blue = color

    with file_path.open("w", encoding="utf-8") as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("comment VoxelOreGen smooth isosurface mesh\n")
        ply_file.write(f"comment cutoff_grade={cutoff_grade:.4f}\n")
        ply_file.write(f"element vertex {len(smoothed)}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write(f"element face {len(faces)}\n")
        ply_file.write("property list uchar int vertex_index\n")
        ply_file.write("end_header\n")

        for x, y, z in smoothed:
            ply_file.write(f"{x:.6f} {y:.6f} {z:.6f} {red} {green} {blue}\n")

        for i0, i1, i2 in faces:
            ply_file.write(f"3 {i0} {i1} {i2}\n")

    return len(smoothed), len(faces)


def run_voxel_workflow(
    grid_size: Tuple[int, int, int],
    rng: random.Random,
    repeat_enabled: bool = False,
    repeat_stages: Sequence[int] = (4, 5, 6),
    repeat_min: int = 1,
    repeat_max: int = 1,
    repeat_random: bool = True,
) -> Tuple[OreState, List[Dict[str, float]]]:
    cfg = WorkflowConfig(grid_size=grid_size)
    x_size, y_size, z_size = grid_size
    voxel_count = x_size * y_size * z_size

    if repeat_min < 1:
        raise ValueError("repeat_min must be >= 1")
    if repeat_max < repeat_min:
        raise ValueError("repeat_max must be >= repeat_min")

    selected_repeat_stages = {int(stage_id) for stage_id in repeat_stages}
    invalid_repeat_stages = sorted(stage_id for stage_id in selected_repeat_stages if stage_id not in (4, 5, 6))
    if invalid_repeat_stages:
        raise ValueError("repeat_stages only supports stage ids 4, 5, 6")

    def _stage_passes(stage_id: int) -> int:
        if not repeat_enabled or stage_id not in selected_repeat_stages:
            return 1
        if repeat_random:
            return rng.randint(repeat_min, repeat_max)
        return repeat_max

    stage4_passes = _stage_passes(4)
    stage5_passes = _stage_passes(5)
    stage6_passes = _stage_passes(6)

    grade_raw = [0.0] * voxel_count
    base_grade = [0.0] * voxel_count
    permeability = [0.0] * voxel_count
    lithology = [LITHOLOGY_DEEP_INTRUSIVE] * voxel_count
    alteration = [ALTERATION_NONE] * voxel_count
    is_fault_zone = [False] * voxel_count
    intrusive_mask = [False] * voxel_count

    stage_logs: List[Dict[str, float]] = []
    stage_snapshots_raw: List[Dict[str, object]] = []

    def log(
        stage_rank: int,
        name: str,
        keyword: str,
        mean_before: float,
        extra: Dict[str, float] | None = None,
    ) -> None:
        mean_after = _mean(grade_raw)
        payload: Dict[str, float] = {
            "rank": float(stage_rank),
            "name": name,
            "keyword": keyword,
            "effective_weight": 1.0,
            "mean_before": mean_before,
            "mean_after": mean_after,
            "delta": mean_after - mean_before,
        }
        if extra:
            payload.update(extra)
        stage_logs.append(payload)

    def capture_stage(stage_rank: int, name: str) -> None:
        stage_snapshots_raw.append(
            {
                "rank": float(stage_rank),
                "name": name,
                "raw_potential": grade_raw[:],
            }
        )

    # Stage 1: initialize voxel space and background lithology/permeability.
    before_stage = _mean(grade_raw)
    split_z = z_size / 2.0
    for z in range(z_size):
        for y in range(y_size):
            for x in range(x_size):
                idx = _index(x, y, z, grid_size)
                if z < split_z:
                    lithology[idx] = LITHOLOGY_DEEP_INTRUSIVE
                    permeability[idx] = 0.12 + 0.08 * rng.random()
                else:
                    lithology[idx] = LITHOLOGY_CARBONATE
                    permeability[idx] = 0.45 + 0.25 * rng.random()

    stage_name = "Voxel Space + Stratigraphic Initialization"
    log(
        1,
        stage_name,
        "Grid_Init|Lithology_Assignment|Permeability_Assignment",
        before_stage,
        {
            "voxel_count": float(voxel_count),
            "deep_intrusive_ratio": sum(1 for v in lithology if v == LITHOLOGY_DEEP_INTRUSIVE) / voxel_count,
            "carbonate_ratio": sum(1 for v in lithology if v == LITHOLOGY_CARBONATE) / voxel_count,
        },
    )
    capture_stage(1, stage_name)

    # Stage 2: random source location within safe central box.
    before_stage = _mean(grade_raw)
    x_low, x_high = _safe_box_bounds(x_size)
    y_low, y_high = _safe_box_bounds(y_size)
    z_low, z_high = _safe_box_bounds(z_size)

    core_x = rng.randint(x_low, x_high)
    core_y = rng.randint(y_low, y_high)
    core_z = rng.randint(z_low, z_high)
    core_idx = _index(core_x, core_y, core_z, grid_size)
    grade_raw[core_idx] = cfg.core_grade_raw

    intrusive_radius = max(2.0, min(grid_size) * 0.11)
    intr2 = intrusive_radius * intrusive_radius
    intrusive_count = 0
    for idx in range(voxel_count):
        x, y, z = _xyz(idx, grid_size)
        dx = x - core_x
        dy = y - core_y
        dz = z - core_z
        if dx * dx + dy * dy + dz * dz <= intr2:
            intrusive_mask[idx] = True
            intrusive_count += 1

    stage_name = "Core Source Random Localization"
    log(
        2,
        stage_name,
        "Safe_Box_Source|Core_Grade_Assignment",
        before_stage,
        {
            "core_x": float(core_x),
            "core_y": float(core_y),
            "core_z": float(core_z),
            "core_grade_raw": cfg.core_grade_raw,
            "intrusive_cells": float(intrusive_count),
        },
    )
    capture_stage(2, stage_name)

    # Stage 3: anisotropic ellipsoid decay field.
    before_stage = _mean(grade_raw)
    radii = [
        max(2.0, x_size * rng.uniform(0.18, 0.34)),
        max(2.0, y_size * rng.uniform(0.16, 0.31)),
        max(2.0, z_size * rng.uniform(0.12, 0.26)),
    ]
    radii[1] = radii[1] * rng.uniform(0.82, 1.12)
    radii[2] = radii[2] * rng.uniform(0.74, 1.08)
    rx, ry, rz = radii

    influenced = 0
    for idx in range(voxel_count):
        x, y, z = _xyz(idx, grid_size)
        nx = (x - core_x) / max(1e-6, rx)
        ny = (y - core_y) / max(1e-6, ry)
        nz = (z - core_z) / max(1e-6, rz)
        nd = math.sqrt(nx * nx + ny * ny + nz * nz)
        if nd > cfg.ellipsoid_threshold:
            continue

        influenced += 1
        base = (1.0 - nd) * cfg.core_grade_raw
        base_grade[idx] = max(0.0, base)

    stage_name = "Anisotropic Ellipsoid Decay"
    log(
        3,
        stage_name,
        "Ellipsoid_Distance|Grade_Decay_Field",
        before_stage,
        {
            "radius_x": rx,
            "radius_y": ry,
            "radius_z": rz,
            "distance_threshold": cfg.ellipsoid_threshold,
            "influenced_voxels": float(influenced),
        },
    )
    capture_stage(3, stage_name)

    # Stage 4: permeability-modulated 3-octave fractal noise (fBm).
    before_stage = _mean(grade_raw)
    stage4_noise_seeds: List[int] = []
    for pass_idx in range(stage4_passes):
        noise_seed = rng.randrange(1, 2**31)
        stage4_noise_seeds.append(noise_seed)
        for idx in range(voxel_count):
            base = grade_raw[idx] if pass_idx > 0 else base_grade[idx]
            if base <= 0.0:
                continue

            x, y, z = _xyz(idx, grid_size)
            amp = 1.0
            freq = 1.0
            total = 0.0
            norm = 0.0

            for octave in range(3):
                sample = _value_noise3_signed(
                    (x / max(1.0, x_size - 1.0)) * freq * 6.0,
                    (y / max(1.0, y_size - 1.0)) * freq * 6.0,
                    (z / max(1.0, z_size - 1.0)) * freq * 6.0,
                    noise_seed + octave * 9973,
                )
                total += amp * sample
                norm += amp
                amp *= 0.5
                freq *= 2.0

            disturbance = 0.0 if norm <= 1e-12 else total / norm
            multiplier = 1.0 + disturbance * permeability[idx]
            grade_raw[idx] = max(0.0, base * multiplier)

    stage_name = "Porous-Media Infiltration + Fractal Heterogeneity"
    log(
        4,
        stage_name,
        "fBm_Noise|Permeability_Modulation|Impregnation_Heterogeneity",
        before_stage,
        {
            "noise_octaves": 3.0,
            "repeat_passes": float(stage4_passes),
            "noise_seed_first": float(stage4_noise_seeds[0]),
            "noise_seed_last": float(stage4_noise_seeds[-1]),
        },
    )
    capture_stage(4, stage_name)

    # Stage 5: fault displacement and fault-zone enhancement.
    before_stage = _mean(grade_raw)
    moved_count_total = 0
    fault_zone_count_total = 0
    last_fault_nx = 0.0
    last_fault_ny = 0.0
    last_fault_nz = 1.0
    last_anchor = (0, 0, 0)
    last_slip = (0, 0, 0)

    for _pass_idx in range(stage5_passes):
        fault_nx = rng.uniform(-1.0, 1.0)
        fault_ny = rng.uniform(-1.0, 1.0)
        fault_nz = rng.uniform(-1.0, 1.0)
        fault_nx, fault_ny, fault_nz = _normalize3(fault_nx, fault_ny, fault_nz)

        anchor = (
            rng.randint(max(1, int(x_size * 0.35)), min(x_size - 2, int(x_size * 0.65))),
            rng.randint(max(1, int(y_size * 0.35)), min(y_size - 2, int(y_size * 0.65))),
            rng.randint(max(1, int(z_size * 0.35)), min(z_size - 2, int(z_size * 0.65))),
        )
        slip = (
            int(round(4 * rng.uniform(0.75, 1.15))),
            int(round(rng.uniform(-1.0, 1.0))),
            int(round(-3 * rng.uniform(0.8, 1.2))),
        )

        new_grade = [0.0] * voxel_count
        new_perm = [0.0] * voxel_count
        new_lithology = lithology[:]
        new_alteration = alteration[:]
        new_fault = [False] * voxel_count
        new_intrusive = [False] * voxel_count

        def place(src_idx: int, dst_idx: int, fault_mark: bool) -> None:
            if grade_raw[src_idx] >= new_grade[dst_idx]:
                new_grade[dst_idx] = grade_raw[src_idx]
                new_perm[dst_idx] = permeability[src_idx]
                new_lithology[dst_idx] = lithology[src_idx]
                new_alteration[dst_idx] = alteration[src_idx]
                new_intrusive[dst_idx] = intrusive_mask[src_idx]
            if fault_mark:
                new_fault[dst_idx] = True

        moved_count = 0
        fault_zone_count = 0
        for idx in range(voxel_count):
            x, y, z = _xyz(idx, grid_size)
            dx = x - anchor[0]
            dy = y - anchor[1]
            dz = z - anchor[2]
            signed = dx * fault_nx + dy * fault_ny + dz * fault_nz

            dst_x, dst_y, dst_z = x, y, z
            if signed > 0.0:
                dst_x = x + slip[0]
                dst_y = y + slip[1]
                dst_z = z + slip[2]
                if 0 <= dst_x < x_size and 0 <= dst_y < y_size and 0 <= dst_z < z_size:
                    moved_count += 1
                else:
                    continue

            dst_idx = _index(dst_x, dst_y, dst_z, grid_size)
            in_fault_zone = abs(signed) < cfg.fault_zone_half_width
            if in_fault_zone:
                fault_zone_count += 1
            place(idx, dst_idx, in_fault_zone)

        for idx in range(voxel_count):
            if new_fault[idx]:
                new_grade[idx] *= 1.3

        grade_raw = new_grade
        permeability = [max(0.0, p) for p in new_perm]
        lithology = new_lithology
        alteration = new_alteration
        is_fault_zone = new_fault
        intrusive_mask = new_intrusive

        moved_count_total += moved_count
        fault_zone_count_total += fault_zone_count
        last_fault_nx = fault_nx
        last_fault_ny = fault_ny
        last_fault_nz = fault_nz
        last_anchor = anchor
        last_slip = slip

    stage_name = "Fault Slip + Breccia Zone Overprint"
    log(
        5,
        stage_name,
        "Fault_Plane|Slip_Vector|Fault_Zone_Amplification",
        before_stage,
        {
            "fault_nx": last_fault_nx,
            "fault_ny": last_fault_ny,
            "fault_nz": last_fault_nz,
            "fault_anchor_x": float(last_anchor[0]),
            "fault_anchor_y": float(last_anchor[1]),
            "fault_anchor_z": float(last_anchor[2]),
            "slip_x": float(last_slip[0]),
            "slip_y": float(last_slip[1]),
            "slip_z": float(last_slip[2]),
            "repeat_passes": float(stage5_passes),
            "moved_voxels": float(moved_count_total),
            "fault_zone_hits": float(fault_zone_count_total),
        },
    )
    capture_stage(5, stage_name)

    # Stage 6: low-probability micro-physical and geochemical mutations.
    before_stage = _mean(grade_raw)
    hydraulic_triggered = 0.0
    hydraulic_boosted = 0.0
    boiling_triggered = 0.0
    boiling_boosted = 0.0
    skarn_triggered = 0.0
    skarn_boosted = 0.0

    for _pass_idx in range(stage6_passes):
        if rng.random() < cfg.hydraulic_trigger_prob:
            hydraulic_triggered += 1.0
            segments = _branch_l_system_segments((core_x, core_y, core_z), grid_size, rng, iterations=3)
            if segments:
                for idx in range(voxel_count):
                    x, y, z = _xyz(idx, grid_size)
                    point = (x + 0.5, y + 0.5, z + 0.5)
                    min_dist = min(_distance_point_to_segment(point, seg[0], seg[1]) for seg in segments)
                    if min_dist < 1.5:
                        grade_raw[idx] = max(grade_raw[idx], cfg.core_grade_raw)
                        hydraulic_boosted += 1.0

        if rng.random() < cfg.boiling_trigger_prob:
            boiling_triggered += 1.0
            z0 = max(0, int(round(0.56 * (z_size - 1))) + rng.randint(-1, 1))
            z1 = min(z_size - 1, z0 + rng.randint(2, 4))
            threshold = 0.1 * cfg.core_grade_raw

            for z in range(z0, z1 + 1):
                for y in range(y_size):
                    for x in range(x_size):
                        idx = _index(x, y, z, grid_size)
                        if grade_raw[idx] > threshold:
                            grade_raw[idx] *= 5.0
                            boiling_boosted += 1.0

        if rng.random() < cfg.skarn_trigger_prob:
            skarn_triggered += 1.0
            intrusive_points: List[Tuple[int, int, int]] = []
            for idx, mark in enumerate(intrusive_mask):
                if mark:
                    intrusive_points.append(_xyz(idx, grid_size))

            if intrusive_points:
                max_d2 = 3.0 * 3.0
                for idx in range(voxel_count):
                    if lithology[idx] != LITHOLOGY_CARBONATE:
                        continue
                    x, y, z = _xyz(idx, grid_size)

                    near_contact = False
                    for ix, iy, iz in intrusive_points:
                        dx = x - ix
                        dy = y - iy
                        dz = z - iz
                        if dx * dx + dy * dy + dz * dz < max_d2:
                            near_contact = True
                            break

                    if near_contact:
                        alteration[idx] = ALTERATION_SKARN
                        grade_raw[idx] *= 2.0
                        skarn_boosted += 1.0

    stage_name = "Probabilistic Micro-Events"
    log(
        6,
        stage_name,
        "Hydraulic_Fracturing|Boiling_Phase_Separation|Skarn_Neutralization",
        before_stage,
        {
            "repeat_passes": float(stage6_passes),
            "hydraulic_triggered": hydraulic_triggered,
            "hydraulic_boosted_voxels": hydraulic_boosted,
            "boiling_triggered": boiling_triggered,
            "boiling_boosted_voxels": boiling_boosted,
            "skarn_triggered": skarn_triggered,
            "skarn_boosted_voxels": skarn_boosted,
        },
    )
    capture_stage(6, stage_name)

    # Normalize into [0,1] potential for downstream rendering/export.
    norm_scale = max(cfg.core_grade_raw, _percentile(grade_raw, 0.995))
    potential = [clamp01(value / max(1e-6, norm_scale)) for value in grade_raw]

    temperature = [0.0] * voxel_count
    pressure = [0.0] * voxel_count
    structure = [0.0] * voxel_count
    fluid_flux = [0.0] * voxel_count
    reactivity = [0.0] * voxel_count
    preservation = [0.0] * voxel_count
    porosity = [0.0] * voxel_count
    ph = [0.0] * voxel_count
    eh = [0.0] * voxel_count

    for idx in range(voxel_count):
        _, _, z = _xyz(idx, grid_size)
        depth = 1.0 - (z / max(1, z_size - 1))
        temperature[idx] = clamp01(0.2 + 0.7 * depth)
        pressure[idx] = clamp01(0.25 + 0.7 * depth)
        structure[idx] = 0.8 if is_fault_zone[idx] else 0.2
        fluid_flux[idx] = clamp01(0.15 + 0.7 * permeability[idx])
        reactivity[idx] = 0.7 if lithology[idx] == LITHOLOGY_CARBONATE else 0.35
        preservation[idx] = clamp01(0.35 + 0.55 * depth)
        porosity[idx] = clamp01(0.08 + 0.4 * permeability[idx])
        ph[idx] = clamp01((5.8 if alteration[idx] == ALTERATION_SKARN else 6.8) / 14.0)
        eh[idx] = clamp01(0.42 + 0.25 * (1.0 - depth))

    stage_snapshots: List[Dict[str, object]] = []
    for stage in stage_snapshots_raw:
        raw_values = stage["raw_potential"]
        if not isinstance(raw_values, list):
            continue
        normalized = [clamp01(float(value) / max(1e-6, norm_scale)) for value in raw_values]
        stage_snapshots.append(
            {
                "rank": float(stage["rank"]),
                "name": str(stage["name"]),
                "potential": normalized,
            }
        )

    stage_snapshots.append(
        {
            "rank": 7.0,
            "name": "Cutoff Isosurface Extraction Ready",
            "potential": potential[:],
        }
    )

    state = OreState(
        grid_size=grid_size,
        potential=potential,
        temperature=temperature,
        pressure=pressure,
        permeability=[clamp01(p) for p in permeability],
        structure=structure,
        fluid_flux=fluid_flux,
        reactivity=reactivity,
        preservation=preservation,
        porosity=porosity,
        ph=ph,
        eh=eh,
        metal_channels={},
        complex_channels={},
        metadata={
            "workflow": True,
            "workflow_core": (core_x, core_y, core_z),
            "workflow_norm_scale": norm_scale,
            "workflow_repeat_enabled": repeat_enabled,
            "workflow_repeat_stages": sorted(selected_repeat_stages),
            "workflow_repeat_min": repeat_min,
            "workflow_repeat_max": repeat_max,
            "workflow_repeat_random": repeat_random,
            "workflow_repeat_passes": {
                "stage4": stage4_passes,
                "stage5": stage5_passes,
                "stage6": stage6_passes,
            },
            "workflow_lithology": lithology,
            "workflow_alteration": alteration,
            "workflow_fault_zone": is_fault_zone,
            "workflow_intrusive": intrusive_mask,
            "workflow_stage_snapshots": stage_snapshots,
        },
    )

    # Stage 7 is represented by downstream cutoff-driven isosurface extraction in OBJ export.
    before_stage = _mean(grade_raw)
    log(
        7,
        "Cutoff Isosurface Extraction Ready",
        "Cutoff_Grade|IsoSurface_Extraction|OBJ_Export",
        before_stage,
        {
            "normalization_scale": norm_scale,
            "recommended_cutoff": 0.5,
        },
    )

    state.clamp_all()
    return state, stage_logs
