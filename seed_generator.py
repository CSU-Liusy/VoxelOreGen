from __future__ import annotations

import random
from typing import Dict, List, Tuple

from ore_state import OreState, clamp01


DIRECTIONS_3D = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (dx == 0 and dy == 0 and dz == 0)
]


def clamp_int(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(max_value, value))


def sample_center_near(
    anchor: Tuple[int, int, int],
    spread: float,
    grid_size: Tuple[int, int, int],
    rng: random.Random,
) -> Tuple[int, int, int]:
    x_size, y_size, z_size = grid_size
    ax, ay, az = anchor

    x = clamp_int(int(round(rng.gauss(ax, spread))), 0, x_size - 1)
    y = clamp_int(int(round(rng.gauss(ay, spread))), 0, y_size - 1)
    z = clamp_int(int(round(rng.gauss(az, spread))), 0, z_size - 1)
    return (x, y, z)


def turn_direction(
    current_direction: Tuple[int, int, int], rng: random.Random
) -> Tuple[int, int, int]:
    if rng.random() < 0.5:
        return rng.choice(DIRECTIONS_3D)

    cx, cy, cz = current_direction
    while True:
        nx = clamp_int(cx + rng.choice((-1, 0, 1)), -1, 1)
        ny = clamp_int(cy + rng.choice((-1, 0, 1)), -1, 1)
        nz = clamp_int(cz + rng.choice((-1, 0, 1)), -1, 1)
        if nx == 0 and ny == 0 and nz == 0:
            continue
        return (nx, ny, nz)


def step_position(
    position: Tuple[int, int, int],
    direction: Tuple[int, int, int],
    grid_size: Tuple[int, int, int],
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    x_size, y_size, z_size = grid_size
    x, y, z = position
    dx, dy, dz = direction

    nx = x + dx
    ny = y + dy
    nz = z + dz

    if nx < 0 or nx >= x_size:
        dx = -dx
        nx = x + dx
    if ny < 0 or ny >= y_size:
        dy = -dy
        ny = y + dy
    if nz < 0 or nz >= z_size:
        dz = -dz
        nz = z + dz

    nx = clamp_int(nx, 0, x_size - 1)
    ny = clamp_int(ny, 0, y_size - 1)
    nz = clamp_int(nz, 0, z_size - 1)

    return (nx, ny, nz), (dx, dy, dz)


def stamp_sphere(
    center: Tuple[int, int, int],
    radius: int,
    strength: float,
    grid_size: Tuple[int, int, int],
    ore_strength: Dict[Tuple[int, int, int], float],
) -> None:
    x_size, y_size, z_size = grid_size
    cx, cy, cz = center

    x_min = max(0, cx - radius)
    x_max = min(x_size - 1, cx + radius)
    y_min = max(0, cy - radius)
    y_max = min(y_size - 1, cy + radius)
    z_min = max(0, cz - radius)
    z_max = min(z_size - 1, cz + radius)

    radius_scale = max(1e-6, radius + 0.4)
    for z in range(z_min, z_max + 1):
        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                dx = x - cx
                dy = y - cy
                dz = z - cz
                distance = (dx * dx + dy * dy + dz * dz) ** 0.5
                radial = 1.0 - (distance / radius_scale)
                if radial <= 0.0:
                    continue

                key = (x, y, z)
                ore_strength[key] = clamp01(ore_strength.get(key, 0.0) + strength * radial)


def trace_vein(
    start: Tuple[int, int, int],
    steps: int,
    radius: int,
    base_strength: float,
    turn_probability: float,
    rng: random.Random,
    grid_size: Tuple[int, int, int],
    ore_strength: Dict[Tuple[int, int, int], float],
) -> None:
    position = start
    direction = rng.choice(DIRECTIONS_3D)

    for step_idx in range(steps):
        taper = 1.0 - (step_idx / max(1, steps - 1))
        local_strength = base_strength * (0.65 + 0.35 * taper) * rng.uniform(0.9, 1.1)
        stamp_sphere(position, radius, local_strength, grid_size, ore_strength)

        if rng.random() < turn_probability:
            direction = turn_direction(direction, rng)
        position, direction = step_position(position, direction, grid_size)


def generate_seed_state(
    grid_size: Tuple[int, int, int],
    rng: random.Random,
) -> OreState:
    x_size, y_size, z_size = grid_size
    voxel_count = x_size * y_size * z_size
    ore_strength: Dict[Tuple[int, int, int], float] = {}

    min_dim = min(grid_size)
    max_dim = max(grid_size)

    main_center = (
        rng.randint(max(0, x_size // 5), max(0, x_size - x_size // 5 - 1)),
        rng.randint(max(0, y_size // 5), max(0, y_size - y_size // 5 - 1)),
        rng.randint(max(0, z_size // 5), max(0, z_size - z_size // 5 - 1)),
    )

    major_radius = max(1, int(round(min_dim * 0.1)))
    major_steps = max(12, int(round(max_dim * rng.uniform(1.0, 1.7))))
    trace_vein(
        start=main_center,
        steps=major_steps,
        radius=major_radius,
        base_strength=0.4,
        turn_probability=0.2,
        rng=rng,
        grid_size=grid_size,
        ore_strength=ore_strength,
    )

    branch_count = rng.randint(1, 3)
    for _ in range(branch_count):
        branch_start = sample_center_near(
            main_center,
            spread=max(1.0, min_dim * 0.15),
            grid_size=grid_size,
            rng=rng,
        )
        trace_vein(
            start=branch_start,
            steps=max(6, int(major_steps * rng.uniform(0.4, 0.8))),
            radius=max(1, major_radius - 1),
            base_strength=rng.uniform(0.22, 0.32),
            turn_probability=0.38,
            rng=rng,
            grid_size=grid_size,
            ore_strength=ore_strength,
        )

    satellite_count = rng.randint(3, 6)
    for _ in range(satellite_count):
        sat_center = sample_center_near(
            main_center,
            spread=max(1.5, min_dim * 0.22),
            grid_size=grid_size,
            rng=rng,
        )
        trace_vein(
            start=sat_center,
            steps=max(4, int(round(max_dim * rng.uniform(0.2, 0.55)))),
            radius=max(1, int(round(major_radius * rng.uniform(0.45, 0.8)))),
            base_strength=rng.uniform(0.14, 0.25),
            turn_probability=0.46,
            rng=rng,
            grid_size=grid_size,
            ore_strength=ore_strength,
        )

    potential = [0.0] * voxel_count
    temperature = [0.0] * voxel_count
    pressure = [0.0] * voxel_count
    permeability = [0.0] * voxel_count
    structure = [0.0] * voxel_count
    fluid_flux = [0.0] * voxel_count
    reactivity = [0.0] * voxel_count
    preservation = [0.0] * voxel_count

    fault_axes: List[Tuple[float, float, float, float]] = []
    for _ in range(rng.randint(2, 4)):
        nx = rng.uniform(-1.0, 1.0)
        ny = rng.uniform(-1.0, 1.0)
        nz = rng.uniform(-0.4, 0.4)
        norm = max(1e-6, (nx * nx + ny * ny + nz * nz) ** 0.5)
        nx /= norm
        ny /= norm
        nz /= norm
        offset = rng.uniform(-0.6, 0.6)
        width = rng.uniform(0.08, 0.18)
        fault_axes.append((nx, ny, nz, offset / max(width, 1e-6)))

    for z in range(z_size):
        depth = z / max(1, z_size - 1)
        for y in range(y_size):
            yn = (2.0 * y / max(1, y_size - 1)) - 1.0
            for x in range(x_size):
                xn = (2.0 * x / max(1, x_size - 1)) - 1.0
                zn = (2.0 * z / max(1, z_size - 1)) - 1.0
                idx = z * (x_size * y_size) + y * x_size + x

                potential[idx] = ore_strength.get((x, y, z), 0.0)

                fault_intensity = 0.0
                for nx, ny, nz, scaled_offset in fault_axes:
                    plane_dist = abs(nx * xn + ny * yn + nz * zn + scaled_offset)
                    local = max(0.0, 1.0 - plane_dist)
                    fault_intensity = max(fault_intensity, local)

                structure[idx] = clamp01(0.25 + 0.75 * fault_intensity)
                permeability[idx] = clamp01(
                    0.2 + 0.55 * structure[idx] + 0.15 * rng.random() + 0.1 * (1.0 - depth)
                )
                temperature[idx] = clamp01(0.35 + 0.55 * depth + 0.1 * rng.random())
                pressure[idx] = clamp01(0.3 + 0.65 * depth + 0.08 * structure[idx])
                fluid_flux[idx] = clamp01(0.2 + 0.6 * structure[idx] * permeability[idx])
                reactivity[idx] = clamp01(0.2 + 0.6 * (1.0 - depth) + 0.2 * rng.random())
                preservation[idx] = clamp01(0.35 + 0.5 * depth + 0.1 * rng.random())

    state = OreState(
        grid_size=grid_size,
        potential=potential,
        temperature=temperature,
        pressure=pressure,
        permeability=permeability,
        structure=structure,
        fluid_flux=fluid_flux,
        reactivity=reactivity,
        preservation=preservation,
        metadata={"main_center": main_center},
    )
    state.clamp_all()
    return state
