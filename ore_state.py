from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass
class OreState:
    """In-memory seed state that gets incubated by staged metallogenic rules."""

    grid_size: Tuple[int, int, int]
    potential: List[float]
    temperature: List[float]
    pressure: List[float]
    permeability: List[float]
    structure: List[float]
    fluid_flux: List[float]
    reactivity: List[float]
    preservation: List[float]
    porosity: List[float] = field(default_factory=list)
    ph: List[float] = field(default_factory=list)
    eh: List[float] = field(default_factory=list)
    metal_channels: Dict[str, List[float]] = field(default_factory=dict)
    complex_channels: Dict[str, List[float]] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def voxel_count(self) -> int:
        x_size, y_size, z_size = self.grid_size
        return x_size * y_size * z_size

    def index(self, x: int, y: int, z: int) -> int:
        x_size, y_size, _ = self.grid_size
        return z * (x_size * y_size) + y * x_size + x

    def xyz(self, idx: int) -> Tuple[int, int, int]:
        x_size, y_size, _ = self.grid_size
        layer = x_size * y_size
        z = idx // layer
        rem = idx % layer
        y = rem // x_size
        x = rem % x_size
        return x, y, z

    def iter_neighbors6(self, x: int, y: int, z: int):
        x_size, y_size, z_size = self.grid_size
        if x > 0:
            yield self.index(x - 1, y, z)
        if x + 1 < x_size:
            yield self.index(x + 1, y, z)
        if y > 0:
            yield self.index(x, y - 1, z)
        if y + 1 < y_size:
            yield self.index(x, y + 1, z)
        if z > 0:
            yield self.index(x, y, z - 1)
        if z + 1 < z_size:
            yield self.index(x, y, z + 1)

    def clamp_all(self) -> None:
        channels = [
            self.potential,
            self.temperature,
            self.pressure,
            self.porosity,
            self.permeability,
            self.ph,
            self.eh,
            self.structure,
            self.fluid_flux,
            self.reactivity,
            self.preservation,
        ]

        for mapping in (self.metal_channels, self.complex_channels):
            for channel in mapping.values():
                channels.append(channel)

        for channel in channels:
            if not channel:
                continue
            for idx, value in enumerate(channel):
                channel[idx] = clamp01(value)

    def ensure_extended_channels(self) -> None:
        """Ensure optional channels are initialized to voxel length when absent."""
        n = self.voxel_count

        if len(self.porosity) != n:
            self.porosity = [0.0] * n
        if len(self.ph) != n:
            self.ph = [0.0] * n
        if len(self.eh) != n:
            self.eh = [0.0] * n

        for mapping in (self.metal_channels, self.complex_channels):
            for key, channel in list(mapping.items()):
                if len(channel) != n:
                    mapping[key] = [0.0] * n
