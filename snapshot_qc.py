from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - dependency availability is runtime-specific
    imageio = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - dependency availability is runtime-specific
    plt = None


def parse_axis(value: str) -> int:
    value = value.lower().strip()
    mapping = {"x": 0, "y": 1, "z": 2}
    if value not in mapping:
        raise argparse.ArgumentTypeError("--axis must be one of: x, y, z")
    return mapping[value]


def load_manifest(snapshot_dir: Path) -> Dict[str, object]:
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found under {snapshot_dir}")

    with manifest_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("manifest.json payload must be an object")
    return payload


def list_snapshot_files(snapshot_dir: Path, manifest: Dict[str, object]) -> List[Path]:
    files_raw = manifest.get("files", [])
    if not isinstance(files_raw, list):
        raise ValueError("manifest.json field 'files' must be a list")

    files: List[Path] = []
    for name in files_raw:
        if not isinstance(name, str):
            continue
        candidate = snapshot_dir / name
        if candidate.exists() and candidate.suffix.lower() == ".npz":
            files.append(candidate)

    if not files:
        files = sorted(snapshot_dir.glob("step_*.npz"))

    if not files:
        raise FileNotFoundError(f"No step_*.npz files found under {snapshot_dir}")
    return files


def resolve_slice_index(shape: Sequence[int], axis: int, slice_index: int | None) -> int:
    upper = shape[axis] - 1
    if upper < 0:
        raise ValueError("Invalid snapshot shape")

    if slice_index is None:
        return upper // 2

    if slice_index < 0 or slice_index > upper:
        raise ValueError(f"slice_index out of range [0, {upper}]")
    return slice_index


def extract_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return volume[index, :, :]
    if axis == 1:
        return volume[:, index, :]
    return volume[:, :, index]


def normalize_to_u8(frame: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    span = max(1e-12, vmax - vmin)
    norm = np.clip((frame.astype(np.float64) - vmin) / span, 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def compute_global_range(snapshot_files: Sequence[Path], field: str) -> Tuple[float, float]:
    vmin = float("inf")
    vmax = float("-inf")
    for file_path in snapshot_files:
        with np.load(file_path) as data:
            if field not in data:
                raise KeyError(f"{field} missing in {file_path.name}")
            volume = data[field]
            vmin = min(vmin, float(np.min(volume)))
            vmax = max(vmax, float(np.max(volume)))

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return 0.0, 1.0
    if abs(vmax - vmin) < 1e-12:
        return vmin, vmin + 1.0
    return vmin, vmax


def build_rgb_frame(
    field_slice: np.ndarray,
    precip_slice: np.ndarray,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    gray = normalize_to_u8(field_slice, vmin, vmax)
    rgb = np.stack([gray, gray, gray], axis=-1)

    if precip_slice.dtype != np.bool_:
        precip_mask = precip_slice.astype(np.uint8) > 0
    else:
        precip_mask = precip_slice

    if np.any(precip_mask):
        rgb[..., 0] = np.where(precip_mask, 255, rgb[..., 0])
        rgb[..., 1] = np.where(precip_mask, np.minimum(rgb[..., 1], 64), rgb[..., 1])
        rgb[..., 2] = np.where(precip_mask, np.minimum(rgb[..., 2], 64), rgb[..., 2])

    return rgb


def export_frames_and_gif(
    snapshot_files: Sequence[Path],
    output_dir: Path,
    field: str,
    axis: int,
    slice_index: int,
    fps: int,
    export_frames: bool,
    export_gif: bool,
) -> Tuple[int, Path | None]:
    if imageio is None:
        raise RuntimeError("imageio is required for frame/gif export. Please install imageio.")

    vmin, vmax = compute_global_range(snapshot_files, field)
    axis_token = "xyz"[axis]
    frame_dir = output_dir / f"frames_{field}_{axis_token}{slice_index:02d}"
    if export_frames:
        frame_dir.mkdir(parents=True, exist_ok=True)

    gif_frames: List[np.ndarray] = []
    saved_frames = 0

    for file_path in snapshot_files:
        with np.load(file_path) as data:
            field_volume = data[field]
            precip_mask = data.get("precip_mask", np.zeros_like(field_volume, dtype=np.uint8))
            step = int(data.get("step", 0))

            field_slice = extract_slice(field_volume, axis=axis, index=slice_index)
            precip_slice = extract_slice(precip_mask, axis=axis, index=slice_index)
            rgb = build_rgb_frame(field_slice, precip_slice, vmin=vmin, vmax=vmax)

        if export_frames:
            out_file = frame_dir / f"frame_{step:04d}.png"
            imageio.imwrite(out_file, rgb)
            saved_frames += 1
        if export_gif:
            gif_frames.append(rgb)

    gif_path: Path | None = None
    if export_gif and gif_frames:
        gif_path = output_dir / f"{field}_{axis_token}{slice_index:02d}.gif"
        imageio.mimsave(gif_path, gif_frames, duration=1.0 / max(1, fps), loop=0)

    return saved_frames, gif_path


def load_time_series(snapshot_files: Sequence[Path]) -> List[Dict[str, float]]:
    series: List[Dict[str, float]] = []
    cumulative_precip = 0.0

    for file_path in snapshot_files:
        with np.load(file_path) as data:
            step = int(data.get("step", 0))
            ore = data["ore_grade"].astype(np.float64)
            temp = data["temperature"].astype(np.float64)
            fluid = data["fluid_metal"].astype(np.float64)
            precip_mask = data.get("precip_mask", np.zeros_like(ore, dtype=np.uint8))
            precip_amount = float(data.get("precip_amount", 0.0))

        cx = ore.shape[0] // 2
        cy = ore.shape[1] // 2
        cz = ore.shape[2] // 2
        center_temp = float(temp[cx, cy, cz])

        cumulative_precip += precip_amount
        series.append(
            {
                "step": float(step),
                "precip_amount": precip_amount,
                "cumulative_precip": cumulative_precip,
                "ore_total": float(np.sum(ore)),
                "center_temperature": center_temp,
                "fluid_total": float(np.sum(fluid)),
                "active_precip_voxels": float(np.sum(precip_mask > 0)),
            }
        )

    series.sort(key=lambda item: item["step"])
    return series


def write_time_series_csv(series: Sequence[Dict[str, float]], output_csv: Path) -> None:
    fieldnames = [
        "step",
        "precip_amount",
        "cumulative_precip",
        "ore_total",
        "center_temperature",
        "fluid_total",
        "active_precip_voxels",
    ]

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in series:
            writer.writerow({name: row.get(name, 0.0) for name in fieldnames})


def write_time_series_json(series: Sequence[Dict[str, float]], output_json: Path) -> None:
    payload = {
        "steps": len(series),
        "series": series,
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_curve_png(series: Sequence[Dict[str, float]], output_png: Path) -> bool:
    if plt is None:
        return False

    steps = [row["step"] for row in series]
    precip = [row["precip_amount"] for row in series]
    cumulative = [row["cumulative_precip"] for row in series]
    center_temp = [row["center_temperature"] for row in series]

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    axes[0].plot(steps, precip, color="#D55E00", linewidth=1.8)
    axes[0].set_ylabel("Precip/step")
    axes[0].grid(alpha=0.25)

    axes[1].plot(steps, cumulative, color="#009E73", linewidth=1.8)
    axes[1].set_ylabel("Cum. ore")
    axes[1].grid(alpha=0.25)

    axes[2].plot(steps, center_temp, color="#0072B2", linewidth=1.8)
    axes[2].set_ylabel("Center temp")
    axes[2].set_xlabel("Step")
    axes[2].grid(alpha=0.25)

    fig.suptitle("Physics Snapshot Time-Series QC")
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert VoxelOreGen physics npz snapshots to slice frames/GIF and export time-series QC stats."
    )
    parser.add_argument(
        "--snapshot-dir",
        type=str,
        required=True,
        help="Directory containing manifest.json and step_*.npz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory for frames/GIF/stats. Defaults to <snapshot-dir>/qc.",
    )
    parser.add_argument(
        "--field",
        choices=("ore_grade", "fluid_metal", "temperature", "permeability"),
        default="ore_grade",
        help="Field used for slice frame/GIF generation.",
    )
    parser.add_argument(
        "--axis",
        type=parse_axis,
        default=2,
        help="Slice axis: x, y, or z.",
    )
    parser.add_argument(
        "--slice-index",
        type=int,
        default=-1,
        help="Slice index on selected axis; -1 means center slice.",
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=6,
        help="FPS for GIF export.",
    )
    parser.add_argument(
        "--export-gif",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to export animated GIF.",
    )
    parser.add_argument(
        "--export-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to export per-step PNG frames.",
    )
    parser.add_argument(
        "--export-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to export time-series CSV/JSON and curve PNG.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot_dir = Path(args.snapshot_dir).resolve()
    if not snapshot_dir.exists():
        raise FileNotFoundError(f"Snapshot directory does not exist: {snapshot_dir}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else (snapshot_dir / "qc")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(snapshot_dir)
    snapshot_files = list_snapshot_files(snapshot_dir, manifest)

    with np.load(snapshot_files[0]) as data0:
        if args.field not in data0:
            raise KeyError(f"Field '{args.field}' not found in {snapshot_files[0].name}")
        shape = data0[args.field].shape

    axis = int(args.axis)
    if axis < 0 or axis > 2:
        raise ValueError("axis must be 0, 1, or 2")

    slice_index_input = None if args.slice_index < 0 else args.slice_index
    slice_index = resolve_slice_index(shape, axis=axis, slice_index=slice_index_input)

    if args.export_frames or args.export_gif:
        frame_count, gif_path = export_frames_and_gif(
            snapshot_files=snapshot_files,
            output_dir=output_dir,
            field=args.field,
            axis=axis,
            slice_index=slice_index,
            fps=max(1, args.gif_fps),
            export_frames=bool(args.export_frames),
            export_gif=bool(args.export_gif),
        )
        axis_token = "xyz"[axis]
        if args.export_frames:
            print(
                f"Frames exported: {frame_count} -> {output_dir / ('frames_' + args.field + '_' + axis_token + f'{slice_index:02d}') }"
            )
        if gif_path is not None:
            print(f"GIF exported: {gif_path}")

    if args.export_stats:
        series = load_time_series(snapshot_files)
        csv_path = output_dir / "time_series.csv"
        json_path = output_dir / "time_series.json"
        png_path = output_dir / "time_series_curves.png"

        write_time_series_csv(series, csv_path)
        write_time_series_json(series, json_path)
        curve_ok = write_curve_png(series, png_path)

        print(f"Time-series CSV: {csv_path}")
        print(f"Time-series JSON: {json_path}")
        if curve_ok:
            print(f"Time-series curves PNG: {png_path}")
        else:
            print("Matplotlib unavailable, skipped time-series curves PNG.")


if __name__ == "__main__":
    main()
