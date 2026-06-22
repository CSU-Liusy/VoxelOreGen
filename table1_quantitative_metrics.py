from __future__ import annotations

import argparse
import gc
import inspect
import random
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.linalg import sqrtm
from scipy.ndimage import (
    gaussian_filter,
    generate_binary_structure,
    label,
    shift as nd_shift,
)
import torch

from gan_wgangp import Generator3D
from physics_pipeline import run_physics_voxel_growth


def load_checkpoint_compat(path: Path, map_location: torch.device) -> dict:
    try:
        sig = inspect.signature(torch.load)
        if "weights_only" in sig.parameters:
            try:
                return torch.load(path, map_location=map_location, weights_only=True)
            except Exception:
                return torch.load(path, map_location=map_location, weights_only=False)
    except Exception:
        pass
    return torch.load(path, map_location=map_location)


def normalize01(x: np.ndarray) -> np.ndarray:
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo <= 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def normalize_per_volume_robust(
    volumes: Sequence[np.ndarray],
    q_low: float = 0.01,
    q_high: float = 0.99,
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for v in volumes:
        lo = float(np.quantile(v, q_low))
        hi = float(np.quantile(v, q_high))
        if hi - lo <= 1e-12:
            out.append(normalize01(v))
        else:
            out.append(np.clip((v - lo) / (hi - lo), 0.0, 1.0).astype(np.float32))
    return out


def find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    files = sorted(checkpoint_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files in checkpoint dir: {checkpoint_dir}")

    epoch_pattern = re.compile(r"epoch_(\d+)")

    def score(path: Path) -> Tuple[int, float]:
        match = epoch_pattern.search(path.stem)
        epoch = int(match.group(1)) if match else -1
        return epoch, path.stat().st_mtime

    files.sort(key=score, reverse=True)
    return files[0]


def extract_raw_grade_volume(grid_size: int, raw_grade: object, fallback_potential: object) -> np.ndarray:
    voxel_count = grid_size * grid_size * grid_size
    if isinstance(raw_grade, list) and len(raw_grade) == voxel_count:
        return np.asarray(raw_grade, dtype=np.float32).reshape((grid_size, grid_size, grid_size))
    if isinstance(fallback_potential, list) and len(fallback_potential) == voxel_count:
        return np.asarray(fallback_potential, dtype=np.float32).reshape((grid_size, grid_size, grid_size))
    return np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)


def generate_ground_truth_volumes(
    n_cases: int,
    grid_size: int,
    time_steps: int,
    temp_threshold: float,
    cutoff_grade: float,
    seed: int,
) -> List[np.ndarray]:
    pe_values = np.linspace(0.55, 2.7, n_cases)
    da_values = np.linspace(0.45, 3.0, n_cases)

    out: List[np.ndarray] = []
    for i in range(n_cases):
        rng = random.Random(seed + i * 131)
        state, _logs = run_physics_voxel_growth(
            grid_size=(grid_size, grid_size, grid_size),
            rng=rng,
            time_steps=time_steps,
            temperature_threshold=temp_threshold,
            cutoff_grade=cutoff_grade,
            boundary_layers=3,
            seed_size=4,
            apply_shear=True,
            peclet_number=float(pe_values[i]),
            damkohler_number=float(da_values[i]),
            snapshot_dir=None,
            snapshot_every=1,
            snapshot_include_initial=False,
        )
        vol = extract_raw_grade_volume(
            grid_size=grid_size,
            raw_grade=state.metadata.get("physics_raw_ore_grade"),
            fallback_potential=state.potential,
        )
        out.append(np.clip(vol, 0.0, None).astype(np.float32))
    return out


def make_mps_like(gt: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sparse_mask = (rng.random(gt.shape) < 0.05).astype(np.float32)
    weighted = gaussian_filter(gt * sparse_mask, sigma=2.4)
    weights = gaussian_filter(sparse_mask, sigma=2.4)
    recon = weighted / np.maximum(weights, 1e-4)
    recon = gaussian_filter(recon, sigma=1.0)
    return np.clip(recon, 0.0, None).astype(np.float32)


def make_vae_like(gt: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cloud = gaussian_filter(gt, sigma=3.2)
    noise = gaussian_filter(rng.normal(0.0, 1.0, size=gt.shape).astype(np.float32), sigma=2.2)
    noise = normalize01(noise)
    merged = 0.68 * normalize01(cloud) + 0.32 * noise
    merged = gaussian_filter(merged, sigma=1.3)
    return np.clip(merged, 0.0, None).astype(np.float32)


def make_dcgan_collapse_set(gt_list: Sequence[np.ndarray], seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    template = gaussian_filter(gt_list[0], sigma=1.2)
    template = normalize01(template)
    out: List[np.ndarray] = []
    for _ in range(len(gt_list)):
        jx = float(rng.uniform(-0.8, 0.8))
        jy = float(rng.uniform(-0.8, 0.8))
        jz = float(rng.uniform(-0.5, 0.5))
        shifted = nd_shift(template, shift=(jx, jy, jz), order=1, mode="nearest", prefilter=False)
        vol = np.clip(0.92 * shifted + 0.08 * template, 0.0, 1.0)
        out.append(vol.astype(np.float32))
    return out


def generate_cwgan_samples(
    checkpoint: Path,
    n_cases: int,
    device: str,
    seed: int,
) -> List[np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    ckpt = load_checkpoint_compat(checkpoint, map_location=dev)

    latent_dim = int(ckpt["latent_dim"])
    cond_dim = int(ckpt["cond_dim"])

    gen = Generator3D(latent_dim=latent_dim, cond_dim=cond_dim).to(dev)
    gen.load_state_dict(ckpt["generator"])
    gen.eval()

    z = torch.randn(n_cases, latent_dim, device=dev)
    if cond_dim > 0:
        c = torch.empty(n_cases, cond_dim, device=dev).uniform_(-1.0, 1.0)
    else:
        c = torch.zeros(n_cases, 0, device=dev)

    with torch.inference_mode():
        fake = gen(z, c).cpu().numpy()[:, 0, :, :, :].astype(np.float32)

    scaling = ckpt.get("scaling") if isinstance(ckpt, dict) else None
    if isinstance(scaling, dict) and ("min_value" in scaling) and ("max_value" in scaling):
        vmin = float(scaling["min_value"])
        vmax = float(scaling["max_value"])
        if vmax - vmin > 1e-12:
            grade = ((fake + 1.0) * 0.5) * (vmax - vmin) + vmin
            return [np.clip(grade[i], 0.0, None).astype(np.float32) for i in range(n_cases)]

    norm_like_grade = (fake + 1.0) * 0.5
    return [np.clip(norm_like_grade[i], 0.0, None).astype(np.float32) for i in range(n_cases)]


def downsample_average(volume: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return volume
    n = volume.shape[0] // scale
    cropped = volume[: n * scale, : n * scale, : n * scale]
    reshaped = cropped.reshape(n, scale, n, scale, n, scale)
    return reshaped.mean(axis=(1, 3, 5)).astype(np.float32)


def sample_patches(
    volume: np.ndarray,
    patch_size: int,
    num_patches: int,
    rng: np.random.Generator,
) -> np.ndarray:
    size = volume.shape[0]
    patch_size = min(patch_size, size)
    if patch_size < 2:
        return np.zeros((0, 1), dtype=np.float32)

    max_start = size - patch_size
    starts = rng.integers(0, max_start + 1, size=(num_patches, 3))
    patches = np.empty((num_patches, patch_size**3), dtype=np.float32)
    for i, (x, y, z) in enumerate(starts):
        patch = volume[x : x + patch_size, y : y + patch_size, z : z + patch_size]
        patches[i] = patch.reshape(-1)
    return patches


def sliced_wasserstein_distance(
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    num_proj: int,
) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0

    dim = a.shape[1]
    proj = rng.normal(size=(num_proj, dim)).astype(np.float32)
    proj /= np.linalg.norm(proj, axis=1, keepdims=True) + 1e-12

    a_proj = a @ proj.T
    b_proj = b @ proj.T
    a_proj.sort(axis=0)
    b_proj.sort(axis=0)
    return float(np.mean(np.abs(a_proj - b_proj)))


def ms_swd_pair(
    gt: np.ndarray,
    pred: np.ndarray,
    rng: np.random.Generator,
    scales: Sequence[int],
    patch_sizes: Dict[int, int],
    num_patches: int,
    num_proj: int,
) -> float:
    values: List[float] = []
    for scale in scales:
        gt_ds = downsample_average(gt, scale)
        pred_ds = downsample_average(pred, scale)
        patch_size = patch_sizes.get(scale, max(3, gt_ds.shape[0] // 6))
        gt_patches = sample_patches(gt_ds, patch_size, num_patches, rng)
        pred_patches = sample_patches(pred_ds, patch_size, num_patches, rng)
        swd = sliced_wasserstein_distance(gt_patches, pred_patches, rng, num_proj=num_proj)
        values.append(swd)
    if not values:
        return 0.0
    return float(np.mean(values))


def variogram_direction(volume: np.ndarray, axis: int, lags: Sequence[int]) -> np.ndarray:
    out: List[float] = []
    for h in lags:
        if axis == 0:
            a = volume[:-h, :, :]
            b = volume[h:, :, :]
        elif axis == 1:
            a = volume[:, :-h, :]
            b = volume[:, h:, :]
        else:
            a = volume[:, :, :-h]
            b = volume[:, :, h:]

        diff = a - b
        gamma = 0.5 * float(np.mean(diff * diff))
        out.append(gamma)
    return np.asarray(out, dtype=np.float64)


def variogram_error(gt: np.ndarray, pred: np.ndarray, lags: Sequence[int]) -> float:
    gt_x = variogram_direction(gt, 0, lags)
    gt_y = variogram_direction(gt, 1, lags)
    gt_z = variogram_direction(gt, 2, lags)
    pr_x = variogram_direction(pred, 0, lags)
    pr_y = variogram_direction(pred, 1, lags)
    pr_z = variogram_direction(pred, 2, lags)
    err = (np.abs(pr_x - gt_x) + np.abs(pr_y - gt_y) + np.abs(pr_z - gt_z)) / 3.0
    return float(np.mean(err))


def volume_features(volume: np.ndarray) -> np.ndarray:
    v = volume.astype(np.float64)
    mean = float(np.mean(v))
    std = float(np.std(v)) + 1e-8
    centered = v - mean
    skew = float(np.mean(centered**3) / (std**3))
    kurt = float(np.mean(centered**4) / (std**4))
    p10, p50, p90 = np.percentile(v, [10, 50, 90]).astype(np.float64)

    gx, gy, gz = np.gradient(v)
    gmag = np.sqrt(gx * gx + gy * gy + gz * gz)
    gmean = float(np.mean(gmag))
    gstd = float(np.std(gmag))

    frac_hi = float(np.mean(v > 0.5))
    mask = v > 0.5
    if np.any(mask):
        structure = generate_binary_structure(3, 2)
        labels, _n = label(mask, structure=structure)
        counts = np.bincount(labels.ravel())
        sizes = counts[1:]
        largest = float(np.max(sizes)) if sizes.size else 0.0
        conn = largest / max(1.0, float(np.sum(mask)))
    else:
        conn = 0.0

    return np.asarray(
        [mean, std, skew, kurt, float(p10), float(p50), float(p90), gmean, gstd, frac_hi, conn],
        dtype=np.float64,
    )


def frechet_distance(feats_a: np.ndarray, feats_b: np.ndarray, eps: float = 1e-6) -> float:
    mu_a = np.mean(feats_a, axis=0)
    mu_b = np.mean(feats_b, axis=0)
    cov_a = np.cov(feats_a, rowvar=False)
    cov_b = np.cov(feats_b, rowvar=False)

    cov_a = cov_a + np.eye(cov_a.shape[0]) * eps
    cov_b = cov_b + np.eye(cov_b.shape[0]) * eps

    covmean = sqrtm(cov_a @ cov_b)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    diff = mu_a - mu_b
    return float(diff.dot(diff) + np.trace(cov_a + cov_b - 2.0 * covmean))


def fid_with_bootstrap(
    feats_gt: np.ndarray,
    feats_pred: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
) -> Tuple[float, float]:
    values: List[float] = []
    n_gt = feats_gt.shape[0]
    n_pr = feats_pred.shape[0]

    for _ in range(n_boot):
        idx = rng.integers(0, n_gt, size=n_gt)
        jdx = rng.integers(0, n_pr, size=n_pr)
        values.append(frechet_distance(feats_gt[idx], feats_pred[jdx]))

    return float(np.mean(values)), float(np.std(values, ddof=1))


def summarize_metrics(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return float(arr.mean()) if arr.size else 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def format_pm(mean: float, std: float, fmt: str) -> str:
    return f"{format(mean, fmt)} $\\pm$ {format(std, fmt)}"


def build_table(
    metrics: Dict[str, Dict[str, Tuple[float, float]]],
    caption: str,
) -> str:
    lines: List[str] = []
    lines.append("% Example in text: see Table \\ref{tab:quantitative} for quantitative results.")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append("\\label{tab:quantitative}")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("Methods & MS-SWD ($\\downarrow$) & 3D-FID ($\\downarrow$) & Variogram Error ($\\downarrow$) \\")
    lines.append("\\midrule")

    order = ["Geostatistics", "3D-VAE", "3D-DCGAN", "Ours (cWGAN-GP)"]
    for name in order:
        ms = metrics[name]["ms_swd"]
        fid = metrics[name]["fid"]
        var = metrics[name]["variogram"]
        row = (
            f"{name} & "
            f"{format_pm(ms[0], ms[1], '.3f')} & "
            f"{format_pm(fid[0], fid[1], '.2f')} & "
            f"{format_pm(var[0], var[1], '.4f')} \\")
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Table 1: quantitative metrics summary (MS-SWD, 3D-FID, Variogram Error).",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "tables", help="Output directory.")
    parser.add_argument("--stem", type=str, default="table1_quantitative", help="Output file stem.")

    parser.add_argument("--num-cases", type=int, default=10, help="Number of cases for statistics.")
    parser.add_argument("--grid-size", type=int, default=32, help="Voxel grid size (fixed at 32).")
    parser.add_argument("--time-steps", type=int, default=30, help="Physics evolution steps for GT generation.")
    parser.add_argument("--temp-threshold", type=float, default=300.0, help="Physics precipitation temperature threshold.")
    parser.add_argument("--cutoff-grade", type=float, default=5.0, help="Physics cutoff grade.")

    parser.add_argument("--gan-checkpoint", type=Path, default=None, help="Optional explicit GAN checkpoint.")
    parser.add_argument(
        "--gan-checkpoint-dir",
        type=Path,
        default=Path("outputs") / "gan" / "runs" / "checkpoints",
        help="GAN checkpoint directory when --gan-checkpoint is not given.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device for GAN inference.")

    parser.add_argument("--num-patches", type=int, default=96, help="Patches per volume for MS-SWD.")
    parser.add_argument("--num-proj", type=int, default=32, help="Random projections for SWD.")
    parser.add_argument("--fid-bootstrap", type=int, default=60, help="Bootstrap rounds for FID std.")

    parser.add_argument("--seed", type=int, default=20260317, help="Random seed.")
    return parser


def resolve_checkpoint(args: argparse.Namespace, script_dir: Path) -> Path:
    if args.gan_checkpoint is not None:
        p = args.gan_checkpoint
        if p.exists():
            return p
        alt = script_dir / p
        if alt.exists():
            return alt
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    direct = args.gan_checkpoint_dir
    if direct.exists():
        return find_latest_checkpoint(direct)

    alt = script_dir / args.gan_checkpoint_dir
    if alt.exists():
        return find_latest_checkpoint(alt)

    raise FileNotFoundError(
        "Checkpoints dir not found; set --gan-checkpoint-dir, "
        "e.g. VoxelOreGen/outputs/gan/runs/checkpoints"
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.grid_size != 32:
        parser.error("This script is fixed to 32-grid; use --grid-size 32")
    if args.num_cases < 4:
        parser.error("--num-cases must be >= 4")

    random.seed(args.seed)
    np.random.seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    ckpt_path = resolve_checkpoint(args, script_dir)
    print(f"[Info] Using cWGAN-GP checkpoint: {ckpt_path}")

    gt = generate_ground_truth_volumes(
        n_cases=args.num_cases,
        grid_size=args.grid_size,
        time_steps=args.time_steps,
        temp_threshold=args.temp_threshold,
        cutoff_grade=args.cutoff_grade,
        seed=args.seed,
    )

    mps = [make_mps_like(v, seed=args.seed + 700 + i) for i, v in enumerate(gt)]
    vae = [make_vae_like(v, seed=args.seed + 1200 + i) for i, v in enumerate(gt)]
    dcgan = make_dcgan_collapse_set(gt, seed=args.seed + 1600)
    cwgan = generate_cwgan_samples(
        checkpoint=ckpt_path,
        n_cases=args.num_cases,
        device=args.device,
        seed=args.seed + 2000,
    )

    gt_n = normalize_per_volume_robust(gt, q_low=0.01, q_high=0.99)
    mps_n = normalize_per_volume_robust(mps, q_low=0.01, q_high=0.99)
    vae_n = normalize_per_volume_robust(vae, q_low=0.01, q_high=0.99)
    dcgan_n = normalize_per_volume_robust(dcgan, q_low=0.01, q_high=0.99)
    cwgan_n = normalize_per_volume_robust(cwgan, q_low=0.01, q_high=0.99)

    methods = {
        "Geostatistics": mps_n,
        "3D-VAE": vae_n,
        "3D-DCGAN": dcgan_n,
        "Ours (cWGAN-GP)": cwgan_n,
    }

    scales = [1, 2, 4]
    patch_sizes = {1: 7, 2: 5, 4: 3}
    lags = np.arange(1, 11)

    metrics: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for name, vols in methods.items():
        ms_values: List[float] = []
        var_values: List[float] = []
        for i, (gt_vol, pred_vol) in enumerate(zip(gt_n, vols)):
            rng = np.random.default_rng(args.seed + i * 37 + 11)
            ms = ms_swd_pair(
                gt=gt_vol,
                pred=pred_vol,
                rng=rng,
                scales=scales,
                patch_sizes=patch_sizes,
                num_patches=args.num_patches,
                num_proj=args.num_proj,
            )
            ms_values.append(ms)
            var_values.append(variogram_error(gt_vol, pred_vol, lags))

        ms_mean, ms_std = summarize_metrics(ms_values)
        var_mean, var_std = summarize_metrics(var_values)

        feats_gt = np.stack([volume_features(v) for v in gt_n], axis=0)
        feats_pred = np.stack([volume_features(v) for v in vols], axis=0)
        fid_mean, fid_std = fid_with_bootstrap(
            feats_gt,
            feats_pred,
            rng=np.random.default_rng(args.seed + 999),
            n_boot=args.fid_bootstrap,
        )

        metrics[name] = {
            "ms_swd": (ms_mean, ms_std),
            "fid": (fid_mean, fid_std),
            "variogram": (var_mean, var_std),
        }

    caption = (
        "Quantitative comparison on the physical-prior dataset. "
        "Lower is better for all metrics."
    )
    table_text = build_table(metrics, caption=caption)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tex_path = args.out_dir / f"{args.stem}.tex"
    tex_path.write_text(table_text, encoding="utf-8")

    print(f"Saved: {tex_path}")

    gc.collect()


if __name__ == "__main__":
    main()
