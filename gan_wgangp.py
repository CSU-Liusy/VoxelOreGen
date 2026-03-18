from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from skimage.measure import marching_cubes
except Exception:
    marching_cubes = None


# ----------------------------
# 数据加载
# ----------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TensorScaling:
    min_value: float
    max_value: float


class VoxelTensorDataset(Dataset):
    """加载 3D 体素张量及条件向量。

    支持输入：
    1) 单个 .npz 文件，至少包含体素张量，推荐键名：tensors / volumes / ore_grade
       可选条件键名：conditions / cond / c
    2) 单个 .npy 文件，仅体素张量（条件将自动补零）

    体素张量期望维度：
    - [N, 32, 32, 32] 或 [N, 1, 32, 32, 32]
    """

    def __init__(
        self,
        data_path: Path,
        cond_dim: int = -1,
        auto_normalize: bool = False,
        range_epsilon: float = 1e-8,
    ) -> None:
        self.data_path = data_path
        self.range_epsilon = range_epsilon

        tensors, conditions = self._load_arrays(data_path)

        if tensors.ndim == 4:
            tensors = tensors[:, None, :, :, :]
        if tensors.ndim != 5:
            raise ValueError(f"体素数据维度错误，期望 4D/5D，实际为 {tensors.shape}")

        if tensors.shape[1] != 1:
            raise ValueError(f"当前脚本仅支持单通道体素，实际通道数为 {tensors.shape[1]}")

        if tuple(tensors.shape[-3:]) != (32, 32, 32):
            raise ValueError(f"当前脚本固定 32x32x32，实际为 {tuple(tensors.shape[-3:])}")

        self.scaling: Optional[TensorScaling] = None
        t_min = float(np.min(tensors))
        t_max = float(np.max(tensors))

        if auto_normalize:
            tensors, self.scaling = self._normalize_to_minus_one_one(tensors)
        else:
            if t_min < -1.001 or t_max > 1.001:
                raise ValueError(
                    f"检测到体素值范围为 [{t_min:.4f}, {t_max:.4f}]，超出 [-1, 1]。"
                    "请使用 --auto-normalize 或先完成离线归一化。"
                )

        num_samples = tensors.shape[0]

        if conditions is None:
            inferred_cond_dim = 0
            if cond_dim > 0:
                conditions = np.zeros((num_samples, cond_dim), dtype=np.float32)
                inferred_cond_dim = cond_dim
            else:
                conditions = np.zeros((num_samples, 0), dtype=np.float32)
        else:
            if conditions.ndim == 1:
                conditions = conditions[:, None]
            if conditions.shape[0] != num_samples:
                raise ValueError(
                    f"样本数不匹配：tensors={num_samples}, conditions={conditions.shape[0]}"
                )

            inferred_cond_dim = int(conditions.shape[1])
            if cond_dim >= 0 and cond_dim != inferred_cond_dim:
                raise ValueError(
                    f"条件维度不匹配：--cond-dim={cond_dim}, 数据中为 {inferred_cond_dim}"
                )

        self.tensors = tensors.astype(np.float32)
        self.conditions = conditions.astype(np.float32)
        self.cond_dim = int(self.conditions.shape[1])

    def _load_arrays(self, path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not path.exists():
            raise FileNotFoundError(f"数据路径不存在：{path}")

        if path.suffix.lower() == ".npy":
            arr = np.load(path)
            return np.asarray(arr, dtype=np.float32), None

        if path.suffix.lower() == ".npz":
            with np.load(path) as data:
                tensors_key = self._find_key(data.files, ["tensors", "volumes", "ore_grade", "data", "x"])
                if tensors_key is None:
                    raise KeyError(
                        f"无法在 {path.name} 中找到体素键名，已尝试: tensors/volumes/ore_grade/data/x"
                    )

                cond_key = self._find_key(data.files, ["conditions", "cond", "c", "labels", "y"])

                tensors = np.asarray(data[tensors_key], dtype=np.float32)
                conditions = np.asarray(data[cond_key], dtype=np.float32) if cond_key else None
                return tensors, conditions

        raise ValueError(f"仅支持 .npy 或 .npz，当前为: {path.suffix}")

    @staticmethod
    def _find_key(candidates: Sequence[str], preferred: Sequence[str]) -> Optional[str]:
        cset = {c.lower(): c for c in candidates}
        for name in preferred:
            if name in cset:
                return cset[name]
        return None

    def _normalize_to_minus_one_one(self, tensors: np.ndarray) -> Tuple[np.ndarray, TensorScaling]:
        vmin = float(np.min(tensors))
        vmax = float(np.max(tensors))
        if abs(vmax - vmin) < self.range_epsilon:
            raise ValueError("体素张量全常数，无法归一化到 [-1, 1]")

        normed = 2.0 * (tensors - vmin) / (vmax - vmin) - 1.0
        return normed.astype(np.float32), TensorScaling(min_value=vmin, max_value=vmax)

    def __len__(self) -> int:
        return self.tensors.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.tensors[idx])
        c = torch.from_numpy(self.conditions[idx])
        return x, c


def _coerce_to_nx32(arr: np.ndarray, source: str) -> np.ndarray:
    """把输入数组规整为 [N, 32, 32, 32]。"""
    if arr.ndim == 3 and tuple(arr.shape) == (32, 32, 32):
        return arr[None, :, :, :]

    if arr.ndim == 4 and tuple(arr.shape[-3:]) == (32, 32, 32):
        return arr

    if arr.ndim == 5 and arr.shape[1] == 1 and tuple(arr.shape[-3:]) == (32, 32, 32):
        return arr[:, 0, :, :, :]

    raise ValueError(f"无法把 {source} 规整为 [N,32,32,32]，当前 shape={arr.shape}")


def _load_voxel_samples_from_file(file_path: Path, tensor_key: str = "") -> np.ndarray:
    """从单个 npz/npy 中读取矿体体素，返回 [N, 32, 32, 32]。"""
    suffix = file_path.suffix.lower()
    if suffix == ".npy":
        arr = np.asarray(np.load(file_path), dtype=np.float32)
        return _coerce_to_nx32(arr, source=str(file_path))

    if suffix == ".npz":
        with np.load(file_path) as data:
            if tensor_key:
                if tensor_key not in data:
                    raise KeyError(f"{file_path.name} 中不存在键 {tensor_key}")
                arr = np.asarray(data[tensor_key], dtype=np.float32)
                return _coerce_to_nx32(arr, source=f"{file_path.name}:{tensor_key}")

            candidate = VoxelTensorDataset._find_key(
                data.files,
                ["ore_grade", "tensor_grade", "tensor_norm", "tensors", "volumes", "raw_potential", "data", "x"],
            )
            if candidate is None:
                raise KeyError(
                    f"{file_path.name} 无可用体素键，已尝试 ore_grade/tensor_grade/tensor_norm/tensors/volumes/raw_potential/data/x"
                )

            arr = np.asarray(data[candidate], dtype=np.float32)
            return _coerce_to_nx32(arr, source=f"{file_path.name}:{candidate}")

    raise ValueError(f"不支持的文件类型: {file_path.suffix}")


def prepare_data(args: argparse.Namespace) -> None:
    """把 outputs 中分散矿体样本汇总并归一化到 [-1,1]。"""
    source_dir = Path(args.source_dir)
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"source_dir 不存在或不是目录: {source_dir}")

    files = sorted(source_dir.rglob(args.pattern))
    files = [p for p in files if p.is_file() and p.suffix.lower() in {".npz", ".npy"}]

    if args.exclude_step_snapshots:
        files = [p for p in files if not p.name.lower().startswith("step_")]

    if args.max_files > 0:
        files = files[: args.max_files]

    if not files:
        raise FileNotFoundError(
            f"在 {source_dir} 下未找到匹配文件，pattern={args.pattern}"
        )

    samples: List[np.ndarray] = []
    sample_index: List[Dict[str, object]] = []
    skipped = 0

    for file_path in files:
        try:
            arr = _load_voxel_samples_from_file(file_path, tensor_key=args.tensor_key)
        except Exception as exc:
            if args.skip_invalid:
                skipped += 1
                print(f"[Skip] {file_path.name}: {exc}")
                continue
            raise

        rel = str(file_path.relative_to(source_dir))
        for i in range(arr.shape[0]):
            samples.append(arr[i])
            sample_index.append(
                {
                    "source": rel,
                    "local_index": int(i),
                }
            )

    if not samples:
        raise RuntimeError("没有可用样本，请检查 pattern/tensor_key 或关闭 exclude_step_snapshots")

    stacked = np.stack(samples, axis=0).astype(np.float32)
    raw_min = float(np.min(stacked))
    raw_max = float(np.max(stacked))
    if abs(raw_max - raw_min) < 1e-8:
        raise ValueError("样本值全常数，无法归一化")

    normalized = (2.0 * (stacked - raw_min) / (raw_max - raw_min) - 1.0).astype(np.float32)
    tensors = normalized[:, None, :, :, :]

    cond_dim = max(0, int(args.cond_dim))
    conditions = np.zeros((tensors.shape[0], cond_dim), dtype=np.float32)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output,
        tensors=tensors,
        conditions=conditions,
        raw_min=np.float32(raw_min),
        raw_max=np.float32(raw_max),
    )

    manifest = {
        "source_dir": str(source_dir),
        "pattern": args.pattern,
        "exclude_step_snapshots": bool(args.exclude_step_snapshots),
        "tensor_key": args.tensor_key,
        "num_files_scanned": len(files),
        "num_files_skipped": skipped,
        "num_samples": int(tensors.shape[0]),
        "shape": [int(v) for v in tensors.shape],
        "raw_min": raw_min,
        "raw_max": raw_max,
        "cond_dim": cond_dim,
        "samples": sample_index,
    }

    manifest_path = output.with_name(output.stem + "_manifest.json")
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(
        f"数据集构建完成: {output}\n"
        f"样本数={tensors.shape[0]}, 原始范围=[{raw_min:.6f}, {raw_max:.6f}], 已归一化到[-1,1]"
    )


# ----------------------------
# 网络结构
# ----------------------------


class Generator3D(nn.Module):
    def __init__(self, latent_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        in_dim = latent_dim + cond_dim
        self.fc = nn.Linear(in_dim, 512 * 4 * 4 * 4)

        self.net = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),  # 8
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),  # 16
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),   # 32
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if self.cond_dim > 0:
            if cond.ndim != 2 or cond.shape[1] != self.cond_dim:
                raise ValueError(f"Generator 条件维度错误，期望 [B, {self.cond_dim}]，实际 {tuple(cond.shape)}")
            h = torch.cat([z, cond], dim=1)
        else:
            h = z

        h = self.fc(h)
        h = h.view(h.shape[0], 512, 4, 4, 4)
        return self.net(h)


class Critic3D(nn.Module):
    def __init__(self, cond_dim: int) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        in_ch = 1 + cond_dim

        self.net = nn.Sequential(
            nn.Conv3d(in_ch, 64, kernel_size=4, stride=2, padding=1),   # 16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),     # 8
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),    # 4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1),    # 2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(512, 1, kernel_size=2, stride=1, padding=0),      # 1
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != 1:
            raise ValueError(f"Critic 输入体素维度错误，期望 [B,1,32,32,32]，实际 {tuple(x.shape)}")

        if self.cond_dim > 0:
            if cond.ndim != 2 or cond.shape[1] != self.cond_dim:
                raise ValueError(f"Critic 条件维度错误，期望 [B, {self.cond_dim}]，实际 {tuple(cond.shape)}")
            b = x.shape[0]
            cond_map = cond.view(b, self.cond_dim, 1, 1, 1).expand(-1, -1, 32, 32, 32)
            h = torch.cat([x, cond_map], dim=1)
        else:
            h = x

        out = self.net(h)
        return out.view(out.shape[0], 1)


# ----------------------------
# WGAN-GP 训练逻辑
# ----------------------------


def gradient_penalty(
    critic: Critic3D,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    cond: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, 1, device=device)
    interpolates = alpha * real_samples + (1.0 - alpha) * fake_samples
    interpolates.requires_grad_(True)

    critic_scores = critic(interpolates, cond)
    grad_outputs = torch.ones_like(critic_scores, device=device)

    gradients = autograd.grad(
        outputs=critic_scores,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()
    return gp


def save_checkpoint(
    path: Path,
    epoch: int,
    global_step: int,
    generator: Generator3D,
    critic: Critic3D,
    g_opt: torch.optim.Optimizer,
    d_opt: torch.optim.Optimizer,
    latent_dim: int,
    cond_dim: int,
    args_dict: Dict[str, object],
    scaling: Optional[TensorScaling],
) -> None:
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "latent_dim": latent_dim,
        "cond_dim": cond_dim,
        "generator": generator.state_dict(),
        "critic": critic.state_dict(),
        "g_opt": g_opt.state_dict(),
        "d_opt": d_opt.state_dict(),
        "args": args_dict,
        "scaling": {
            "min_value": scaling.min_value,
            "max_value": scaling.max_value,
        } if scaling else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    dataset = VoxelTensorDataset(
        data_path=Path(args.data),
        cond_dim=args.cond_dim,
        auto_normalize=args.auto_normalize,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    latent_dim = args.latent_dim
    cond_dim = dataset.cond_dim

    generator = Generator3D(latent_dim=latent_dim, cond_dim=cond_dim).to(device)
    critic = Critic3D(cond_dim=cond_dim).to(device)

    g_opt = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.0, 0.9))
    d_opt = torch.optim.Adam(critic.parameters(), lr=args.lr_d, betas=(0.0, 0.9))

    start_epoch = 1
    global_step = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        generator.load_state_dict(ckpt["generator"])
        critic.load_state_dict(ckpt["critic"])
        g_opt.load_state_dict(ckpt["g_opt"])
        d_opt.load_state_dict(ckpt["d_opt"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        print(f"[Resume] 从 {args.resume} 继续训练，起始 epoch={start_epoch}")

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    sample_dir = out_dir / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    train_meta = {
        "data": str(args.data),
        "num_samples": len(dataset),
        "tensor_shape": [1, 32, 32, 32],
        "cond_dim": cond_dim,
        "latent_dim": latent_dim,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "n_critic": args.n_critic,
        "lambda_gp": args.lambda_gp,
        "lr_g": args.lr_g,
        "lr_d": args.lr_d,
        "auto_normalize": bool(args.auto_normalize),
    }
    with (out_dir / "train_meta.json").open("w", encoding="utf-8") as f:
        json.dump(train_meta, f, ensure_ascii=False, indent=2)

    fixed_z = torch.randn(args.num_visualize, latent_dim, device=device)
    if cond_dim > 0:
        fixed_c = torch.zeros(args.num_visualize, cond_dim, device=device)
    else:
        fixed_c = torch.zeros(args.num_visualize, 0, device=device)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_d_losses: List[float] = []
        epoch_gp_values: List[float] = []
        epoch_g_losses: List[float] = []

        for real_x, real_c in loader:
            global_step += 1

            real_x = real_x.to(device)
            real_c = real_c.to(device)
            bsz = real_x.size(0)

            # 1) 训练 Critic
            z = torch.randn(bsz, latent_dim, device=device)
            with torch.no_grad():
                fake_x = generator(z, real_c)

            d_opt.zero_grad(set_to_none=True)
            real_score = critic(real_x, real_c).mean()
            fake_score = critic(fake_x, real_c).mean()
            gp = gradient_penalty(critic, real_x, fake_x, real_c, device)
            d_loss = fake_score - real_score + args.lambda_gp * gp
            d_loss.backward()
            d_opt.step()

            epoch_d_losses.append(float(d_loss.detach().item()))
            epoch_gp_values.append(float(gp.detach().item()))

            # 2) 每 n_critic 步训练一次 Generator
            g_loss_value = float("nan")
            if global_step % args.n_critic == 0:
                g_opt.zero_grad(set_to_none=True)
                z2 = torch.randn(bsz, latent_dim, device=device)
                gen_x = generator(z2, real_c)
                g_loss = -critic(gen_x, real_c).mean()
                g_loss.backward()
                g_opt.step()
                g_loss_value = float(g_loss.detach().item())
                epoch_g_losses.append(g_loss_value)

            if global_step % args.sample_every == 0:
                with torch.no_grad():
                    sample_x = generator(fixed_z, fixed_c).cpu().numpy()
                np.savez_compressed(
                    sample_dir / f"sample_step_{global_step:07d}.npz",
                    generated=sample_x,
                )

        mean_d = sum(epoch_d_losses) / max(1, len(epoch_d_losses))
        mean_gp = sum(epoch_gp_values) / max(1, len(epoch_gp_values))
        mean_g = sum(epoch_g_losses) / max(1, len(epoch_g_losses)) if epoch_g_losses else float("nan")
        print(f"[Epoch {epoch:03d}]D={mean_d:+.2f} G={mean_g:+.2f} GP={mean_gp:.4f}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(
                path=ckpt_dir / f"wgangp_epoch_{epoch:04d}.pt",
                epoch=epoch,
                global_step=global_step,
                generator=generator,
                critic=critic,
                g_opt=g_opt,
                d_opt=d_opt,
                latent_dim=latent_dim,
                cond_dim=cond_dim,
                args_dict=vars(args),
                scaling=dataset.scaling,
            )

    print("训练完成。")


# ----------------------------
# 生成与网格导出
# ----------------------------


def parse_condition_vector(raw: str, cond_dim: int) -> np.ndarray:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(values) != cond_dim:
        raise ValueError(f"条件向量长度不匹配，期望 {cond_dim}，实际 {len(values)}")
    return np.asarray(values, dtype=np.float32)


def denormalize_to_grade(x: np.ndarray, grade_min: float, grade_max: float) -> np.ndarray:
    # x in [-1, 1] -> [grade_min, grade_max]
    return ((x + 1.0) * 0.5) * (grade_max - grade_min) + grade_min


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# GAN generated mesh\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            # OBJ 1-based index
            f.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment GAN generated mesh\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_index\n")
        f.write("end_header\n")

        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        for face in faces:
            f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def export_mesh_from_volume(
    grade_volume: np.ndarray,
    cutoff: float,
    out_base: Path,
) -> Tuple[int, int]:
    if marching_cubes is None:
        raise RuntimeError("需要安装 scikit-image 才能执行 marching_cubes 导出网格")

    if np.max(grade_volume) < cutoff:
        out_base.with_suffix(".obj").write_text("# empty mesh: all values below cutoff\n", encoding="utf-8")
        out_base.with_suffix(".ply").write_text(
            "ply\nformat ascii 1.0\ncomment empty mesh\n"
            "element vertex 0\nproperty float x\nproperty float y\nproperty float z\n"
            "element face 0\nproperty list uchar int vertex_index\nend_header\n",
            encoding="utf-8",
        )
        return 0, 0

    verts, faces, _normals, _values = marching_cubes(grade_volume, level=cutoff)

    obj_path = out_base.with_suffix(".obj")
    ply_path = out_base.with_suffix(".ply")
    write_obj(obj_path, verts, faces)
    write_ply(ply_path, verts, faces)
    return int(len(verts)), int(len(faces))


def generate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    latent_dim = int(ckpt["latent_dim"])
    cond_dim = int(ckpt["cond_dim"])

    generator = Generator3D(latent_dim=latent_dim, cond_dim=cond_dim).to(device)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num = args.num_samples

    if cond_dim > 0:
        if args.condition_vector:
            c_np = np.tile(parse_condition_vector(args.condition_vector, cond_dim)[None, :], (num, 1))
        elif args.condition_file:
            c_raw = np.load(args.condition_file)
            if c_raw.ndim == 1:
                c_raw = c_raw[None, :]
            if c_raw.shape[1] != cond_dim:
                raise ValueError(f"condition_file 维度错误，期望第二维 {cond_dim}，实际 {c_raw.shape}")
            if c_raw.shape[0] < num:
                times = math.ceil(num / c_raw.shape[0])
                c_raw = np.tile(c_raw, (times, 1))
            c_np = c_raw[:num].astype(np.float32)
        else:
            c_np = np.random.uniform(-1.0, 1.0, size=(num, cond_dim)).astype(np.float32)
    else:
        c_np = np.zeros((num, 0), dtype=np.float32)

    z = torch.randn(num, latent_dim, device=device)
    c = torch.from_numpy(c_np).to(device)

    with torch.no_grad():
        fake = generator(z, c).cpu().numpy()[:, 0, :, :, :]

    np.savez_compressed(
        out_dir / "generated_tensors_norm.npz",
        tensors=fake.astype(np.float32),
        conditions=c_np.astype(np.float32),
    )

    # 反归一化到真实品位范围
    grade = denormalize_to_grade(fake, grade_min=args.grade_min, grade_max=args.grade_max)
    np.savez_compressed(
        out_dir / "generated_tensors_grade.npz",
        tensors=grade.astype(np.float32),
        conditions=c_np.astype(np.float32),
    )

    mesh_meta: List[Dict[str, object]] = []

    for i in range(num):
        base = out_dir / f"gan_sample_{i:05d}"
        np.savez_compressed(
            base.with_suffix(".npz"),
            tensor_norm=fake[i].astype(np.float32),
            tensor_grade=grade[i].astype(np.float32),
            condition=c_np[i].astype(np.float32),
            cutoff_grade=float(args.cutoff_grade),
        )

        if args.export_mesh:
            v_count, f_count = export_mesh_from_volume(
                grade_volume=grade[i],
                cutoff=args.cutoff_grade,
                out_base=base,
            )
            mesh_meta.append(
                {
                    "sample": i,
                    "vertices": v_count,
                    "faces": f_count,
                    "obj": str(base.with_suffix(".obj").name),
                    "ply": str(base.with_suffix(".ply").name),
                }
            )

    if args.export_mesh:
        with (out_dir / "mesh_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(mesh_meta, f, ensure_ascii=False, indent=2)

    print(f"生成完成，共 {num} 个样本，输出目录: {out_dir}")


# ----------------------------
# CLI
# ----------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="3D 条件 WGAN-GP：用于 32x32x32 矿体体素生成、条件控制与网格导出"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 数据预处理
    p_prep = sub.add_parser("prepare-data", help="从 outputs 目录批量汇总体素并归一化到[-1,1]")
    p_prep.add_argument("--source-dir", type=str, required=True, help="源目录，递归搜索其中的 npz/npy")
    p_prep.add_argument("--output", type=str, default="outputs/gan_dataset.npz", help="输出训练数据集 npz")
    p_prep.add_argument("--pattern", type=str, default="*.npz", help="递归匹配模式，例如 *.npz 或 **/*.npz")
    p_prep.add_argument("--tensor-key", type=str, default="", help="强制指定体素键名，默认自动识别")
    p_prep.add_argument("--cond-dim", type=int, default=0, help="为训练集补零条件向量维度")
    p_prep.add_argument("--exclude-step-snapshots", action="store_true", help="跳过 step_*.npz 时间步快照")
    p_prep.add_argument("--skip-invalid", action="store_true", help="遇到不符合 shape 的文件时跳过")
    p_prep.add_argument("--max-files", type=int, default=0, help="最多读取文件数，0 表示不限制")

    # 训练
    p_train = sub.add_parser("train", help="训练 3D 条件 WGAN-GP")
    p_train.add_argument("--data", type=str, required=True, help="输入数据(.npz/.npy)，体素建议已归一化到[-1,1]")
    p_train.add_argument("--out-dir", type=str, default="outputs/gan_runs", help="训练输出目录")
    p_train.add_argument("--auto-normalize", action="store_true", help="自动把输入体素归一化到[-1,1]")
    p_train.add_argument("--cond-dim", type=int, default=-1, help="条件维度；-1 表示从数据自动推断")

    p_train.add_argument("--epochs", type=int, default=300, help="训练轮数")
    p_train.add_argument("--batch-size", type=int, default=16, help="批大小")
    p_train.add_argument("--num-workers", type=int, default=0, help="DataLoader 进程数")
    p_train.add_argument("--latent-dim", type=int, default=128, help="潜变量 z 维度")

    p_train.add_argument("--lr-g", type=float, default=1e-4, help="生成器学习率")
    p_train.add_argument("--lr-d", type=float, default=1e-4, help="判别器(critic)学习率")
    p_train.add_argument("--n-critic", type=int, default=5, help="每训练一次 G 前，先训练 D 的步数")
    p_train.add_argument("--lambda-gp", type=float, default=10.0, help="梯度惩罚系数")

    p_train.add_argument("--log-every", type=int, default=50, help="日志打印间隔(step)")
    p_train.add_argument("--sample-every", type=int, default=200, help="保存生成样本间隔(step)")
    p_train.add_argument("--save-every", type=int, default=10, help="保存 checkpoint 间隔(epoch)")
    p_train.add_argument("--num-visualize", type=int, default=8, help="每次保存的可视化样本数")

    p_train.add_argument("--resume", type=str, default="", help="从 checkpoint 继续训练")
    p_train.add_argument("--device", type=str, default="cuda", help="训练设备: cuda/cpu")
    p_train.add_argument("--seed", type=int, default=20260316, help="随机种子")

    # 生成
    p_gen = sub.add_parser("generate", help="使用训练好的生成器采样，并导出体素/网格")
    p_gen.add_argument("--checkpoint", type=str, required=True, help="训练得到的 checkpoint 文件")
    p_gen.add_argument("--out-dir", type=str, default="outputs/oregen/gan_preview", help="生成结果目录")
    p_gen.add_argument("--num-samples", type=int, default=16, help="生成样本数量")

    p_gen.add_argument("--condition-vector", type=str, default="", help="单个条件向量，格式如: 0,0.3,-0.6")
    p_gen.add_argument("--condition-file", type=str, default="", help="条件矩阵 .npy，形状[N, cond_dim]")

    p_gen.add_argument("--grade-min", type=float, default=0.0, help="反归一化后的最小品位")
    p_gen.add_argument("--grade-max", type=float, default=1.0, help="反归一化后的最大品位")
    p_gen.add_argument("--cutoff-grade", type=float, default=0.35, help="面提取 Cut-off Grade")

    p_gen.add_argument("--export-mesh", action="store_true", help="启用 marching_cubes 导出 obj/ply")
    p_gen.add_argument("--device", type=str, default="cuda", help="推理设备: cuda/cpu")
    p_gen.add_argument("--seed", type=int, default=20260316, help="随机种子")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare-data":
        prepare_data(args)
    elif args.command == "train":
        train(args)
    elif args.command == "generate":
        generate(args)
    else:
        raise ValueError(f"未知命令: {args.command}")


if __name__ == "__main__":
    main()
