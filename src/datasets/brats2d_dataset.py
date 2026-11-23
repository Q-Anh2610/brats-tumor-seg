"""PyTorch Dataset 2D cho BraTS 2020 (đọc .npz -> cache .npy memmap, hỗ trợ Augmentation v1.1)."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def build_train_geometric_augmenter(aug_cfg: dict[str, Any]) -> A.Compose:
    """Tạo bộ biến đổi hình học Albumentations tương thích cả bản 1.x và 2.x."""
    aff_cfg = aug_cfg.get("affine", {})
    affine_kwargs: dict[str, Any] = {
        "scale": tuple(aff_cfg.get("scale", (0.9, 1.1))),
        "translate_percent": tuple(aff_cfg.get("translate_percent", (-0.05, 0.05))),
        "rotate": tuple(aff_cfg.get("rotate", (-15, 15))),
        "interpolation": cv2.INTER_LINEAR,
        "mask_interpolation": cv2.INTER_NEAREST,
        "border_mode": cv2.BORDER_CONSTANT,
        "p": float(aff_cfg.get("p", 0.5)),
    }
    sig_params = inspect.signature(A.Affine.__init__).parameters
    if "fill" in sig_params:
        affine_kwargs["fill"] = 0.0
    elif "cval" in sig_params:
        affine_kwargs["cval"] = 0.0
    if "fill_mask" in sig_params:
        affine_kwargs["fill_mask"] = 0
    elif "cval_mask" in sig_params:
        affine_kwargs["cval_mask"] = 0

    return A.Compose(
        [
            A.HorizontalFlip(p=float(aug_cfg.get("horizontal_flip_p", 0.5))),
            A.VerticalFlip(p=float(aug_cfg.get("vertical_flip_p", 0.5))),
            A.Affine(**affine_kwargs),
        ]
    )


def apply_brain_intensity_aug(
    img: np.ndarray,
    rng: np.random.Generator,
    aug_cfg: dict[str, Any],
) -> np.ndarray:
    """Biến đổi cường độ nhẹ chỉ trên vùng não (img != 0), giữ nguyên nền 0."""
    int_cfg = aug_cfg.get("intensity", {})
    bc_p = float(int_cfg.get("brightness_contrast_p", 0.3))
    gn_p = float(int_cfg.get("gaussian_noise_p", 0.2))

    brain_mask = img != 0.0
    if not np.any(brain_mask):
        return img

    if rng.random() < bc_p:
        lo, hi = int_cfg.get("brightness_contrast_range", (0.9, 1.1))
        scale = float(rng.uniform(lo, hi))
        img = np.where(brain_mask, img * scale, 0.0)

    if rng.random() < gn_p:
        std_lo, std_hi = int_cfg.get("gaussian_noise_std", (0.01, 0.05))
        noise_std = float(rng.uniform(std_lo, std_hi))
        noise = rng.normal(0.0, noise_std, size=img.shape).astype(np.float32)
        img = np.where(brain_mask, img + noise, 0.0)

    return img.astype(np.float32, copy=False)


def prepare_mmap_cache(
    preprocessed_dir: Path,
    cache_dir: Path,
    case_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Giải nén các file <case_id>.npz sang .npy memmap để DataLoader đọc nhanh và nhẹ RAM."""
    cases_dir = preprocessed_dir / "cases"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"

    manifest: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    missing_ids = [
        cid
        for cid in case_ids
        if cid not in manifest
        or not (cache_dir / f"{cid}_img.npy").exists()
        or not (cache_dir / f"{cid}_msk.npy").exists()
    ]

    if missing_ids:
        for cid in tqdm(missing_ids, desc="Unpacking .npz -> .npy cache"):
            npz_path = cases_dir / f"{cid}.npz"
            if not npz_path.exists():
                raise FileNotFoundError(f"Không tìm thấy {npz_path}")
            with np.load(npz_path) as data:
                imgs = data["images"]
                msks = data["masks"]
                has_tumor = data["slice_has_tumor"].astype(int).tolist()
                has_brain = data["slice_has_brain"].astype(int).tolist()

            np.save(cache_dir / f"{cid}_img.npy", imgs)
            np.save(cache_dir / f"{cid}_msk.npy", msks)
            manifest[cid] = {
                "num_slices_z": int(imgs.shape[0]),
                "slice_has_tumor": has_tumor,
                "slice_has_brain": has_brain,
            }

        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    return manifest


class BraTS2DDataset(Dataset):
    """Dataset trả về từng lát axial (float32 [4, 224, 224], float32 [1, 224, 224], case_id, slice_index)."""

    def __init__(
        self,
        cache_dir: Path,
        manifest: dict[str, dict[str, Any]],
        case_ids: list[str],
        is_train: bool = False,
        aug_cfg: dict[str, Any] | None = None,
        neg_slice_ratio: float = 1.0,
        brain_only: bool = False,
        seed: int = 42,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.manifest = manifest
        self.case_ids = list(case_ids)
        self.is_train = is_train
        self.aug_cfg = aug_cfg or {}
        self.neg_slice_ratio = float(neg_slice_ratio)
        self.brain_only = brain_only
        self.base_seed = seed

        self.geo_aug = build_train_geometric_augmenter(self.aug_cfg) if is_train else None
        self.pos_samples: list[tuple[str, int]] = []
        self.neg_samples: list[tuple[str, int]] = []
        self.samples: list[tuple[str, int]] = []

        for cid in self.case_ids:
            info = self.manifest[cid]
            num_z = int(info["num_slices_z"])
            has_tumor = info["slice_has_tumor"]
            has_brain = info["slice_has_brain"]
            for z in range(num_z):
                if self.brain_only and not has_brain[z]:
                    continue
                if has_tumor[z]:
                    self.pos_samples.append((cid, z))
                else:
                    self.neg_samples.append((cid, z))

        self.resample_epoch(epoch=0)

    def resample_epoch(self, epoch: int) -> None:
        """Lấy mẫu lại tập lát âm (không có u) mỗi epoch khi train; val/test luôn giữ 100% lát."""
        if not self.is_train or self.neg_slice_ratio >= 1.0:
            self.samples = self.pos_samples + self.neg_samples
            if not self.is_train:
                # Giữ đúng thứ tự bệnh nhân & chỉ số lát khi đánh giá
                self.samples.sort(key=lambda x: (x[0], x[1]))
            return

        rng = np.random.default_rng(self.base_seed + epoch)
        n_neg = max(0, int(round(len(self.neg_samples) * self.neg_slice_ratio)))
        if n_neg > 0 and self.neg_samples:
            chosen_idx = rng.choice(len(self.neg_samples), size=n_neg, replace=False)
            sampled_neg = [self.neg_samples[int(i)] for i in chosen_idx]
        else:
            sampled_neg = []
        self.samples = self.pos_samples + sampled_neg

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str, int]:
        cid, z = self.samples[idx]
        img_mmap = np.load(self.cache_dir / f"{cid}_img.npy", mmap_mode="r")
        msk_mmap = np.load(self.cache_dir / f"{cid}_msk.npy", mmap_mode="r")

        # [4, H, W] float32 và [H, W] uint8 -> copy sang bộ nhớ thường để augment
        img = np.array(img_mmap[z], dtype=np.float32)
        msk = np.array(msk_mmap[z], dtype=np.uint8)

        if self.is_train and self.geo_aug is not None:
            # Albumentations nhận [H, W, C]
            hwc = np.transpose(img, (1, 2, 0))
            augmented = self.geo_aug(image=hwc, mask=msk)
            img = np.transpose(augmented["image"], (2, 0, 1)).astype(np.float32)
            msk = augmented["mask"]

            # Seed theo worker/call từ numpy global rng của worker PyTorch
            rng = np.random.default_rng()
            img = apply_brain_intensity_aug(img, rng, self.aug_cfg)

        img_tensor = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))
        msk_tensor = torch.from_numpy(np.ascontiguousarray(msk, dtype=np.float32)).unsqueeze(0)
        return img_tensor, msk_tensor, cid, int(z)
