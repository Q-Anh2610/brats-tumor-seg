"""Đọc NIfTI BraTS 2020 và kiểm tra split, nhãn cùng hình học trên lưới gốc."""

from __future__ import annotations

import json
from pathlib import Path
import nibabel as nib
import numpy as np


def read_split(path: Path, name: str = "test") -> list[str]:
    """Đọc danh sách ca từ splits.json, phát hiện ID trùng và rò rỉ giữa các tập."""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    groups = obj["splits"]
    seen = set()
    for group, ids in groups.items():
        if not isinstance(ids, list) or len(ids) != len(set(ids)):
            raise ValueError(f"Invalid or duplicate IDs in split {group}")
        for cid in ids:
            if not isinstance(cid, str) or Path(cid).name != cid or not cid.startswith("BraTS20_Training_"):
                raise ValueError(f"Invalid case ID: {cid}")
        if seen.intersection(ids):
            raise ValueError("Patient leakage across splits.")
        seen.update(ids)
        expected = obj.get("counts", {}).get(group)
        if expected is not None and expected != len(ids):
            raise ValueError(f"Count mismatch: {group}")
    ids = groups[name]
    if not ids:
        raise ValueError("Selected split is empty.")
    return ids


def find_raw_file(raw_dir: Path, cid: str, channel: str) -> Path:
    """Tìm đúng một file theo hậu tố, chấp nhận .nii/.nii.gz và nhãn ngoại lệ _Segm."""
    case_dir = Path(raw_dir) / cid
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Missing case directory: {case_dir}")
    matches = []
    for p in case_dir.iterdir():
        lower = p.name.lower()
        stem = lower[:-7] if lower.endswith(".nii.gz") else lower[:-4] if lower.endswith(".nii") else None
        if stem is None or not p.is_file():
            continue
        suffixes = ("_seg", "_segm") if channel == "seg" else ("_" + channel,)
        if stem.endswith(suffixes):
            matches.append(p)
    if len(matches) != 1:
        raise ValueError(f"{cid}: expected exactly one {channel} NIfTI, found {matches}")
    return matches[0]


def grid_mm(img: nib.Nifti1Image, assume_unknown_mm: bool = False) -> tuple:
    """Đổi affine và spacing sang mm, kiểm tra lưới trực giao.

    Không sửa header gốc. Đơn vị unknown chỉ được xem là mm khi người chạy
    đã xác minh và bật assume_unknown_mm, không suy đoán từ tên dataset."""
    if len(img.shape) != 3:
        raise ValueError(f"Expected 3D NIfTI, got {img.shape}")
    unit = img.header.get_xyzt_units()[0]
    scales = {"mm": 1., "meter": 1000., "micron": .001}
    if unit == "unknown" and assume_unknown_mm:
        scale = 1.
    elif unit in scales:
        scale = scales[unit]
    else:
        raise ValueError("Unknown NIfTI spatial unit. Confirm mm from the data source before using --assume-unknown-mm.")
    affine = np.array(img.affine, dtype=float, copy=True)
    affine[:3, :] *= scale
    if not np.isfinite(affine).all() or not np.allclose(affine[3], [0, 0, 0, 1], rtol=0, atol=1e-6):
        raise ValueError("Invalid affine.")
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    if (spacing <= 0).any():
        raise ValueError("Degenerate affine.")
    axes = affine[:3, :3] / spacing
    if not np.allclose(axes.T @ axes, np.eye(3), rtol=0, atol=1e-4):
        raise ValueError("Non-orthogonal grid: spacing-only HD95 is not valid.")
    header_spacing = np.asarray(img.header.get_zooms()[:3]) * scale
    if not np.allclose(spacing, header_spacing, rtol=1e-4, atol=1e-5):
        raise ValueError("Header spacing disagrees with affine.")
    return affine, spacing, unit


def same_grid(reference: nib.Nifti1Image, other: nib.Nifti1Image,
              assume_unknown_mm: bool = False) -> np.ndarray:
    """Kiểm tra shape và affine vật lý của hai ảnh, trả spacing_mm theo thứ tự trục."""
    ref_aff, spacing, _ = grid_mm(reference, assume_unknown_mm)
    other_aff, _, _ = grid_mm(other, assume_unknown_mm)
    if reference.shape != other.shape or not np.allclose(ref_aff, other_aff, rtol=0, atol=1e-3):
        raise ValueError("Prediction/GT/MRI grid mismatch (shape or affine in mm).")
    return spacing


def checked_array(img: nib.Nifti1Image, allowed: list[int] | None = None) -> np.ndarray:
    """Đọc mảng, từ chối NaN/Inf và mã nhãn ngoài danh sách, không ép kiểu để che lỗi."""
    arr = np.asarray(img.dataobj)
    if not np.isfinite(arr).all():
        raise ValueError("NIfTI contains NaN/Inf.")
    if allowed is not None and not np.isin(arr, allowed).all():
        raise ValueError(f"Unexpected label values: {np.unique(arr)}, allowed: {allowed}")
    return arr


def load_case(raw_dir: Path, pred_dir: Path, cid: str,
              assume_unknown_mm: bool = False) -> tuple:
    """Nạp GT và dự đoán của một ca, kiểm tra lưới rồi tạo mask WT.

    GT gộp nhãn 1/2/4, dự đoán phải sẵn là 0/1 trên lưới gốc.
    Trả ảnh NIfTI, mask, spacing và đường dẫn phục vụ đánh giá/truy vết."""
    gt_path = find_raw_file(raw_dir, cid, "seg")
    pred_path = Path(pred_dir) / f"{cid}_pred_wt.nii.gz"
    gt_img, pred_img = nib.load(str(gt_path)), nib.load(str(pred_path))
    spacing = same_grid(gt_img, pred_img, assume_unknown_mm)
    gt = np.isin(checked_array(gt_img, [0, 1, 2, 4]), [1, 2, 4])
    pred = checked_array(pred_img, [0, 1]).astype(bool)
    return gt_img, pred_img, gt, pred, spacing, gt_path, pred_path
