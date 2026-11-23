"""Kiểm tra toàn vẹn dữ liệu 3D NIfTI BraTS 2020 trước khi tiền xử lý (Quy ước v1.1)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import nibabel as nib
import numpy as np
from tqdm import tqdm

CHANNELS = ("t1", "t1ce", "t2", "flair")
ALLOWED_LABELS = {0, 1, 2, 4}


def find_case_files(case_dir: Path) -> dict[str, Path | None]:
    """Tìm 4 chuỗi MRI và file seg theo tên thành phần, xử lý ngoại lệ tên file BraTS20_Training_355."""
    nii_files = sorted(
        p for p in case_dir.iterdir()
        if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))
    )
    found: dict[str, Path | None] = {ch: None for ch in CHANNELS}
    found["seg"] = None

    for p in nii_files:
        stem = p.name[:-7] if p.name.endswith(".nii.gz") else p.name[:-4]
        lower_stem = stem.lower()
        for ch in CHANNELS:
            if lower_stem.endswith(f"_{ch}"):
                found[ch] = p
                break
        else:
            # BraTS20_Training_355 có file tên W39_1998.09.19_Segm.nii thay vì *_seg.nii
            if lower_stem.endswith("_seg") or lower_stem.endswith("_segm"):
                found["seg"] = p

    return found


def is_orthogonal(affine: np.ndarray, atol: float = 1e-4) -> bool:
    """Kiểm tra ma trận xoay/scale 3x3 có trực giao với các trục không."""
    rot = affine[:3, :3]
    col_norms = np.linalg.norm(rot, axis=0)
    if np.any(col_norms <= 0):
        return False
    normed = rot / col_norms
    gram = normed.T @ normed
    return bool(np.allclose(gram, np.eye(3), atol=atol))


def audit_case(case_dir: Path) -> dict:
    """Kiểm tra 1 ca bệnh nhân BraTS 2020 và trả về bản ghi audit."""
    case_id = case_dir.name
    files = find_case_files(case_dir)
    missing = [k for k, v in files.items() if v is None]
    if missing:
        return {
            "case_id": case_id,
            "valid": False,
            "error": f"missing_files:{','.join(missing)}",
            "shape": "",
            "zooms_mm": "",
            "axcodes": "",
            "orthogonal": False,
            "labels": "",
            "brain_voxels": 0,
            "wt_voxels": 0,
        }

    imgs = {k: nib.load(str(p)) for k, p in files.items() if p is not None}
    ref = imgs["t1"]
    ref_shape = ref.shape
    ref_affine = ref.affine

    for k, img in imgs.items():
        if img.shape != ref_shape:
            return {
                "case_id": case_id,
                "valid": False,
                "error": f"shape_mismatch:{k}:{img.shape}!={ref_shape}",
                "shape": str(ref_shape),
                "zooms_mm": str(ref.header.get_zooms()[:3]),
                "axcodes": "".join(nib.aff2axcodes(ref_affine)),
                "orthogonal": is_orthogonal(ref_affine),
                "labels": "",
                "brain_voxels": 0,
                "wt_voxels": 0,
            }
        if not np.allclose(img.affine, ref_affine, atol=1e-3):
            return {
                "case_id": case_id,
                "valid": False,
                "error": f"affine_mismatch:{k}",
                "shape": str(ref_shape),
                "zooms_mm": str(ref.header.get_zooms()[:3]),
                "axcodes": "".join(nib.aff2axcodes(ref_affine)),
                "orthogonal": is_orthogonal(ref_affine),
                "labels": "",
                "brain_voxels": 0,
                "wt_voxels": 0,
            }

    ortho = is_orthogonal(ref_affine)
    if not ortho:
        return {
            "case_id": case_id,
            "valid": False,
            "error": "non_orthogonal_grid",
            "shape": str(ref_shape),
            "zooms_mm": str(ref.header.get_zooms()[:3]),
            "axcodes": "".join(nib.aff2axcodes(ref_affine)),
            "orthogonal": False,
            "labels": "",
            "brain_voxels": 0,
            "wt_voxels": 0,
        }

    seg_data = np.asarray(imgs["seg"].dataobj)
    unique_labels = {int(v) for v in np.unique(seg_data)}
    unexpected = unique_labels - ALLOWED_LABELS
    if unexpected:
        return {
            "case_id": case_id,
            "valid": False,
            "error": f"unexpected_labels:{sorted(unexpected)}",
            "shape": str(ref_shape),
            "zooms_mm": str(ref.header.get_zooms()[:3]),
            "axcodes": "".join(nib.aff2axcodes(ref_affine)),
            "orthogonal": ortho,
            "labels": str(sorted(unique_labels)),
            "brain_voxels": 0,
            "wt_voxels": 0,
        }

    brain_mask = np.zeros(ref_shape, dtype=bool)
    for ch in CHANNELS:
        brain_mask |= np.asarray(imgs[ch].dataobj) != 0
    brain_voxels = int(brain_mask.sum())
    if brain_voxels == 0:
        return {
            "case_id": case_id,
            "valid": False,
            "error": "empty_brain_mask",
            "shape": str(ref_shape),
            "zooms_mm": str(ref.header.get_zooms()[:3]),
            "axcodes": "".join(nib.aff2axcodes(ref_affine)),
            "orthogonal": ortho,
            "labels": str(sorted(unique_labels)),
            "brain_voxels": 0,
            "wt_voxels": 0,
        }

    wt_voxels = int(np.isin(seg_data, [1, 2, 4]).sum())
    return {
        "case_id": case_id,
        "valid": True,
        "error": "",
        "shape": str(ref_shape),
        "zooms_mm": str(tuple(float(z) for z in ref.header.get_zooms()[:3])),
        "axcodes": "".join(nib.aff2axcodes(ref_affine)),
        "orthogonal": ortho,
        "labels": str(sorted(unique_labels)),
        "brain_voxels": brain_voxels,
        "wt_voxels": wt_voxels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiểm tra toàn bộ ca BraTS 2020 NIfTI")
    parser.add_argument("--data-dir", type=Path, required=True, help="Thư mục chứa BraTS20_Training_*")
    parser.add_argument("--output-csv", type=Path, default=Path("results/nifti_audit.csv"))
    args = parser.parse_args()

    case_dirs = sorted(
        p for p in args.data_dir.iterdir()
        if p.is_dir() and p.name.startswith("BraTS20_Training_")
    )
    if not case_dirs:
        raise RuntimeError(f"Không tìm thấy thư mục BraTS20_Training_* trong {args.data_dir}")

    rows = [audit_case(cd) for cd in tqdm(case_dirs, desc="Auditing BraTS 2020")]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_valid = sum(1 for r in rows if r["valid"])
    n_invalid = len(rows) - n_valid
    print(f"Tổng ca: {len(rows)} | Hợp lệ: {n_valid} | Lỗi: {n_invalid}")
    if n_invalid:
        for r in rows:
            if not r["valid"]:
                print(f"  [LỖI] {r['case_id']}: {r['error']}")


if __name__ == "__main__":
    main()
