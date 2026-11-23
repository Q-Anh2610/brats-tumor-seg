"""Pipeline tiền xử lý dùng chung theo Quy ước v1.1 (RAS+, crop bbox, z-score, pad vuông, resize 224x224)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
from nibabel.orientations import apply_orientation, axcodes2ornt, io_orientation, ornt_transform
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.preprocess.validate_nifti import ALLOWED_LABELS, CHANNELS, find_case_files, is_orthogonal


def get_spatial_unit_factor_to_mm(img: nib.Nifti1Image) -> tuple[str, float]:
    """Đọc đơn vị không gian từ header NIfTI và trả về hệ số đổi sang mm."""
    xyzt_units = img.header.get_xyzt_units()[0]
    if xyzt_units in ("mm", "unknown"):
        return "mm", 1.0
    if xyzt_units == "meter":
        return "meter->mm", 1000.0
    if xyzt_units == "micron":
        return "micron->mm", 1e-3
    raise ValueError(f"Đơn vị không gian không hỗ trợ: {xyzt_units}")


def preprocess_case(
    case_dir: Path,
    out_cases_dir: Path,
    out_meta_dir: Path,
    image_size: int = 224,
    margin_xy: int = 5,
    eps: float = 1e-8,
    save_data: bool = True,
) -> dict:
    """Tiền xử lý 1 ca BraTS 2020, xuất 1 file cases/<case_id>.npz (float16 image + uint8 mask) và metadata/<case_id>.json."""
    case_id = case_dir.name
    files = find_case_files(case_dir)
    if any(v is None for v in files.values()):
        raise FileNotFoundError(f"{case_id}: Thiếu file trong {case_dir}")

    imgs = {k: nib.load(str(p)) for k, p in files.items() if p is not None}
    ref_img = imgs["t1"]
    orig_shape = [int(s) for s in ref_img.shape]
    orig_affine = ref_img.affine
    if not is_orthogonal(orig_affine):
        raise ValueError(f"{case_id}: Lưới không trực giao")

    unit_label, unit_scale = get_spatial_unit_factor_to_mm(ref_img)
    orig_zooms_mm = [float(z) * unit_scale for z in ref_img.header.get_zooms()[:3]]
    orig_axcodes = list(nib.aff2axcodes(orig_affine))

    # Kiểm tra cùng shape và affine
    for k, img in imgs.items():
        if list(img.shape) != orig_shape:
            raise ValueError(f"{case_id}: Lệch shape ở {k}: {img.shape} != {orig_shape}")
        if not np.allclose(img.affine, orig_affine, atol=1e-3):
            raise ValueError(f"{case_id}: Lệch affine ở {k}")

    # Bước 2: Đưa về hướng RAS+ đồng bộ và lưu phép biến đổi để đảo ngược
    orig_ornt = io_orientation(orig_affine)
    ras_ornt_target = axcodes2ornt(("R", "A", "S"))
    ras_ornt = ornt_transform(orig_ornt, ras_ornt_target)

    vols_ras = []
    for ch in CHANNELS:
        arr = np.asarray(imgs[ch].dataobj, dtype=np.float32)
        vols_ras.append(apply_orientation(arr, ras_ornt))
    vol_4ch = np.stack(vols_ras, axis=0)  # [4, X, Y, Z]

    seg_raw = np.asarray(imgs["seg"].dataobj, dtype=np.int16)
    unique_labels = {int(v) for v in np.unique(seg_raw)}
    unexpected = unique_labels - ALLOWED_LABELS
    if unexpected:
        raise ValueError(f"{case_id}: Nhãn ngoài danh sách {ALLOWED_LABELS}: {sorted(unexpected)}")
    seg_wt = np.isin(seg_raw, [1, 2, 4]).astype(np.float32)
    seg_ras = apply_orientation(seg_wt, ras_ornt)  # [X, Y, Z]

    ras_shape = [int(s) for s in seg_ras.shape]
    num_slices_z = ras_shape[2]

    # Bước 3 & 4: Vùng não sơ bộ + bounding box XY với lề 5 voxel
    brain_mask = np.any(vol_4ch != 0, axis=0)  # [X, Y, Z]
    if not np.any(brain_mask):
        raise ValueError(f"{case_id}: Vùng não rỗng")

    xs, ys, _ = np.where(brain_mask)
    x_start = max(0, int(xs.min()) - margin_xy)
    x_end = min(ras_shape[0], int(xs.max()) + margin_xy + 1)
    y_start = max(0, int(ys.min()) - margin_xy)
    y_end = min(ras_shape[1], int(ys.max()) + margin_xy + 1)

    # Bước 5: Z-score riêng từng kênh dùng voxel khác 0 của kênh đó trước crop/padding
    for c_idx in range(4):
        ch_vol = vol_4ch[c_idx]
        nz_mask = ch_vol != 0
        if np.any(nz_mask):
            vals = ch_vol[nz_mask]
            mean = float(vals.mean())
            std = float(vals.std())
            ch_vol[nz_mask] = (vals - mean) / (std + eps)
            ch_vol[~nz_mask] = 0.0

    # Crop đồng bộ theo XY, giữ nguyên toàn bộ Z
    vol_crop = vol_4ch[:, x_start:x_end, y_start:y_end, :]  # [4, Xc, Yc, Z]
    seg_crop = seg_ras[x_start:x_end, y_start:y_end, :]     # [Xc, Yc, Z]
    crop_h, crop_w = int(seg_crop.shape[0]), int(seg_crop.shape[1])

    # Bước 6: Padding đối xứng thành hình vuông (phần lẻ đặt ở cuối trục)
    side = max(crop_h, crop_w)
    pad_x_total = side - crop_h
    pad_y_total = side - crop_w
    x_before = pad_x_total // 2
    x_after = pad_x_total - x_before
    y_before = pad_y_total // 2
    y_after = pad_y_total - y_before

    vol_pad = np.pad(
        vol_crop,
        ((0, 0), (x_before, x_after), (y_before, y_after), (0, 0)),
        mode="constant",
        constant_values=0.0,
    )
    seg_pad = np.pad(
        seg_crop,
        ((x_before, x_after), (y_before, y_after), (0, 0)),
        mode="constant",
        constant_values=0.0,
    )

    # Bước 7 & 8: Chuyển lát Z lên batch [Z, 4, side, side] -> resize 224x224 bằng PyTorch
    img_t = torch.from_numpy(np.transpose(vol_pad, (3, 0, 1, 2))).float()
    seg_t = torch.from_numpy(np.transpose(seg_pad, (2, 0, 1))).unsqueeze(1).float()

    img_resized = F.interpolate(img_t, size=(image_size, image_size), mode="bilinear", align_corners=False)
    seg_resized = F.interpolate(seg_t, size=(image_size, image_size), mode="nearest")

    # Lưu trên đĩa dạng float16 + uint8 để giảm ~3-4x dung lượng khi lưu Kaggle Dataset;
    # DataLoader sẽ ép lại về float32 [4, 224, 224] và [1, 224, 224] đúng Quy ước v1.1.
    img_out = img_resized.numpy().astype(np.float16)              # [Z, 4, 224, 224]
    seg_out = seg_resized.squeeze(1).numpy().astype(np.uint8)     # [Z, 224, 224]
    slice_has_tumor = (seg_out.reshape(num_slices_z, -1).sum(axis=1) > 0).astype(np.uint8)
    slice_has_brain = (
        np.any(img_out != 0, axis=1).reshape(num_slices_z, -1).sum(axis=1) > 0
    ).astype(np.uint8)

    if save_data:
        out_cases_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_cases_dir / f"{case_id}.npz",
            images=img_out,
            masks=seg_out,
            slice_has_tumor=slice_has_tumor,
            slice_has_brain=slice_has_brain,
        )

    meta = {
        "case_id": case_id,
        "dataset": "BraTS2020",
        "ref_image_path": f"{case_id}/{files['t1'].name}",
        "channel_order": ["T1", "T1ce", "T2", "FLAIR"],
        "orig_shape": orig_shape,
        "orig_affine": [[float(x) for x in row] for row in orig_affine],
        "orig_zooms_mm": orig_zooms_mm,
        "spatial_units": unit_label,
        "orig_axcodes": orig_axcodes,
        "ras_ornt": [[int(r[0]), int(r[1])] for r in ras_ornt],
        "ras_shape": ras_shape,
        "crop_bbox": {
            "x_start": x_start,
            "x_end": x_end,
            "y_start": y_start,
            "y_end": y_end,
        },
        "crop_shape_xy": [crop_h, crop_w],
        "pad_xy": {
            "x_before": x_before,
            "x_after": x_after,
            "y_before": y_before,
            "y_after": y_after,
        },
        "padded_shape_xy": [side, side],
        "resized_shape_xy": [image_size, image_size],
        "num_slices_z": num_slices_z,
        "num_tumor_slices": int(slice_has_tumor.sum()),
        "num_brain_slices": int(slice_has_brain.sum()),
        "interpolation": {
            "image": "bilinear",
            "mask": "nearest",
            "align_corners": False,
        },
    }

    out_meta_dir.mkdir(parents=True, exist_ok=True)
    (out_meta_dir / f"{case_id}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiền xử lý toàn bộ BraTS 2020 theo Quy ước v1.1")
    parser.add_argument("--data-dir", type=Path, required=True, help="Thư mục gốc chứa BraTS20_Training_*")
    parser.add_argument("--out-dir", type=Path, default=Path("data/preprocessed/brats2020"))
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    out_cases_dir = args.out_dir / "cases"
    out_meta_dir = args.out_dir / "metadata"

    case_dirs = sorted(
        p for p in args.data_dir.iterdir()
        if p.is_dir() and p.name.startswith("BraTS20_Training_")
    )
    if not case_dirs:
        raise RuntimeError(f"Không tìm thấy thư mục BraTS20_Training_* trong {args.data_dir}")

    index_records = []
    for cd in tqdm(case_dirs, desc="Preprocessing BraTS 2020"):
        meta = preprocess_case(
            cd,
            out_cases_dir=out_cases_dir,
            out_meta_dir=out_meta_dir,
            image_size=args.image_size,
        )
        index_records.append(
            {
                "case_id": meta["case_id"],
                "num_slices_z": meta["num_slices_z"],
                "num_brain_slices": meta["num_brain_slices"],
                "num_tumor_slices": meta["num_tumor_slices"],
            }
        )

    (args.out_dir / "index.json").write_text(
        json.dumps(index_records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Hoàn tất tiền xử lý {len(case_dirs)} ca vào {args.out_dir}")


if __name__ == "__main__":
    main()
