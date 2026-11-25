"""Tiền xử lý BraTS-Africa theo pipeline v1.1, xuất NPZ và metadata dùng chung."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from nibabel.orientations import apply_orientation, axcodes2ornt, io_orientation, ornt_transform

from src.preprocess.audit_africa import CHANNELS, discover_cases, find_case_files, geometry_mm


PIPELINE = {"version": "1.1", "image_size": 224, "crop_margin_xy": 5,
            "zscore_eps": 1e-8, "wt_labels": [1, 2, 3],
            "image_interpolation": "bilinear", "mask_interpolation": "nearest",
            "align_corners": False, "disk_image_dtype": "float16", "disk_mask_dtype": "uint8"}


def pipeline_hash() -> str:
    """Nhận biết các lần xử lý dùng cùng cấu hình."""
    return hashlib.sha256(json.dumps(PIPELINE, sort_keys=True).encode()).hexdigest()


def preprocess_case(case_dir: Path, group: str, out_dir: Path) -> dict:
    """Chuẩn hóa một ca, giữ toàn bộ lát axial và các thông tin khôi phục lưới gốc."""
    files = find_case_files(case_dir)
    images = {name: nib.load(str(path)) for name, path in files.items()}
    ref = images["t1"]
    orig_shape = list(ref.shape)
    orig_affine = np.array(ref.affine, dtype=np.float64)
    ref_affine_mm, spacing_mm = geometry_mm(ref)
    for name, image in images.items():
        affine_mm, _ = geometry_mm(image)
        if list(image.shape) != orig_shape or not np.allclose(affine_mm, ref_affine_mm, rtol=0, atol=1e-3):
            raise ValueError(f"{case_dir.name}: lưới {name} không khớp T1")

    # 1. Đưa ảnh và mask về cùng hướng RAS+ bằng hoán vị/lật trục.
    ras_ornt = ornt_transform(io_orientation(orig_affine), axcodes2ornt(("R", "A", "S")))
    volumes = []
    for name in CHANNELS:
        array = np.asarray(images[name].dataobj, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError(f"{case_dir.name}: {name} có NaN hoặc Inf")
        volumes.append(apply_orientation(array, ras_ornt))
    volume = np.stack(volumes, axis=0)

    segmentation = np.asarray(images["seg"].dataobj)
    if not np.isfinite(segmentation).all() or not np.isin(segmentation, [0, 1, 2, 3]).all():
        raise ValueError(f"{case_dir.name}: mask có mã nhãn không hợp lệ")
    wt = np.isin(segmentation, [1, 2, 3]).astype(np.float32)
    wt = apply_orientation(wt, ras_ornt)
    ras_shape = list(wt.shape)

    # 2. Crop XY từ hợp vùng não, không dùng GT để chọn vùng crop.
    brain = np.any(volume != 0, axis=0)
    if not brain.any():
        raise ValueError(f"{case_dir.name}: vùng não rỗng")
    xs, ys, _ = np.where(brain)
    margin = PIPELINE["crop_margin_xy"]
    x0, x1 = max(0, int(xs.min()) - margin), min(ras_shape[0], int(xs.max()) + margin + 1)
    y0, y1 = max(0, int(ys.min()) - margin), min(ras_shape[1], int(ys.max()) + margin + 1)

    # 3. Z-score từng kênh trên toàn ca trước crop, nền tiếp tục bằng 0.
    normalization = {}
    for index, name in enumerate(CHANNELS):
        channel = volume[index]
        selected = channel != 0
        values = channel[selected]
        if not values.size:
            raise ValueError(f"{case_dir.name}: kênh {name} rỗng")
        mean, std = float(values.mean()), float(values.std())
        channel[selected] = (values - mean) / (std + PIPELINE["zscore_eps"])
        channel[~selected] = 0
        normalization[name] = {"mean": mean, "std": std, "num_nonzero_voxels": int(values.size)}

    cropped = volume[:, x0:x1, y0:y1, :]
    cropped_mask = wt[x0:x1, y0:y1, :]
    crop_h, crop_w = cropped_mask.shape[:2]
    side = max(crop_h, crop_w)
    xb = (side - crop_h) // 2
    xa = side - crop_h - xb
    yb = (side - crop_w) // 2
    ya = side - crop_w - yb

    # 4. Pad vuông bằng 0, phần lẻ đặt ở cuối trục giống pipeline BraTS 2020.
    padded = np.pad(cropped, ((0, 0), (xb, xa), (yb, ya), (0, 0)), constant_values=0)
    padded_mask = np.pad(cropped_mask, ((xb, xa), (yb, ya), (0, 0)), constant_values=0)

    # 5. Đưa Z lên batch, resize MRI bilinear và mask nearest bằng cùng thư viện.
    image_tensor = torch.from_numpy(np.transpose(padded, (3, 0, 1, 2))).float()
    mask_tensor = torch.from_numpy(np.transpose(padded_mask, (2, 0, 1))).unsqueeze(1).float()
    size = PIPELINE["image_size"]
    resized = F.interpolate(image_tensor, size=(size, size), mode="bilinear", align_corners=False)
    resized_mask = F.interpolate(mask_tensor, size=(size, size), mode="nearest")
    stored_images = resized.numpy().astype(np.float16)
    stored_masks = resized_mask.squeeze(1).numpy().astype(np.uint8)
    if not np.isfinite(stored_images).all():
        raise ValueError(f"{case_dir.name}: dữ liệu sau chuẩn hóa không hữu hạn")
    num_z = ras_shape[2]
    has_tumor = (stored_masks.reshape(num_z, -1).sum(axis=1) > 0).astype(np.uint8)
    has_brain = (np.any(stored_images != 0, axis=1).reshape(num_z, -1).sum(axis=1) > 0).astype(np.uint8)

    # 6. Ghi mỗi ca thành một NPZ, tránh để file dở dang nếu quá trình bị ngắt.
    cases_dir, meta_dir = out_dir / "cases", out_dir / "metadata"
    cases_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    target = cases_dir / f"{case_dir.name}.npz"
    temporary = target.with_suffix(".npz.partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, images=stored_images, masks=stored_masks,
                            slice_has_tumor=has_tumor, slice_has_brain=has_brain)
    temporary.replace(target)

    metadata = {"case_id": case_dir.name, "dataset": "BraTSAfrica", "group": group,
                "ref_image_path": f"{group}/{case_dir.name}/{files['t1'].name}",
                "channel_order": ["T1", "T1ce", "T2", "FLAIR"],
                "source_channel_order": list(CHANNELS.values()),
                "wt_labels": [1, 2, 3], "orig_shape": orig_shape,
                "orig_affine": orig_affine.tolist(), "orig_zooms_mm": spacing_mm.tolist(),
                "spatial_units": images['t1'].header.get_xyzt_units()[0],
                "orig_axcodes": list(nib.aff2axcodes(orig_affine)),
                "ras_ornt": ras_ornt.astype(int).tolist(), "ras_shape": ras_shape,
                "crop_bbox": {"x_start": x0, "x_end": x1, "y_start": y0, "y_end": y1},
                "crop_shape_xy": [crop_h, crop_w],
                "pad_xy": {"x_before": xb, "x_after": xa, "y_before": yb, "y_after": ya},
                "padded_shape_xy": [side, side], "resized_shape_xy": [size, size],
                "num_slices_z": num_z, "num_tumor_slices": int(has_tumor.sum()),
                "num_brain_slices": int(has_brain.sum()),
                "interpolation": {"image": "bilinear", "mask": "nearest", "align_corners": False},
                "normalization": normalization, "pipeline_hash": pipeline_hash()}
    (meta_dir / f"{case_dir.name}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return {"case_id": case_dir.name, "group": group, "num_slices_z": num_z,
            "num_brain_slices": int(has_brain.sum()), "num_tumor_slices": int(has_tumor.sum())}


def run_worker(case_dir: Path, group: str, out_dir: Path) -> dict:
    """Giới hạn luồng CPU mỗi tiến trình để các ca không tranh chấp tài nguyên."""
    torch.set_num_threads(2)
    return preprocess_case(case_dir, group, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers cần lớn hơn 0")
    if (args.out_dir / "index.json").exists():
        raise FileExistsError("Bộ dữ liệu đã hoàn thành ở đường dẫn này, chọn thư mục mới")
    cases = discover_cases(args.data_dir)
    if len(cases) != 146 or len({case.name for _, case in cases}) != 146:
        raise ValueError("Cần đúng 146 ca riêng biệt theo bộ dữ liệu đã kiểm tra")
    existing = list((args.out_dir / "cases").glob("*.npz"))
    if existing:
        raise FileExistsError("Có file NPZ từ lần chạy dở, cần kiểm tra trước khi chạy lại")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_worker, case, group, args.out_dir): case.name for group, case in cases}
        for future in as_completed(futures):
            row = future.result()
            records.append(row)
            print(f"[{len(records)}/{len(cases)}] {row['case_id']} {row['group']} OK", flush=True)
    records.sort(key=lambda row: (row["group"], row["case_id"]))
    (args.out_dir / "index.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    cohorts = {group: [row["case_id"] for row in records if row["group"] == group]
               for group, _ in cases}
    (args.out_dir / "cohorts.json").write_text(json.dumps(cohorts, indent=2), encoding="utf-8")
    info = {"created_utc": datetime.now(timezone.utc).isoformat(), "num_cases": len(records),
            "pipeline": PIPELINE, "pipeline_hash": pipeline_hash(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "versions": {"numpy": np.__version__, "nibabel": nib.__version__, "torch": torch.__version__},
            "usage": "Thống kê và kiểm thử ngoài nguồn, không trộn vào train/validation BraTS 2020"}
    (args.out_dir / "preprocessing_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Hoàn tất {len(records)} ca tại {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
