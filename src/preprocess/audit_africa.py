"""Kiểm tra NIfTI BraTS-Africa và xuất thống kê ban đầu theo từng nhóm bệnh."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np


# Giữ tên kênh chung để ghép bảng với eda_brats2020_cases.csv.
CHANNELS = {"t1": "t1n", "t1ce": "t1c", "t2": "t2w", "flair": "t2f"}
ALLOWED_LABELS = {0, 1, 2, 3}
GROUPS = ("95_Glioma", "51_OtherNeoplasms")


def find_case_files(case_dir: Path) -> dict[str, Path]:
    """Tìm từng chuỗi theo hậu tố, không phụ thuộc thứ tự file trong thư mục."""
    found = {}
    for key, suffix in {**CHANNELS, "seg": "seg"}.items():
        candidates = []
        for path in case_dir.iterdir():
            lower = path.name.lower()
            stem = lower[:-7] if lower.endswith(".nii.gz") else lower[:-4] if lower.endswith(".nii") else None
            if path.is_file() and stem is not None and stem.endswith("-" + suffix):
                candidates.append(path)
        if len(candidates) != 1:
            raise ValueError(f"{key}: tìm thấy {len(candidates)} file, cần đúng một file")
        found[key] = candidates[0]
    return found


def geometry_mm(image: nib.Nifti1Image) -> tuple[np.ndarray, np.ndarray]:
    """Đọc lưới không gian và đổi đơn vị về mm, không tự gán mm khi unit thiếu."""
    if len(image.shape) != 3:
        raise ValueError(f"Ảnh không phải 3D: {image.shape}")
    unit = image.header.get_xyzt_units()[0]
    scales = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}
    if unit not in scales:
        raise ValueError(f"Đơn vị không gian chưa xác định: {unit}")
    affine = np.array(image.affine, dtype=float, copy=True)
    affine[:3, :] *= scales[unit]
    if not np.isfinite(affine).all() or not np.allclose(affine[3], [0, 0, 0, 1], rtol=0, atol=1e-6):
        raise ValueError("Affine không hợp lệ")
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    if np.any(spacing <= 0):
        raise ValueError("Spacing không hợp lệ")
    axes = affine[:3, :3] / spacing
    if not np.allclose(axes.T @ axes, np.eye(3), rtol=0, atol=1e-4):
        raise ValueError("Lưới không trực giao")
    if not np.allclose(spacing, np.asarray(image.header.get_zooms()[:3]) * scales[unit],
                       rtol=1e-4, atol=1e-5):
        raise ValueError("Spacing trong header không khớp affine")
    if not np.allclose(np.abs(axes).sum(axis=0), 1, rtol=0, atol=1e-4):
        raise ValueError("Lưới nghiêng, cần kiểm tra riêng trước khi đếm lát axial")
    return affine, spacing


def discover_cases(data_dir: Path) -> list[tuple[str, Path]]:
    """Nhận thư mục BraTS-Africa hoặc thư mục cha PKG chứa nó."""
    if not all((data_dir / group).is_dir() for group in GROUPS):
        nested = data_dir / "BraTS-Africa"
        if all((nested / group).is_dir() for group in GROUPS):
            data_dir = nested
        else:
            raise ValueError("DATA_DIR cần chứa 95_Glioma và 51_OtherNeoplasms")
    cases = []
    for group in GROUPS:
        cases.extend((group, path) for path in sorted((data_dir / group).iterdir())
                     if path.is_dir() and path.name.startswith("BraTS-SSA-"))
    if not cases:
        raise ValueError("Không tìm thấy ca BraTS-SSA-*")
    return cases


def audit_case(case_dir: Path, group: str) -> tuple[dict, dict | None, dict]:
    """Kiểm tra một ca, trả dòng audit, dòng thống kê và thông tin file nguồn."""
    record = {"case_id": case_dir.name, "group": group, "valid": False, "error": ""}
    inputs = {"case_id": case_dir.name, "group": group, "files": {}}
    try:
        files = find_case_files(case_dir)
        images = {key: nib.load(str(path)) for key, path in files.items()}
        reference = images["t1"]
        ref_affine, spacing_mm = geometry_mm(reference)
        shape = tuple(reference.shape)
        original_axcodes = nib.aff2axcodes(reference.affine)

        # Cùng shape chưa đủ, cần cùng vị trí và hướng trong không gian vật lý.
        for key, image in images.items():
            affine, _ = geometry_mm(image)
            if image.shape != shape or not np.allclose(affine, ref_affine, rtol=0, atol=1e-3):
                raise ValueError(f"{key}: shape hoặc affine không khớp T1")
            inputs["files"][key] = {
                "relative_path": f"{group}/{case_dir.name}/{files[key].name}",
                "bytes": files[key].stat().st_size,
                "spatial_unit": image.header.get_xyzt_units()[0],
            }

        # Đọc mask đúng kiểu gốc, tránh ép int làm mất nhãn lẻ hoặc NaN.
        segmentation = np.asarray(images["seg"].dataobj)
        if not np.isfinite(segmentation).all():
            raise ValueError("Mask có NaN hoặc Inf")
        labels = np.unique(segmentation)
        if not np.isin(labels, list(ALLOWED_LABELS)).all():
            raise ValueError(f"Mã nhãn ngoài 0/1/2/3: {labels.tolist()}")

        # Đưa về RAS+ bằng hoán vị/lật trục, không nội suy dữ liệu.
        transform = nib.orientations.ornt_transform(
            nib.orientations.io_orientation(reference.affine),
            nib.orientations.axcodes2ornt(("R", "A", "S")))
        seg_ras = nib.orientations.apply_orientation(segmentation, transform)
        wt = np.isin(seg_ras, [1, 2, 3])
        brain = np.zeros(wt.shape, dtype=bool)
        wt_voxels = int(wt.sum())
        voxel_volume_mm3 = float(np.prod(spacing_mm))

        eda = {"case_id": case_dir.name, "dataset": "BraTSAfrica", "group": group,
               "split": "external_descriptive",
               "shape_x": int(wt.shape[0]), "shape_y": int(wt.shape[1]),
               "num_slices_z": int(wt.shape[2]),
               "spacing_x_mm": float(spacing_mm[0]), "spacing_y_mm": float(spacing_mm[1]),
               "spacing_z_mm": float(spacing_mm[2]), "wt_voxels": wt_voxels,
               "wt_volume_ml": wt_voxels * voxel_volume_mm3 / 1000.0,
               "num_tumor_slices": int(np.count_nonzero(wt.any(axis=(0, 1)))),
               "label1_voxels": int(np.count_nonzero(segmentation == 1)),
               "label2_voxels": int(np.count_nonzero(segmentation == 2)),
               "label3_voxels": int(np.count_nonzero(segmentation == 3))}

        # Các thống kê dùng voxel khác 0 của từng kênh, cùng cách tính với BraTS 2020.
        for key in CHANNELS:
            volume = np.asarray(images[key].dataobj, dtype=np.float32)
            if not np.isfinite(volume).all():
                raise ValueError(f"{key}: ảnh có NaN hoặc Inf")
            volume = nib.orientations.apply_orientation(volume, transform)
            nonzero = volume != 0
            brain |= nonzero
            values = volume[nonzero]
            if values.size == 0:
                raise ValueError(f"{key}: toàn bộ kênh bằng 0")
            eda[f"{key}_mean_brain"] = float(values.mean())
            eda[f"{key}_std_brain"] = float(values.std())
            p05, p50, p95 = np.percentile(values, [5, 50, 95])
            eda[f"{key}_p05_brain"] = float(p05)
            eda[f"{key}_p50_brain"] = float(p50)
            eda[f"{key}_p95_brain"] = float(p95)
            tumor_values = volume[wt]
            eda[f"{key}_mean_wt"] = float(tumor_values.mean()) if tumor_values.size else 0.0
            eda[f"{key}_std_wt"] = float(tumor_values.std()) if tumor_values.size else 0.0

        brain_voxels = int(brain.sum())
        outside = int(np.count_nonzero(wt & ~brain))
        eda.update(brain_voxels=brain_voxels, wt_ratio_in_brain=wt_voxels / brain_voxels,
                   num_brain_slices=int(np.count_nonzero(brain.any(axis=(0, 1)))),
                   tumor_slice_ratio=eda["num_tumor_slices"] / eda["num_slices_z"],
                   wt_outside_brain_voxels=outside)
        record.update(valid=True, num_files=len(files), shape="x".join(map(str, shape)),
                      spacing_mm="x".join(f"{value:g}" for value in spacing_mm),
                      axcodes="".join(original_axcodes), labels=",".join(str(int(v)) for v in labels),
                      wt_voxels=wt_voxels, gt_empty=wt_voxels == 0,
                      wt_outside_brain_voxels=outside)
        inputs["seg_sha256"] = hashlib.sha256(files["seg"].read_bytes()).hexdigest()
        return record, eda, inputs
    except (ValueError, OSError, RuntimeError, EOFError) as exc:
        record["error"] = str(exc)
        return record, None, inputs


def distribution(values: list[float]) -> dict:
    """Tổng hợp theo bệnh nhân, SD mẫu chỉ tính khi có từ hai ca."""
    return {"mean": statistics.mean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else None,
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def summarize(records: list[dict]) -> dict:
    """Thống kê mô tả, không trộn dữ liệu Africa vào train hoặc validation."""
    if not records:
        return {"num_cases": 0}
    brain = sum(row["brain_voxels"] for row in records)
    slices = sum(row["num_slices_z"] for row in records)
    tumor_slices = sum(row["num_tumor_slices"] for row in records)
    return {"num_cases": len(records), "total_slices": slices,
            "total_tumor_slices": tumor_slices, "tumor_slice_pct": 100 * tumor_slices / slices,
            "wt_voxel_pct_in_brain": 100 * sum(row["wt_voxels"] for row in records) / brain,
            "wt_volume_ml": distribution([row["wt_volume_ml"] for row in records]),
            "num_tumor_slices_per_case": distribution([row["num_tumor_slices"] for row in records]),
            "gt_empty_cases": sum(row["wt_voxels"] == 0 for row in records),
            "wt_outside_brain_cases": sum(row["wt_outside_brain_voxels"] > 0 for row in records)}


def write_csv(path: Path, records: list[dict]) -> None:
    """Ghi cả các ca lỗi vào bảng audit, không bỏ ca khỏi báo cáo kiểm tra."""
    if not records:
        path.write_text("", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in records for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/africa/audit_run01"))
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError("Thư mục kết quả đã có dữ liệu, chọn tên lần chạy khác")
    cases = discover_cases(args.data_dir)
    duplicate_ids = {cid for cid, count in Counter(path.name for _, path in cases).items() if count > 1}
    audit, eda, inputs = [], [], []
    for index, (group, case_dir) in enumerate(cases, 1):
        if case_dir.name in duplicate_ids:
            row, stats, metadata = {"case_id": case_dir.name, "group": group, "valid": False,
                                     "error": "Case ID trùng giữa các nhóm"}, None, {}
        else:
            row, stats, metadata = audit_case(case_dir, group)
        audit.append(row)
        inputs.append(metadata)
        if stats is not None:
            eda.append(stats)
        status = "OK" if row["valid"] else "ERROR " + row["error"]
        print(f"[{index}/{len(cases)}] {case_dir.name} {status}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "nifti_audit_africa.csv", audit)
    write_csv(args.out_dir / "eda_africa_cases.csv", eda)
    observed_labels = sorted({int(value) for row in audit if row["valid"]
                              for value in row["labels"].split(",")})
    summary = {"dataset": "BraTSAfrica", "num_discovered_cases": len(cases),
               "num_valid_cases": len(eda), "num_invalid_cases": len(cases) - len(eda),
               "observed_label_values": observed_labels, "whole_tumor_labels": [1, 2, 3],
               "all_cases": summarize(eda),
               "by_group": {group: summarize([row for row in eda if row["group"] == group])
                            for group in GROUPS},
               "source": "https://www.cancerimagingarchive.net/collection/brats-africa/",
               "interpretation": "Giữ riêng nhóm glioma và nhóm u khác khi so sánh với BraTS 2020"}
    (args.out_dir / "africa_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "data_dir": str(args.data_dir.resolve()),
                "channel_mapping": CHANNELS, "allowed_labels": sorted(ALLOWED_LABELS),
                "scope": "all_cases_and_all_five_volumes", "hash_scope": "segmentation_files_only",
                "statistics": "Vùng não là hợp voxel khác 0, thống kê cường độ trên voxel khác 0 từng kênh",
                "inputs": inputs}
    (args.out_dir / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if summary["num_invalid_cases"]:
        raise SystemExit("Có ca lỗi, xem nifti_audit_africa.csv trước khi dùng thống kê")


if __name__ == "__main__":
    main()
