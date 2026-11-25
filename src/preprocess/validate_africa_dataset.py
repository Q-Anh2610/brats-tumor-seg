"""Kiểm tra toàn bộ NPZ/metadata trước khi đưa bộ Africa lên Kaggle."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from nibabel.orientations import apply_orientation, axcodes2ornt, io_orientation, ornt_transform

from src.preprocess.preprocess_africa import pipeline_hash


def file_hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def restore_mask(mask: np.ndarray, metadata: dict) -> np.ndarray:
    """Khôi phục mask mẫu qua metadata, chỉ để kiểm tra phép ghép và hướng trục."""
    torch.set_num_threads(2)
    padded = F.interpolate(torch.from_numpy(mask.astype(np.float32)).unsqueeze(1),
                           size=metadata["padded_shape_xy"], mode="bilinear", align_corners=False)
    padded = padded.squeeze(1).numpy()
    pad = metadata["pad_xy"]
    height, width = metadata["padded_shape_xy"]
    cropped = padded[:, pad["x_before"]:height - pad["x_after"],
                     pad["y_before"]:width - pad["y_after"]]
    bbox = metadata["crop_bbox"]
    ras = np.zeros(metadata["ras_shape"], dtype=np.float32)
    ras[bbox["x_start"]:bbox["x_end"], bbox["y_start"]:bbox["y_end"], :] = cropped.transpose(1, 2, 0)
    reverse = ornt_transform(axcodes2ornt(("R", "A", "S")),
                             io_orientation(np.asarray(metadata["orig_affine"])))
    return apply_orientation(ras, reverse) >= 0.5


def validate_case(data_dir: Path, raw_dir: Path, case_id: str, roundtrip: bool) -> dict:
    """Mở lại dữ liệu đã lưu, kiểm tra schema, metadata và cờ lát cắt."""
    path = data_dir / "cases" / f"{case_id}.npz"
    meta_path = data_dir / "metadata" / f"{case_id}.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["case_id"] != case_id or meta["dataset"] != "BraTSAfrica":
        raise ValueError(f"{case_id}: sai định danh metadata")
    if meta["pipeline_hash"] != pipeline_hash() or meta["wt_labels"] != [1, 2, 3]:
        raise ValueError(f"{case_id}: sai cấu hình pipeline")
    reference = nib.load(str(raw_dir / meta["ref_image_path"]))
    if list(reference.shape) != meta["orig_shape"] or not np.allclose(
        reference.affine, meta["orig_affine"], rtol=0, atol=1e-6
    ):
        raise ValueError(f"{case_id}: metadata không khớp ảnh gốc")
    z = meta["num_slices_z"]
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"images", "masks", "slice_has_tumor", "slice_has_brain"}:
            raise ValueError(f"{case_id}: sai key NPZ")
        images, masks = archive["images"], archive["masks"]
        if images.shape != (z, 4, 224, 224) or masks.shape != (z, 224, 224):
            raise ValueError(f"{case_id}: sai shape NPZ")
        if images.dtype != np.float16 or masks.dtype != np.uint8:
            raise ValueError(f"{case_id}: sai dtype NPZ")
        if not np.isfinite(images).all() or not np.isin(masks, [0, 1]).all():
            raise ValueError(f"{case_id}: giá trị NPZ không hợp lệ")
        expected_tumor = masks.reshape(z, -1).any(axis=1).astype(np.uint8)
        expected_brain = images.reshape(z, -1).any(axis=1).astype(np.uint8)
        if not np.array_equal(expected_tumor, archive["slice_has_tumor"]):
            raise ValueError(f"{case_id}: cờ slice_has_tumor sai")
        if not np.array_equal(expected_brain, archive["slice_has_brain"]):
            raise ValueError(f"{case_id}: cờ slice_has_brain sai")
        if int(expected_tumor.sum()) != meta["num_tumor_slices"]:
            raise ValueError(f"{case_id}: số lát có u không khớp metadata")
        restored = restore_mask(masks, meta) if roundtrip else None
    report = {"case_id": case_id, "group": meta["group"], "npz_bytes": path.stat().st_size,
              "npz_sha256": file_hash(path), "metadata_sha256": file_hash(meta_path),
              "orig_axcodes": meta["orig_axcodes"], "num_slices_z": z, "valid": True}
    if restored is not None:
        seg_path = raw_dir / meta["group"] / case_id / f"{case_id}-seg.nii.gz"
        raw_mask = np.isin(np.asarray(nib.load(str(seg_path)).dataobj), [1, 2, 3])
        if restored.shape != raw_mask.shape:
            raise ValueError(f"{case_id}: sai shape sau khôi phục")
        denominator = int(restored.sum()) + int(raw_mask.sum())
        report["gt_resize_roundtrip_dice"] = (
            2 * int(np.count_nonzero(restored & raw_mask)) / denominator if denominator else 1.0)
        report["roundtrip_note"] = "Kiểm tra resize/metadata của GT, không phải kết quả mô hình"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    index = json.loads((args.data_dir / "index.json").read_text(encoding="utf-8"))
    ids = [row["case_id"] for row in index]
    if len(ids) != 146 or len(set(ids)) != 146:
        raise ValueError("Index cần đúng 146 ca riêng biệt")
    if {p.stem for p in (args.data_dir / "cases").glob("*.npz")} != set(ids):
        raise ValueError("Thiếu hoặc thừa NPZ")
    if {p.stem for p in (args.data_dir / "metadata").glob("*.json")} != set(ids):
        raise ValueError("Thiếu hoặc thừa metadata")
    samples = set(ids[:1] + ids[-1:])
    for case_id in ids:
        metadata = json.loads((args.data_dir / "metadata" / f"{case_id}.json").read_text())
        if metadata["orig_axcodes"] != ["R", "A", "S"]:
            samples.add(case_id)
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(validate_case, args.data_dir, args.raw_dir, case_id, case_id in samples)
                   for case_id in ids]
        for future in as_completed(futures):
            row = future.result()
            records.append(row)
            print(f"[{len(records)}/146] {row['case_id']} verified", flush=True)
    report = {"num_cases": len(records), "num_valid_cases": len(records),
              "total_npz_bytes": sum(row["npz_bytes"] for row in records),
              "roundtrip_samples": sorted(samples), "pipeline_hash": pipeline_hash(),
              "records": sorted(records, key=lambda row: row["case_id"])}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Đã xác minh {len(records)} NPZ và metadata, {report['total_npz_bytes'] / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
