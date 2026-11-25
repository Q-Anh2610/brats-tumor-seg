"""Kiểm tra dữ liệu gốc/dự đoán cho eval và lưu thông tin đầu vào."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib

from src.eval.nifti_io import checked_array, find_raw_file, read_split


def discover_raw_dir(download_dir: Path) -> Path:
    """Tìm thư mục chứa trực tiếp các ca training, không đoán số tầng thư mục."""
    prefix = "BraTS20_Training_"
    cases = [p for p in download_dir.rglob(prefix + "*")
             if p.is_dir() and p.name[len(prefix):].isdigit()]
    parents = sorted({p.parent for p in cases})
    if len(parents) != 1:
        raise ValueError(f"Cần đúng một thư mục training, tìm thấy: {parents}")
    return parents[0]


def audit_inputs(raw_dir: Path, pred_dir: Path, case_ids: list[str]) -> list[dict]:
    """Kiểm tra đủ GT/FLAIR/dự đoán, trả đường dẫn từng ca, chưa tính metric."""
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục dự đoán: {pred_dir}")
    observed = {p.name[:-len("_pred_wt.nii.gz")] for p in pred_dir.glob("*_pred_wt.nii.gz")}
    missing, extra = set(case_ids) - observed, observed - set(case_ids)
    if missing or extra:
        raise ValueError(f"Thiếu dự đoán: {sorted(missing)}, ID ngoài split: {sorted(extra)}")
    records = []
    for cid in case_ids:
        records.append({"case_id": cid,
                        "gt": str(find_raw_file(raw_dir, cid, "seg").resolve()),
                        "flair": str(find_raw_file(raw_dir, cid, "flair").resolve()),
                        "prediction": str((pred_dir / f"{cid}_pred_wt.nii.gz").resolve())})
    return records


def main() -> None:
    """Kiểm tra file, xem header/nhãn ca mẫu và lưu JSON phục vụ bước eval."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-dir", type=Path, default=Path("/content/brats2020_raw"))
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--splits", type=Path, default=Path("configs/splits.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/input_checks"))
    args = parser.parse_args()

    # 1. Xác định thư mục training và danh sách test đã khóa.
    raw_dir = discover_raw_dir(args.download_dir)
    case_ids = read_split(args.splits, "test")
    if len(case_ids) != 55:
        raise ValueError(f"Đang có {len(case_ids)} ca test, đối chiếu lại phiên bản splits.json.")
    records = audit_inputs(raw_dir, args.pred_dir, case_ids)

    # 2. In thông tin một ca để xem đơn vị trước khi chạy kiểm tra hình học đầy đủ.
    sample = records[0]
    for role, allowed in (("gt", [0, 1, 2, 4]), ("flair", None), ("prediction", [0, 1])):
        image = nib.load(sample[role])
        checked_array(image, allowed)
        print(role, sample[role])
        print("shape:", image.shape, "spacing:", image.header.get_zooms()[:3],
              "unit:", image.header.get_xyzt_units()[0])
        print("affine:", image.affine)

    # 3. Lưu đường dẫn/hash split, không gán giả định unknown=mm ở bước chuẩn bị.
    record = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "raw_dir_runtime_only": str(raw_dir.resolve()),
              "pred_dir": str(args.pred_dir.resolve()), "test_count": len(case_ids),
              "split_sha256": hashlib.sha256(args.splits.read_bytes()).hexdigest(),
              "scope": "file_presence_and_one_case_header_check",
              "geometry_checked_all_cases": False, "inputs": records}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    target = args.out_dir / "inputs_checked.json"
    target.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Đủ file của {len(case_ids)} ca. Tiếp tục chạy eval --check-only.")
    print("RAW_DIR:", raw_dir)
    print("Đã lưu:", target)


if __name__ == "__main__":
    main()
