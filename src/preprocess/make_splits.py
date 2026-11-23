"""Tạo file configs/splits.json theo Quy ước chung v1.1 (seed=42, 70/15/15 theo bệnh nhân)."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def collect_case_ids_from_nifti_dir(data_dir: Path) -> list[str]:
    """Quét các thư mục BraTS20_Training_* trong thư mục NIfTI."""
    case_ids = sorted(
        p.name
        for p in data_dir.iterdir()
        if p.is_dir() and p.name.startswith("BraTS20_Training_")
    )
    return case_ids


def collect_case_ids_from_name_mapping(csv_path: Path) -> list[str]:
    """Đọc danh sách BraTS_2020_subject_ID từ name_mapping.csv."""
    case_ids: list[str] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = (row.get("BraTS_2020_subject_ID") or "").strip()
            if cid:
                case_ids.append(cid)
    return sorted(set(case_ids))


def create_splits(
    case_ids: list[str],
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    excluded_cases: list[str] | None = None,
) -> dict:
    """Sắp xếp ID tăng dần, loại ca lỗi (nếu có), xáo trộn với seed cố định và chia 70/15/15."""
    excluded = set(excluded_cases or [])
    valid_ids = sorted(cid for cid in case_ids if cid not in excluded)

    rng = random.Random(seed)
    shuffled = list(valid_ids)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_val = round(n_total * val_ratio)
    n_test = round(n_total * test_ratio)
    n_train = n_total - n_val - n_test

    train_ids = sorted(shuffled[:n_train])
    val_ids = sorted(shuffled[n_train : n_train + n_val])
    test_ids = sorted(shuffled[n_train + n_val :])

    # Kiểm tra không trùng lặp giữa các tập
    assert set(train_ids).isdisjoint(val_ids)
    assert set(train_ids).isdisjoint(test_ids)
    assert set(val_ids).isdisjoint(test_ids)
    assert len(train_ids) + len(val_ids) + len(test_ids) == n_total

    return {
        "dataset": "BraTS2020",
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "counts": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        "excluded_cases": sorted(excluded),
        "splits": {"train": train_ids, "val": val_ids, "test": test_ids},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tạo configs/splits.json theo Quy ước v1.1")
    parser.add_argument("--nifti-dir", type=Path, default=None, help="Thư mục chứa BraTS20_Training_*")
    parser.add_argument("--name-mapping", type=Path, default=None, help="Đường dẫn name_mapping.csv")
    parser.add_argument("--output", type=Path, default=Path("configs/splits.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.nifti_dir is not None:
        case_ids = collect_case_ids_from_nifti_dir(args.nifti_dir)
    elif args.name_mapping is not None:
        case_ids = collect_case_ids_from_name_mapping(args.name_mapping)
    else:
        raise ValueError("Cần truyền --nifti-dir hoặc --name-mapping.")

    if not case_ids:
        raise RuntimeError("Không tìm thấy case_id nào.")

    payload = create_splits(case_ids, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Đã tạo {args.output}: {payload['counts']} (tổng {sum(payload['counts'].values())} ca)")


if __name__ == "__main__":
    main()
