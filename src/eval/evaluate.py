"""Đánh giá mask WT 3D trên lưới gốc, xuất CSV theo ca, tổng hợp và thông tin lần chạy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from src.eval.metrics_3d import metrics_3d
from src.eval.nifti_io import load_case, read_split


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    """Ghi các bản ghi thành CSV UTF-8, giữ tên cột và giá trị thô."""
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summary_rows(rows: list[dict]) -> list[dict]:
    """Tổng hợp mỗi metric theo ca với SD mẫu ddof=1.

    Nếu có HD95 vô hạn, mean_all là inf và sd_all là N/A.
    Nhóm hữu hạn được báo riêng cùng số lượng, không âm thầm bỏ ca thất bại."""
    output = []
    for metric in ("dice", "iou", "hd95_mm"):
        values = np.asarray([r[metric] for r in rows], dtype=float)
        finite = values[np.isfinite(values)]
        n, nf = len(values), len(finite)
        output.append({
            "metric": metric, "n_total": n, "n_finite": nf,
            "n_infinite": n - nf,
            "mean_all": float(values.mean()) if nf == n else "inf",
            "sd_all": float(values.std(ddof=1)) if nf == n and n > 1 else "N/A",
            "mean_finite": float(finite.mean()) if nf else "N/A",
            "sd_finite": float(finite.std(ddof=1)) if nf > 1 else "N/A",
        })
    return output


def sha256(path: Path) -> str:
    """Tạo mã băm nội dung file để nhận biết thay đổi dữ liệu giữa các lần chạy."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_output(*args: str) -> str | None:
    """Lấy commit/trạng thái Git nếu có, không tạo commit và không thay đổi repository."""
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    """Đọc tham số, kiểm tra các ca trong split và tính metric từng khối 3D.

    --check-only chỉ kiểm tra đầu vào, không tính HD95 hay ghi bảng điểm.
    --case-id chạy thử một ca, manifest đánh dấu rõ đây là tập con.
    Chỉ ghi bảng mới khi toàn bộ ca được yêu cầu đã xử lý thành công."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--splits", type=Path, default=Path("configs/splits.json"))
    parser.add_argument("--split-name", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", type=Path, default=Path("results/eval/unetpp_test"))
    parser.add_argument("--case-id", help="One case for smoke testing, output is explicitly marked subset.")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--assume-unknown-mm", action="store_true",
                        help="Use only after confirming unknown-unit files are in mm from the data source.")
    args = parser.parse_args()
    # 1. Lấy đúng danh sách đánh giá đã khóa trong cấu hình.
    full_ids = read_split(args.splits, args.split_name)
    if args.case_id and args.case_id not in full_ids:
        parser.error("--case-id is not in the requested split.")
    ids = [args.case_id] if args.case_id else full_ids
    if not args.pred_dir.is_dir():
        parser.error(f"Prediction directory not found: {args.pred_dir}")
    observed = {p.name[:-len("_pred_wt.nii.gz")] for p in args.pred_dir.glob("*_pred_wt.nii.gz")}
    missing, extra = set(ids) - observed, observed - set(full_ids)
    if missing or extra:
        parser.error(f"Missing predictions: {sorted(missing)}, IDs outside split: {sorted(extra)}")
    # 2. Xử lý theo ca, nếu có lỗi thì không ghi bảng kết quả mới.
    rows, audit, errors = [], [], []
    for index, cid in enumerate(ids, 1):
        try:
            gi, pi, gt, pred, spacing, gp, pp = load_case(
                args.raw_dir, args.pred_dir, cid, args.assume_unknown_mm)
            if not args.check_only:
                rows.append({"case_id": cid, **metrics_3d(pred, gt, spacing)})
            audit.append({
                "case_id": cid, "shape": list(gi.shape), "spacing_mm": spacing.tolist(),
                "gt_unit": gi.header.get_xyzt_units()[0],
                "pred_unit": pi.header.get_xyzt_units()[0],
                "gt_path": str(gp.resolve()), "pred_path": str(pp.resolve()),
                **({} if args.check_only else {"gt_sha256": sha256(gp), "pred_sha256": sha256(pp)}),
            })
            print(f"[{index}/{len(ids)}] OK {cid}", flush=True)
        except (ValueError, OSError, RuntimeError) as exc:
            errors.append(f"{cid}: {exc}")
            print(f"[ERROR] {errors[-1]}", flush=True)
    if errors:
        raise SystemExit("FAILED: no new result files written. Fix all errors:\n" + "\n".join(errors))
    if args.check_only:
        print(f"Preflight passed: {len(ids)} cases. No metrics written.")
        return
    # 3. Lưu bảng theo ca, bảng tổng hợp và thông tin tái lập lần chạy.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["case_id", "dice", "iou", "hd95_mm", "gt_empty", "pred_empty"]
    write_csv(args.out_dir / "per_case.csv", rows, fields)
    summary = summary_rows(rows)
    write_csv(args.out_dir / "summary.csv", summary, list(summary[0]))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv, "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--porcelain"),
        "python": platform.python_version(),
        "versions": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "nibabel")},
        "split_name": args.split_name, "split_sha256": sha256(args.splits),
        "scope": "full_split" if len(ids) == len(full_ids) else "subset_smoke_test",
        "n_cases": len(ids), "expected_split_cases": len(full_ids),
        "assume_unknown_mm": args.assume_unknown_mm,
        "metrics": {"hd95": "max(p95(P->G),p95(G->P)), linear percentile, 6-connectivity",
                    "sd_ddof": 1, "grid": "original", "threshold": "already binary, no new threshold"},
        "n_gt_empty": sum(r["gt_empty"] for r in rows),
        "n_pred_empty": sum(r["pred_empty"] for r in rows),
        "inputs": audit,
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for row in summary:
        print(row)
    print(f"Saved {len(rows)} cases ({manifest['scope']}) -> {args.out_dir}")


if __name__ == "__main__":
    main()
