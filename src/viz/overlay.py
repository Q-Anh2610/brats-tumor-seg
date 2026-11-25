"""Vẽ MRI, GT và dự đoán cùng lát axial RAS+, xuất PNG cùng thông tin lựa chọn ca/lát."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np
from src.eval.nifti_io import checked_array, find_raw_file, load_case, same_grid


def select_examples(rows: list[dict], per_group: int = 2) -> list[tuple]:
    """Chọn các ca Dice thấp, quanh trung vị và cao, ba nhóm không trùng bệnh nhân."""
    ordered = sorted(rows, key=lambda r: (float(r["dice"]), r["case_id"]))
    if len(ordered) < 3 * per_group:
        raise ValueError("Not enough cases for three disjoint groups, use --case-id or reduce --per-group.")
    low = ordered[:per_group]
    high = ordered[-per_group:]
    middle_pool = ordered[per_group:-per_group]
    start = (len(middle_pool) - per_group) // 2
    middle = middle_pool[start:start + per_group]
    return [(name, r) for name, group in (("low", low), ("median", middle), ("high", high)) for r in group]


def draw_case(raw_dir, pred_dir, cid, out_dir, channel="flair", assume_unknown_mm=False,
              slice_index=None, dice=None, group="manual"):
    """Kiểm tra lưới rồi đưa MRI/GT/dự đoán về cùng hướng để hiển thị.

    Không nội suy mask. Mặc định chọn lát có diện tích GT lớn nhất,
    nếu GT rỗng thì xét dự đoán, sau đó vùng ảnh khác 0.
    Mọi ô dùng chung độ sáng và tỷ lệ không gian. Dice trên tiêu đề là 3D.
    slice_index là chỉ số axial sau RAS+, không phải chỉ số ảnh gốc."""
    gi, pi, gt, pred, _, _, _ = load_case(raw_dir, pred_dir, cid, assume_unknown_mm)
    mri = nib.load(str(find_raw_file(raw_dir, cid, channel)))
    same_grid(gi, mri, assume_unknown_mm)
    checked_array(mri)
    # Đổi hướng đồng bộ ba volume để hiển thị, không nội suy lại mask.
    transform = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(gi.affine), nib.orientations.axcodes2ornt(("R", "A", "S")))
    mri_ras = nib.as_closest_canonical(mri)
    direction = mri_ras.affine[:3, :3]
    direction = direction / np.linalg.norm(direction, axis=0)
    if not np.allclose(direction, np.eye(3), rtol=0, atol=1e-4):
        raise ValueError("Oblique MRI: this axial viewer requires a separate resampling review.")
    image = np.asarray(mri_ras.dataobj, dtype=np.float32)
    gt = nib.orientations.apply_orientation(gt, transform)
    pred = nib.orientations.apply_orientation(pred, transform)
    if slice_index is None:
        area = gt.sum(axis=(0, 1))
        if not area.any():
            area = pred.sum(axis=(0, 1))
        if not area.any():
            area = np.count_nonzero(image, axis=(0, 1))
        z = int(area.argmax())
        selection = "largest GT area, fallback prediction, then brain"
    else:
        z = slice_index
        selection = "manual RAS+ axial index"
    if not 0 <= z < image.shape[2]:
        raise ValueError(f"RAS+ slice must be in [0, {image.shape[2] - 1}].")
    sl = image[:, :, z].T
    gs, ps = gt[:, :, z].T, pred[:, :, z].T
    nonzero = sl[sl != 0]
    vmin, vmax = np.percentile(nonzero, [1, 99]) if nonzero.size else (0., 1.)
    if vmax <= vmin:
        vmax = vmin + 1
    spacing = np.linalg.norm(mri_ras.affine[:3, :3], axis=0)
    extent = (0, image.shape[0] * spacing[0], 0, image.shape[1] * spacing[1])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), layout="constrained")
    for ax, title in zip(axes, ("MRI", "MRI + Ground truth WT", "MRI + Prediction WT")):
        ax.imshow(sl, cmap="gray", origin="lower", vmin=vmin, vmax=vmax,
                  extent=extent, interpolation="nearest")
        ax.set_title(title)
        ax.set_axis_off()
    axes[1].imshow(np.ma.masked_where(~gs, gs), cmap=ListedColormap(["#29cc72"]),
                   alpha=.45, vmin=0, vmax=1, origin="lower", extent=extent, interpolation="nearest")
    axes[2].imshow(np.ma.masked_where(~ps, ps), cmap=ListedColormap(["#ff5773"]),
                   alpha=.45, vmin=0, vmax=1, origin="lower", extent=extent, interpolation="nearest")
    score = f" | 3D Dice={float(dice):.4f}" if dice is not None else ""
    fig.suptitle(f"{cid} | {channel.upper()} | RAS+ axial z={z}{score}\n"
                 "Neurological view: image left=patient L, right=R, top=Anterior, bottom=Posterior")
    fig.legend(handles=[Patch(color="#29cc72", label="Ground truth"),
                        Patch(color="#ff5773", label="Prediction")],
               loc="outside lower center", ncol=2)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{group}_{cid}_ras_z{z:03d}.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return {"case_id": cid, "group": group, "ras_axial_slice": z, "selection": selection,
            "dice_3d": None if dice is None else float(dice), "channel": channel, "figure": target.name}


def main() -> None:
    """Đọc CSV/chọn ca thủ công, vẽ overlay và lưu selection.json để truy vết."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path)
    parser.add_argument("--case-id")
    parser.add_argument("--slice-index", type=int, help="RAS+ axial z, not original LPS metadata index.")
    parser.add_argument("--per-group", type=int, default=2)
    parser.add_argument("--channel", choices=["t1", "t1ce", "t2", "flair"], default="flair")
    parser.add_argument("--assume-unknown-mm", action="store_true")
    args = parser.parse_args()
    if args.slice_index is not None and not args.case_id:
        parser.error("--slice-index requires --case-id.")
    if args.per_group < 1:
        parser.error("--per-group must be positive.")
    rows = []
    if args.metrics_csv:
        with args.metrics_csv.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if len({r["case_id"] for r in rows}) != len(rows):
            parser.error("Duplicate case IDs in metrics CSV.")
        if any(not np.isfinite(float(r["dice"])) or not 0 <= float(r["dice"]) <= 1 for r in rows):
            parser.error("Invalid Dice values.")
    if args.case_id:
        score = next((r["dice"] for r in rows if r["case_id"] == args.case_id), None)
        selected = [("manual", {"case_id": args.case_id, "dice": score})]
    else:
        if not rows:
            parser.error("Provide --metrics-csv for automatic examples or --case-id for one case.")
        selected = select_examples(rows, args.per_group)
    records = []
    for group, row in selected:
        cid = row["case_id"]
        if Path(cid).name != cid or not cid.startswith("BraTS20_Training_"):
            parser.error("Invalid case ID.")
        record = draw_case(args.raw_dir, args.pred_dir, cid, args.out_dir, args.channel,
                           args.assume_unknown_mm, args.slice_index, row["dice"], group)
        records.append(record)
        print(f"Saved {record['figure']}")
    (args.out_dir / "selection.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
