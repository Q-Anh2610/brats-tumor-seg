"""Suy luận lát 2D và khôi phục mask 3D về lưới gốc <case_id>_pred_wt.nii.gz theo Quy ước v1.1 Mục 9."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
from nibabel.orientations import apply_orientation, axcodes2ornt, io_orientation, ornt_transform
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.datasets.brats2d_dataset import prepare_mmap_cache
from src.models.unetpp import build_unetpp_2d


def restore_case_probabilities_to_orig_grid(
    probs_224: np.ndarray,
    meta: dict,
) -> np.ndarray:
    """
    Khôi phục bản đồ xác suất [Z, 224, 224] về lưới không gian gốc (trước khi áp ngưỡng 0.5).
    Thứ tự Quy ước v1.1:
      224x224 -> resize bilinear về padded_shape_xy -> bỏ padding -> đặt vào crop_bbox
      của ras_shape -> ghép theo Z -> đảo phép đổi hướng về orig_shape.
    """
    padded_h, padded_w = [int(v) for v in meta["padded_shape_xy"]]
    pad_xy = meta["pad_xy"]
    x_before = int(pad_xy["x_before"])
    x_after = int(pad_xy["x_after"])
    y_before = int(pad_xy["y_before"])
    y_after = int(pad_xy["y_after"])

    bbox = meta["crop_bbox"]
    x_start, x_end = int(bbox["x_start"]), int(bbox["x_end"])
    y_start, y_end = int(bbox["y_start"]), int(bbox["y_end"])
    ras_shape = tuple(int(v) for v in meta["ras_shape"])
    orig_shape = tuple(int(v) for v in meta["orig_shape"])

    # 1. Nội suy bilinear từ 224x224 về padded_shape_xy
    t = torch.from_numpy(probs_224).unsqueeze(1).float()  # [Z, 1, 224, 224]
    t_padded = F.interpolate(t, size=(padded_h, padded_w), mode="bilinear", align_corners=False)
    arr_padded = t_padded.squeeze(1).numpy()  # [Z, padded_h, padded_w]

    # 2. Bỏ padding (tránh lỗi slice [:-0] khi after == 0)
    x_slice_end = padded_h - x_after
    y_slice_end = padded_w - y_after
    arr_crop = arr_padded[:, x_before:x_slice_end, y_before:y_slice_end]  # [Z, crop_h, crop_w]

    # 3. Đặt lại vào vùng crop của volume RAS+ (ngoài vùng crop đặt xác suất 0)
    prob_ras = np.zeros(ras_shape, dtype=np.float32)
    prob_ras[x_start:x_end, y_start:y_end, :] = np.transpose(arr_crop, (1, 2, 0))

    # 4. Đảo phép đổi hướng từ RAS+ về lưới gốc
    orig_affine = np.array(meta["orig_affine"], dtype=np.float64)
    orig_ornt = io_orientation(orig_affine)
    ras_ornt_target = axcodes2ornt(("R", "A", "S"))
    inv_ornt = ornt_transform(ras_ornt_target, orig_ornt)
    prob_orig = apply_orientation(prob_ras, inv_ornt)

    if prob_orig.shape != orig_shape:
        raise RuntimeError(f"{meta['case_id']}: Shape sau khôi phục {prob_orig.shape} != {orig_shape}")

    return prob_orig


@torch.no_grad()
def predict_case_probs(
    model: torch.nn.Module,
    img_mmap: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Dự đoán toàn bộ Z lát của 1 ca -> bản đồ xác suất [Z, 224, 224]."""
    num_z = img_mmap.shape[0]
    probs_list = []
    for start in range(0, num_z, batch_size):
        end = min(start + batch_size, num_z)
        batch = torch.from_numpy(np.array(img_mmap[start:end], dtype=np.float32)).to(device)
        logits = model(batch)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        probs_list.append(probs)
    return np.concatenate(probs_list, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dự đoán và khôi phục mask 3D cho tập test/val")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True, help="Thư mục chứa cases/ và metadata/")
    parser.add_argument("--cache-dir", type=Path, default=Path("/content/brats_npy_cache"))
    parser.add_argument("--splits", type=Path, default=Path("configs/splits.json"))
    parser.add_argument("--split-name", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", type=Path, required=True, help="Thư mục lưu <case_id>_pred_wt.nii.gz")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    m_cfg = cfg["model"]
    threshold = float(cfg["evaluation"]["threshold"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unetpp_2d(
        encoder_name=m_cfg["encoder_name"],
        encoder_weights=None,
        in_channels=int(m_cfg["in_channels"]),
        classes=int(m_cfg["classes"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    splits_data = json.loads(args.splits.read_text(encoding="utf-8"))
    case_ids: list[str] = splits_data["splits"][args.split_name]
    prepare_mmap_cache(args.data_dir, args.cache_dir, case_ids)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = args.data_dir / "metadata"

    summary_rows = []
    for cid in tqdm(case_ids, desc=f"Predicting 3D ({args.split_name})"):
        meta = json.loads((meta_dir / f"{cid}.json").read_text(encoding="utf-8"))
        img_mmap = np.load(args.cache_dir / f"{cid}_img.npy", mmap_mode="r")
        msk_mmap = np.load(args.cache_dir / f"{cid}_msk.npy", mmap_mode="r")

        probs_224 = predict_case_probs(model, img_mmap, device=device, batch_size=args.batch_size)
        prob_orig = restore_case_probabilities_to_orig_grid(probs_224, meta)

        # Chỉ áp ngưỡng sau khi khôi phục về lưới gốc
        pred_wt_orig = (prob_orig >= threshold).astype(np.uint8)
        orig_affine = np.array(meta["orig_affine"], dtype=np.float64)
        out_nii = nib.Nifti1Image(pred_wt_orig, affine=orig_affine)
        out_nii.set_data_dtype(np.uint8)
        nib.save(out_nii, str(args.out_dir / f"{cid}_pred_wt.nii.gz"))

        # Tính sơ bộ Dice 224 để kiểm tra nhanh trước khi chạy module eval chính thức của B
        pred_224 = (probs_224 >= threshold).astype(np.uint8)
        gt_224 = (np.array(msk_mmap) >= 1).astype(np.uint8)
        inter = float((pred_224 & gt_224).sum())
        p_sum, g_sum = float(pred_224.sum()), float(gt_224.sum())
        quick_dice = 1.0 if (p_sum == 0 and g_sum == 0) else (0.0 if (p_sum == 0 or g_sum == 0) else (2.0 * inter) / (p_sum + g_sum))

        summary_rows.append(
            {
                "case_id": cid,
                "pred_voxels_orig": int(pred_wt_orig.sum()),
                "pred_empty": int(pred_wt_orig.sum() == 0),
                "quick_dice_224": round(quick_dice, 5),
            }
        )

    csv_path = args.out_dir / f"pred_summary_{args.split_name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    dices = [r["quick_dice_224"] for r in summary_rows]
    n_empty = sum(r["pred_empty"] for r in summary_rows)
    print(f"Đã xuất {len(case_ids)} file *_pred_wt.nii.gz vào {args.out_dir}")
    print(f"Quick check (lưới 224): Dice = {np.mean(dices):.4f} ± {np.std(dices, ddof=1):.4f} | Ca dự đoán rỗng: {n_empty}/{len(case_ids)}")
    print("Lưu ý: Báo cáo chính thức sẽ chạy 55 file .nii.gz này qua module eval/viz của B để tính Dice, IoU, HD95 trên lưới gốc.")


if __name__ == "__main__":
    main()
