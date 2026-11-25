"""Các độ đo phân đoạn WT 3D theo từng bệnh nhân, tuân theo Quy ước v1.1."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


def metrics_3d(pred: np.ndarray, gt: np.ndarray, spacing_mm) -> dict:
    """Tính Dice, IoU và HD95 (mm) từ hai mask nhị phân cùng lưới.

    pred, gt: mảng [X, Y, Z], chỉ nhận giá trị 0/1.
    spacing_mm: khoảng cách voxel theo ba trục, đã đổi về mm.
    Kết quả gồm ba metric và cờ gt_empty/pred_empty.
    HD95 lấy max của percentile 95 từng chiều, không gộp hai tập khoảng cách."""
    # 1. Kiểm tra đầu vào trước khi tính, tránh tính nhầm trên xác suất.
    pred, gt = np.asarray(pred), np.asarray(gt)
    if pred.ndim != 3 or pred.shape != gt.shape:
        raise ValueError("Masks must have the same 3D shape.")
    if not np.isin(pred, [0, 1]).all() or not np.isin(gt, [0, 1]).all():
        raise ValueError("Metrics accept binary masks only.")
    spacing = np.asarray(spacing_mm, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("Invalid spacing in mm.")
    pred, gt = pred.astype(bool), gt.astype(bool)
    p, g = int(pred.sum()), int(gt.sum())
    result = {"gt_empty": int(g == 0), "pred_empty": int(p == 0)}
    # 2. Áp dụng quy ước mask rỗng thay vì thêm epsilon vào metric.
    if not p or not g:
        both = not p and not g
        return {**result, "dice": float(both), "iou": float(both),
                "hd95_mm": 0.0 if both else float("inf")}
    # 3. Dice/IoU dùng tổng voxel của cả khối, không lấy trung bình từng lát.
    inter = int(np.count_nonzero(pred & gt))
    result.update(dice=2 * inter / (p + g), iou=inter / (p + g - inter))
    # 4. Trích biên và tính HD95 bằng khoảng cách vật lý theo mm.
    structure = generate_binary_structure(3, 1)  # Biên 6 hướng, ngoài thể tích được xem là nền.
    bp = pred & ~binary_erosion(pred, structure=structure, border_value=0)
    bg = gt & ~binary_erosion(gt, structure=structure, border_value=0)
    # Tính khoảng cách hai chiều theo spacing, giải phóng mảng tạm để giảm RAM.
    dist = distance_transform_edt(~bg, sampling=spacing)
    pg95 = np.percentile(dist[bp], 95, method="linear")
    del dist
    dist = distance_transform_edt(~bp, sampling=spacing)
    gp95 = np.percentile(dist[bg], 95, method="linear")
    result["hd95_mm"] = float(max(pg95, gp95))
    return result
