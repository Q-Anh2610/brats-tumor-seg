"""Thống kê & trực quan hóa dữ liệu BraTS 2020 (EDA - Bước 2 cho báo cáo giữa kỳ)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from src.preprocess.preprocess_nifti import get_spatial_unit_factor_to_mm
from src.preprocess.validate_nifti import CHANNELS, find_case_files

CHANNEL_LABELS = {"t1": "T1", "t1ce": "T1ce", "t2": "T2", "flair": "FLAIR"}


def load_split_map(splits_path: Path | None) -> dict[str, str]:
    """Đọc configs/splits.json để ánh xạ case_id -> train/val/test."""
    if splits_path is None or not splits_path.exists():
        return {}
    data = json.loads(splits_path.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for split_name, ids in data.get("splits", {}).items():
        for cid in ids:
            mapping[cid] = split_name
    return mapping


def plot_tumor_volume_distribution(df: pd.DataFrame, out_path: Path) -> None:
    """Biểu đồ 1: Phân bố thể tích Whole Tumor (cm³) toàn bộ và theo train/val/test."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    sns.histplot(df["wt_volume_ml"], bins=30, kde=True, color="#2563eb", ax=axes[0])
    median_v = df["wt_volume_ml"].median()
    mean_v = df["wt_volume_ml"].mean()
    axes[0].axvline(mean_v, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Mean: {mean_v:.1f} cm³")
    axes[0].axvline(median_v, color="#059669", linestyle="-.", linewidth=1.5, label=f"Median: {median_v:.1f} cm³")
    axes[0].set_title("Phân bố thể tích Whole Tumor (369 ca BraTS 2020)")
    axes[0].set_xlabel("Thể tích khối u (cm³ / mL)")
    axes[0].set_ylabel("Số lượng bệnh nhân")
    axes[0].legend()

    if "split" in df.columns and (df["split"] != "unknown").any():
        order = [s for s in ("train", "val", "test") if s in set(df["split"])]
        sns.boxplot(data=df, x="split", y="wt_volume_ml", order=order, hue="split", legend=False, palette="Set2", ax=axes[1])
        axes[1].set_title("So sánh phân bố thể tích u theo tập Train / Val / Test")
        axes[1].set_xlabel("Tập dữ liệu")
        axes[1].set_ylabel("Thể tích khối u (cm³ / mL)")
    else:
        sns.boxplot(y=df["wt_volume_ml"], color="#60a5fa", ax=axes[1])
        axes[1].set_title("Boxplot thể tích Whole Tumor")
        axes[1].set_ylabel("Thể tích khối u (cm³ / mL)")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_intensity_distributions(sampled_intensities: dict[str, np.ndarray], out_path: Path) -> None:
    """Biểu đồ 2: Phân bố cường độ thô trên vùng não (x != 0) của 4 chuỗi MRI."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    colors = {"t1": "#1d4ed8", "t1ce": "#059669", "t2": "#d97706", "flair": "#dc2626"}

    for ax, ch in zip(axes.flat, CHANNELS):
        vals = sampled_intensities[ch]
        hi = np.percentile(vals, 99.5)
        lo = np.percentile(vals, 0.5)
        clipped = vals[(vals >= lo) & (vals <= hi)]
        sns.histplot(clipped, bins=60, stat="density", kde=True, color=colors[ch], ax=ax)
        ax.set_title(f"Chuỗi {CHANNEL_LABELS[ch]} (vùng não ≠ 0, cắt [0.5%, 99.5%])")
        ax.set_xlabel("Cường độ tín hiệu MRI thô")
        ax.set_ylabel("Mật độ")

    fig.suptitle("Phân bố cường độ trên vùng não của 4 chuỗi MRI – BraTS 2020", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_class_imbalance_and_slices(
    df: pd.DataFrame,
    z_tumor_counts: np.ndarray,
    z_total_counts: np.ndarray,
    out_path: Path,
) -> None:
    """Biểu đồ 3: Mất cân bằng lớp ở mức lát cắt & phân bố vị trí lát có u theo trục Z."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    total_slices = int(df["num_slices_z"].sum())
    tumor_slices = int(df["num_tumor_slices"].sum())
    non_tumor_slices = total_slices - tumor_slices

    total_brain_vox = int(df["brain_voxels"].sum())
    total_wt_vox = int(df["wt_voxels"].sum())
    non_wt_brain_vox = total_brain_vox - total_wt_vox

    categories = [
        "Lát có u\n(mức lát)",
        "Lát không u\n(mức lát)",
        "Voxel u\n(trong não)",
        "Voxel nền não\n(trong não)",
    ]
    pcts = [
        100.0 * tumor_slices / max(total_slices, 1),
        100.0 * non_tumor_slices / max(total_slices, 1),
        100.0 * total_wt_vox / max(total_brain_vox, 1),
        100.0 * non_wt_brain_vox / max(total_brain_vox, 1),
    ]
    bar_colors = ["#dc2626", "#94a3b8", "#ea580c", "#cbd5e1"]
    bars = axes[0].bar(categories, pcts, color=bar_colors, edgecolor="#334155", linewidth=0.6)
    for bar, pct in zip(bars, pcts):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )
    axes[0].set_ylim(0, 112)
    axes[0].set_ylabel("Tỷ lệ (%)")
    axes[0].set_title("Mất cân bằng lớp (mức lát axial & mức voxel vùng não)")

    z_idx = np.arange(len(z_tumor_counts))
    z_ratio = 100.0 * z_tumor_counts / np.maximum(z_total_counts, 1)
    axes[1].plot(z_idx, z_ratio, color="#2563eb", linewidth=2)
    axes[1].fill_between(z_idx, z_ratio, color="#93c5fd", alpha=0.4)
    axes[1].set_title("Tỷ lệ ca có khối u theo chỉ số lát Axial (Z)")
    axes[1].set_xlabel("Chỉ số lát cắt Axial (z)")
    axes[1].set_ylabel("Tỷ lệ bệnh nhân có u (%)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_sample_cases(data_dir: Path, df: pd.DataFrame, out_path: Path) -> None:
    """Biểu đồ 4: Trực quan hóa 3 ca mẫu (u nhỏ, u vừa, u lớn) với 4 chuỗi MRI + GT overlay."""
    sorted_df = df.sort_values("wt_volume_ml").reset_index(drop=True)
    n = len(sorted_df)
    pick_indices = [max(0, int(0.1 * n)), int(0.5 * n), min(n - 1, int(0.9 * n))]
    tags = ["Khối u nhỏ", "Khối u trung bình", "Khối u lớn"]

    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    for row_idx, (df_idx, tag) in enumerate(zip(pick_indices, tags)):
        row = sorted_df.iloc[df_idx]
        cid = row["case_id"]
        files = find_case_files(data_dir / cid)
        seg = np.asarray(nib.load(str(files["seg"])).dataobj)
        wt = np.isin(seg, [1, 2, 4])
        # Chọn lát axial có diện tích u lớn nhất
        z_best = int(np.argmax(wt.sum(axis=(0, 1))))

        for col_idx, ch in enumerate(CHANNELS):
            vol = np.asarray(nib.load(str(files[ch])).dataobj, dtype=np.float32)
            sl = np.rot90(vol[:, :, z_best])
            p99 = np.percentile(sl[sl > 0], 99) if np.any(sl > 0) else 1.0
            axes[row_idx, col_idx].imshow(np.clip(sl / max(p99, 1e-6), 0, 1), cmap="gray")
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(CHANNEL_LABELS[ch], fontsize=11, fontweight="bold")
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel(
                    f"{tag}\n{cid}\n(z={z_best}, {row['wt_volume_ml']:.1f} cm³)",
                    fontsize=9,
                )
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])

        # Cột thứ 5: FLAIR + WT Ground Truth overlay
        flair = np.asarray(nib.load(str(files["flair"])).dataobj, dtype=np.float32)
        flair_sl = np.rot90(flair[:, :, z_best])
        wt_sl = np.rot90(wt[:, :, z_best])
        p99 = np.percentile(flair_sl[flair_sl > 0], 99) if np.any(flair_sl > 0) else 1.0
        axes[row_idx, 4].imshow(np.clip(flair_sl / max(p99, 1e-6), 0, 1), cmap="gray")
        axes[row_idx, 4].imshow(np.ma.masked_where(~wt_sl, wt_sl), cmap="autumn", alpha=0.55)
        if row_idx == 0:
            axes[row_idx, 4].set_title("FLAIR + WT Mask", fontsize=11, fontweight="bold")
        axes[row_idx, 4].set_xticks([])
        axes[row_idx, 4].set_yticks([])

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_eda(data_dir: Path, splits_path: Path | None, out_dir: Path, seed: int = 42) -> None:
    """Chạy toàn bộ thống kê EDA trên BraTS 2020 và xuất CSV/JSON + 4 biểu đồ."""
    split_map = load_split_map(splits_path)
    case_dirs = sorted(
        p for p in data_dir.iterdir()
        if p.is_dir() and p.name.startswith("BraTS20_Training_")
    )
    if not case_dirs:
        raise RuntimeError(f"Không tìm thấy ca BraTS20_Training_* trong {data_dir}")

    rng = np.random.default_rng(seed)
    sampled_intensities: dict[str, list[np.ndarray]] = {ch: [] for ch in CHANNELS}
    z_tumor_counts = None
    z_total_counts = None
    records = []

    for cd in tqdm(case_dirs, desc="EDA BraTS 2020"):
        cid = cd.name
        files = find_case_files(cd)
        ref = nib.load(str(files["t1"]))
        _, unit_scale = get_spatial_unit_factor_to_mm(ref)
        zooms_mm = [float(z) * unit_scale for z in ref.header.get_zooms()[:3]]
        voxel_vol_mm3 = zooms_mm[0] * zooms_mm[1] * zooms_mm[2]

        seg = np.asarray(nib.load(str(files["seg"])).dataobj, dtype=np.int16)
        wt_mask = np.isin(seg, [1, 2, 4])

        # Load 4 kênh để tạo hợp vùng não và tính thống kê từng kênh
        vols = {ch: np.asarray(nib.load(str(files[ch])).dataobj, dtype=np.float32) for ch in CHANNELS}
        brain_mask = (vols["t1"] != 0) | (vols["t1ce"] != 0) | (vols["t2"] != 0) | (vols["flair"] != 0)

        num_z = int(seg.shape[2])
        if z_tumor_counts is None:
            z_tumor_counts = np.zeros(num_z, dtype=np.int64)
            z_total_counts = np.zeros(num_z, dtype=np.int64)
        if num_z == len(z_tumor_counts):
            z_tumor_counts += (wt_mask.sum(axis=(0, 1)) > 0).astype(np.int64)
            z_total_counts += 1

        brain_vox = int(brain_mask.sum())
        wt_vox = int(wt_mask.sum())
        num_brain_slices = int((brain_mask.sum(axis=(0, 1)) > 0).sum())
        num_tumor_slices = int((wt_mask.sum(axis=(0, 1)) > 0).sum())

        row = {
            "case_id": cid,
            "split": split_map.get(cid, "unknown"),
            "shape_x": int(seg.shape[0]),
            "shape_y": int(seg.shape[1]),
            "num_slices_z": num_z,
            "spacing_x_mm": zooms_mm[0],
            "spacing_y_mm": zooms_mm[1],
            "spacing_z_mm": zooms_mm[2],
            "brain_voxels": brain_vox,
            "wt_voxels": wt_vox,
            "wt_volume_ml": (wt_vox * voxel_vol_mm3) / 1000.0,
            "wt_ratio_in_brain": wt_vox / max(brain_vox, 1),
            "num_brain_slices": num_brain_slices,
            "num_tumor_slices": num_tumor_slices,
            "tumor_slice_ratio": num_tumor_slices / max(num_z, 1),
            "label1_voxels": int((seg == 1).sum()),
            "label2_voxels": int((seg == 2).sum()),
            "label4_voxels": int((seg == 4).sum()),
        }

        for ch in CHANNELS:
            v = vols[ch]
            b_vals = v[v != 0]
            if b_vals.size > 0:
                row[f"{ch}_mean_brain"] = float(b_vals.mean())
                row[f"{ch}_std_brain"] = float(b_vals.std())
                row[f"{ch}_p05_brain"] = float(np.percentile(b_vals, 5))
                row[f"{ch}_p50_brain"] = float(np.percentile(b_vals, 50))
                row[f"{ch}_p95_brain"] = float(np.percentile(b_vals, 95))
                sample_n = min(1500, b_vals.size)
                idx = rng.choice(b_vals.size, size=sample_n, replace=False)
                sampled_intensities[ch].append(b_vals[idx])
            else:
                for k in ("mean_brain", "std_brain", "p05_brain", "p50_brain", "p95_brain"):
                    row[f"{ch}_{k}"] = 0.0

            wt_vals = v[wt_mask]
            row[f"{ch}_mean_wt"] = float(wt_vals.mean()) if wt_vals.size > 0 else 0.0
            row[f"{ch}_std_wt"] = float(wt_vals.std()) if wt_vals.size > 0 else 0.0

        records.append(row)

    df = pd.DataFrame(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "eda_brats2020_cases.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    total_slices = int(df["num_slices_z"].sum())
    tumor_slices = int(df["num_tumor_slices"].sum())
    summary = {
        "dataset": "BraTS2020",
        "num_cases": int(len(df)),
        "total_slices": total_slices,
        "total_tumor_slices": tumor_slices,
        "tumor_slice_pct": round(100.0 * tumor_slices / max(total_slices, 1), 2),
        "wt_voxel_pct_in_brain": round(
            100.0 * float(df["wt_voxels"].sum()) / max(float(df["brain_voxels"].sum()), 1.0), 2
        ),
        "wt_volume_ml": {
            "mean": round(float(df["wt_volume_ml"].mean()), 2),
            "std": round(float(df["wt_volume_ml"].std(ddof=1)), 2),
            "median": round(float(df["wt_volume_ml"].median()), 2),
            "min": round(float(df["wt_volume_ml"].min()), 2),
            "max": round(float(df["wt_volume_ml"].max()), 2),
        },
        "num_tumor_slices_per_case": {
            "mean": round(float(df["num_tumor_slices"].mean()), 2),
            "std": round(float(df["num_tumor_slices"].std(ddof=1)), 2),
            "min": int(df["num_tumor_slices"].min()),
            "max": int(df["num_tumor_slices"].max()),
        },
    }
    (out_dir / "eda_brats2020_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    merged_samples = {ch: np.concatenate(sampled_intensities[ch]) for ch in CHANNELS}
    plot_tumor_volume_distribution(df, fig_dir / "eda_01_tumor_volume_dist.png")
    plot_intensity_distributions(merged_samples, fig_dir / "eda_02_intensity_distributions.png")
    plot_class_imbalance_and_slices(
        df, z_tumor_counts, z_total_counts, fig_dir / "eda_03_class_imbalance_slices.png"
    )
    plot_sample_cases(data_dir, df, fig_dir / "eda_04_sample_cases.png")

    print("Hoàn tất EDA BraTS 2020:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"- Bảng từng ca (gửi B để so sánh BraTS-Africa): {csv_path}")
    print(f"- 4 biểu đồ lưu tại: {fig_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA BraTS 2020 NIfTI 3D")
    parser.add_argument("--data-dir", type=Path, required=True, help="Thư mục chứa BraTS20_Training_*")
    parser.add_argument("--splits", type=Path, default=Path("configs/splits.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    run_eda(args.data_dir, args.splits, args.out_dir)


if __name__ == "__main__":
    main()
