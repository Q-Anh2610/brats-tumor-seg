"""Vẽ EDA và so sánh BraTS-Africa với BraTS 2020, PCA fit trên train BraTS 2020."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.preprocess.audit_africa import find_case_files


COHORTS = ["BraTS 2020", "Africa glioma", "Africa other tumors"]
COLORS = ["#2563eb", "#d97706", "#7c3aed"]
CHANNELS = [("t1", "T1"), ("t1ce", "T1ce"), ("t2", "T2"), ("flair", "FLAIR")]


def cohort_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Mỗi dòng là một nhóm, thể tích theo ca và tỷ lệ voxel tính gộp toàn nhóm."""
    records = []
    for cohort in COHORTS:
        part = frame.loc[frame["cohort"] == cohort]
        records.append({"cohort": cohort, "num_cases": len(part),
                        "wt_mean_ml": part["wt_volume_ml"].mean(),
                        "wt_sd_ml": part["wt_volume_ml"].std(ddof=1),
                        "wt_median_ml": part["wt_volume_ml"].median(),
                        "wt_min_ml": part["wt_volume_ml"].min(),
                        "wt_max_ml": part["wt_volume_ml"].max(),
                        "tumor_slice_pct": 100 * part["num_tumor_slices"].sum() / part["num_slices_z"].sum(),
                        "wt_voxel_pct_in_brain": 100 * part["wt_voxels"].sum() / part["brain_voxels"].sum()})
    return pd.DataFrame(records)


def draw_distributions(frame: pd.DataFrame, figure_dir: Path) -> None:
    """Hiển thị đủ ba nhóm, tránh gộp các loại u khác nhau thành một nhóm glioma."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for cohort, color in zip(COHORTS, COLORS):
        values = frame.loc[frame["cohort"] == cohort, "wt_volume_ml"]
        axes[0].hist(values, bins=np.linspace(0, 450, 31), density=True,
                     histtype="step", linewidth=2, color=color, label=f"{cohort} (n={len(values)})")
    axes[0].set(xlabel="Whole-tumor volume (mL)", ylabel="Density", title="Whole-tumor volume")
    axes[0].legend(fontsize=8)
    boxes = axes[1].boxplot([frame.loc[frame["cohort"] == c, "wt_volume_ml"] for c in COHORTS],
                            tick_labels=["BraTS 2020", "Africa\nglioma", "Africa\nother tumors"],
                            patch_artist=True, showfliers=True)
    for box, color in zip(boxes["boxes"], COLORS):
        box.set(facecolor=color, alpha=0.4)
    axes[1].set(ylabel="Whole-tumor volume (mL)", title="Patient-level distribution")
    fig.savefig(figure_dir / "africa_01_wt_volume_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for axis, (channel, name) in zip(axes.flat, CHANNELS):
        boxes = axis.boxplot([frame.loc[frame["cohort"] == c, f"{channel}_mean_brain"] for c in COHORTS],
                             tick_labels=["BraTS 2020", "Africa\nglioma", "Africa\nother tumors"],
                             patch_artist=True)
        for box, color in zip(boxes["boxes"], COLORS):
            box.set(facecolor=color, alpha=0.4)
        axis.set(title=name, ylabel="Mean input intensity, nonzero voxels")
    fig.suptitle("MRI intensity statistics before the shared z-score pipeline")
    fig.savefig(figure_dir / "africa_02_intensity_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for axis, column, label in zip(axes, ("tumor_slice_ratio", "wt_ratio_in_brain"),
                                  ("Axial slices containing tumor (%)", "WT voxels within brain union (%)")):
        boxes = axis.boxplot([100 * frame.loc[frame["cohort"] == c, column] for c in COHORTS],
                             tick_labels=["BraTS 2020", "Africa\nglioma", "Africa\nother tumors"],
                             patch_artist=True)
        for box, color in zip(boxes["boxes"], COLORS):
            box.set(facecolor=color, alpha=0.4)
        axis.set(ylabel=label, title="Patient-level proportions")
    fig.savefig(figure_dir / "africa_03_class_imbalance_comparison.png", dpi=180)
    plt.close(fig)


def run_pca(frame: pd.DataFrame, feature_names: list[str], out_dir: Path,
            suffix: str, description: str) -> dict:
    """Fit scaler/PCA chỉ bằng train, giữ Africa ngoài bước ước lượng phép biến đổi."""
    fit_mask = (frame["dataset"] == "BraTS2020") & (frame["split"] == "train")
    if int(fit_mask.sum()) != 259:
        raise ValueError("PCA cần đúng 259 ca train theo split đã khóa")
    values = frame[feature_names].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Đặc trưng PCA chứa NaN/Inf")
    scaler = StandardScaler()
    scaler.fit(values[fit_mask])
    standardized = scaler.transform(values)
    pca = PCA(n_components=2, svd_solver="full")
    pca.fit(standardized[fit_mask])
    projected = pca.transform(standardized)
    scores = frame[["case_id", "dataset", "cohort", "split"]].copy()
    scores["PC1"], scores["PC2"] = projected[:, 0], projected[:, 1]
    scores.to_csv(out_dir / f"pca_{suffix}_scores.csv", index=False)
    pd.DataFrame({"feature": feature_names, "PC1": pca.components_[0],
                  "PC2": pca.components_[1]}).to_csv(out_dir / f"pca_{suffix}_loadings.csv", index=False)

    fig, axis = plt.subplots(figsize=(8.6, 6.2), layout="constrained")
    for cohort, color in zip(COHORTS, COLORS):
        selected = frame["cohort"] == cohort
        axis.scatter(projected[selected, 0], projected[selected, 1], s=24,
                     alpha=0.58, color=color, label=f"{cohort} (n={int(selected.sum())})")
    ratio = pca.explained_variance_ratio_
    axis.set(xlabel=f"PC1 ({100 * ratio[0]:.1f}% train variance)",
             ylabel=f"PC2 ({100 * ratio[1]:.1f}% train variance)",
             title=f"PCA of {description}\nScaler and PCA fitted on BraTS 2020 training cases only")
    axis.legend(fontsize=9)
    axis.grid(alpha=0.15)
    fig.savefig(out_dir / "figures" / f"africa_04_pca_{suffix}.png", dpi=180)
    plt.close(fig)
    metadata = {"feature_names": feature_names, "num_features": len(feature_names),
                "fit_cases": frame.loc[fit_mask, "case_id"].tolist(), "fit_dataset": "BraTS2020 train",
                "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
                "components": pca.components_.tolist(), "pca_mean": pca.mean_.tolist(),
                "explained_variance_ratio": ratio.tolist(),
                "uses_ground_truth_volume": "wt_volume_ml" in feature_names,
                "purpose": "Thống kê mô tả, không dùng PCA để chọn mô hình hoặc chỉnh pipeline theo Africa"}
    (out_dir / f"pca_{suffix}_transform.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def draw_africa_samples(data_dir: Path, africa: pd.DataFrame, figure_dir: Path) -> None:
    """Vẽ ba ca nhỏ/vừa/lớn ở từng nhóm, dùng cùng hướng và cùng lát cho MRI/GT."""
    for group in ("95_Glioma", "51_OtherNeoplasms"):
        part = africa.loc[africa["group"] == group].sort_values("wt_volume_ml").reset_index(drop=True)
        positions = [int(len(part) * fraction) for fraction in (0.1, 0.5, 0.9)]
        fig, axes = plt.subplots(3, 5, figsize=(14, 9), layout="constrained")
        for row_index, (position, tag) in enumerate(zip(positions, ("Small", "Median", "Large"))):
            row = part.iloc[position]
            files = find_case_files(data_dir / group / row["case_id"])
            seg_image = nib.as_closest_canonical(nib.load(str(files["seg"])))
            wt = np.isin(np.asarray(seg_image.dataobj), [1, 2, 3])
            z = int(wt.sum(axis=(0, 1)).argmax())
            for column, (channel, label) in enumerate(CHANNELS):
                image = nib.as_closest_canonical(nib.load(str(files[channel])))
                sl = np.asarray(image.dataobj, dtype=np.float32)[:, :, z].T
                selected = sl[sl != 0]
                low, high = np.percentile(selected, [1, 99]) if selected.size else (0, 1)
                if high <= low:
                    high = low + 1
                axes[row_index, column].imshow(sl, cmap="gray", origin="lower", vmin=low, vmax=high)
                if row_index == 0:
                    axes[row_index, column].set_title(label)
                if column == 0:
                    axes[row_index, column].set_ylabel(
                        f"{tag}, {row['wt_volume_ml']:.1f} mL\n{row['case_id']}\nRAS+ axial z={z}", fontsize=8)
                if channel == "flair":
                    axes[row_index, 4].imshow(sl, cmap="gray", origin="lower", vmin=low, vmax=high)
                    overlay = np.ma.masked_where(~wt[:, :, z].T, wt[:, :, z].T)
                    axes[row_index, 4].imshow(overlay, cmap="autumn", alpha=0.5, origin="lower")
            if row_index == 0:
                axes[row_index, 4].set_title("FLAIR + WT ground truth")
            for axis in axes[row_index]:
                axis.set_xticks([])
                axis.set_yticks([])
        fig.suptitle(f"BraTS-Africa, {group}, tumor-volume percentiles 10 / 50 / 90")
        fig.savefig(figure_dir / f"africa_05_samples_{group}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brats2020-csv", type=Path, required=True)
    parser.add_argument("--africa-csv", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--africa-data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    reference = pd.read_csv(args.brats2020_csv)
    africa = pd.read_csv(args.africa_csv)
    splits = json.loads(args.splits.read_text(encoding="utf-8"))["splits"]
    mapping = {case_id: split for split, ids in splits.items() for case_id in ids}
    if len(reference) != 369 or set(reference["case_id"]) != set(mapping):
        raise ValueError("Bảng BraTS 2020 không khớp 369 ca trong split")
    if len(africa) != 146 or africa["case_id"].duplicated().any():
        raise ValueError("Bảng Africa cần đủ 146 ca riêng biệt")
    reference["dataset"], reference["cohort"] = "BraTS2020", COHORTS[0]
    reference["split"] = reference["case_id"].map(mapping)
    africa["dataset"] = "BraTSAfrica"
    africa["cohort"] = africa["group"].map({"95_Glioma": COHORTS[1], "51_OtherNeoplasms": COHORTS[2]})
    if africa["cohort"].isna().any():
        raise ValueError("Có nhóm Africa chưa xác định")
    frame = pd.concat([reference, africa], ignore_index=True)
    frame["brain_volume_ml"] = (frame["brain_voxels"] * frame["spacing_x_mm"] *
                                 frame["spacing_y_mm"] * frame["spacing_z_mm"] / 1000)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    frame.to_csv(out / "combined_case_features.csv", index=False)
    table = cohort_table(frame)
    table.to_csv(out / "cohort_comparison.csv", index=False)
    draw_distributions(frame, figures)
    features = [f"{channel}_{stat}_brain" for channel, _ in CHANNELS
                for stat in ("mean", "std", "p05", "p50", "p95")] + ["brain_volume_ml"]
    main_pca = run_pca(frame, features, out, "mri", "MRI statistics (no tumor-label features)")
    label_pca = run_pca(frame, features + ["wt_volume_ml"], out, "with_wt", "MRI statistics and GT tumor volume")
    draw_africa_samples(args.africa_data_dir, africa, figures)
    provenance = {"inputs": {str(path.name): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in (args.brats2020_csv, args.africa_csv, args.splits)},
                  "pca_fit_count": len(main_pca["fit_cases"]),
                  "pca_mri_features": main_pca["num_features"],
                  "pca_with_wt_features": label_pca["num_features"],
                  "notes": ["So sánh cường độ trên NIfTI đầu vào, chưa dùng z-score của pipeline chung",
                            "PCA chỉ mô tả phân bố, không chứng minh nguyên nhân domain shift",
                            "Không trộn Africa vào train/validation hoặc chọn checkpoint"]}
    (out / "comparison_manifest.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(table.to_string(index=False))
    print(f"Đã lưu biểu đồ và PCA tại {out}")


if __name__ == "__main__":
    main()
