"""Huấn luyện U-Net++ 2D trên BraTS 2020 với Mixed Precision và chọn checkpoint theo Val Dice 3D."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.brats2d_dataset import BraTS2DDataset, prepare_mmap_cache
from src.models.unetpp import SoftDiceBCELoss, build_unetpp_2d


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_ctx(device_type: str, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device_type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0

    for imgs, msks, _, _ in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        msks = msks.to(device, non_blocking=True)
        bs = imgs.size(0)

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(device.type, enabled=use_amp and device.type == "cuda"):
            logits = model(imgs)
            loss = criterion(logits, msks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item()) * bs
        n_samples += bs

    return total_loss / max(n_samples, 1)


@torch.no_grad()
def validate_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    use_amp: bool,
    threshold: float = 0.5,
) -> tuple[float, float, float]:
    """Đánh giá Val loss và Dice 3D theo từng bệnh nhân (nhất quán với Quy ước v1.1)."""
    model.eval()
    total_loss = 0.0
    n_samples = 0

    case_inter: dict[str, float] = defaultdict(float)
    case_pred: dict[str, float] = defaultdict(float)
    case_gt: dict[str, float] = defaultdict(float)

    for imgs, msks, case_ids, _ in tqdm(loader, desc="Val", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        msks = msks.to(device, non_blocking=True)
        bs = imgs.size(0)

        with autocast_ctx(device.type, enabled=use_amp and device.type == "cuda"):
            logits = model(imgs)
            loss = criterion(logits, msks)

        total_loss += float(loss.item()) * bs
        n_samples += bs

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()
        gt = (msks >= 0.5).float()

        inter_b = (preds * gt).sum(dim=(1, 2, 3)).cpu().numpy()
        pred_b = preds.sum(dim=(1, 2, 3)).cpu().numpy()
        gt_b = gt.sum(dim=(1, 2, 3)).cpu().numpy()

        for cid, iv, pv, gv in zip(case_ids, inter_b, pred_b, gt_b):
            case_inter[cid] += float(iv)
            case_pred[cid] += float(pv)
            case_gt[cid] += float(gv)

    dices = []
    for cid in sorted(case_gt.keys()):
        pv = case_pred[cid]
        gv = case_gt[cid]
        if pv == 0.0 and gv == 0.0:
            dices.append(1.0)
        elif pv == 0.0 or gv == 0.0:
            dices.append(0.0)
        else:
            dices.append((2.0 * case_inter[cid]) / (pv + gv))

    mean_dice = float(np.mean(dices)) if dices else 0.0
    std_dice = float(np.std(dices, ddof=1)) if len(dices) > 1 else 0.0
    return total_loss / max(n_samples, 1), mean_dice, std_dice


def save_training_curves(history: list[dict], out_path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_dice = [h["val_dice_3d"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, train_loss, label="Train Loss", marker="o", markersize=3)
    axes[0].plot(epochs, val_loss, label="Val Loss", marker="s", markersize=3)
    axes[0].set_title("Dice + BCE Loss theo Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, val_dice, color="#059669", label="Val 3D Dice (mean)", marker="o", markersize=3)
    axes[1].set_title("Validation 3D Dice theo Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].set_ylim(0, 1.0)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện U-Net++ 2D trên BraTS 2020")
    parser.add_argument("--config", type=Path, default=Path("configs/train_config.yaml"))
    parser.add_argument("--data-dir", type=Path, required=True, help="Thư mục chứa cases/ và metadata/")
    parser.add_argument("--cache-dir", type=Path, default=Path("/content/brats_npy_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/unetpp_2d"))
    parser.add_argument("--epochs", type=int, default=None, help="Ghi đè số epoch trong config nếu cần")
    parser.add_argument("--batch-size", type=int, default=None, help="Ghi đè batch_size trong config nếu cần")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(cfg["project"]["seed"])
    set_seed(seed)

    splits_data = json.loads(Path(cfg["splits"]["splits_file"]).read_text(encoding="utf-8"))
    train_ids: list[str] = splits_data["splits"]["train"]
    val_ids: list[str] = splits_data["splits"]["val"]

    train_cfg = cfg["training"]
    epochs = args.epochs if args.epochs is not None else int(train_cfg["epochs"])
    batch_size = args.batch_size if args.batch_size is not None else int(train_cfg["batch_size"])
    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg["weight_decay"])
    patience = int(train_cfg["early_stopping_patience"])
    use_amp = bool(train_cfg["amp"])
    num_workers = int(train_cfg["num_workers"])
    neg_ratio = float(train_cfg.get("train_neg_slice_ratio", 0.5))
    threshold = float(cfg["evaluation"]["threshold"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị: {device} | Epochs: {epochs} | Batch size: {batch_size} | AMP: {use_amp}")

    manifest = prepare_mmap_cache(args.data_dir, args.cache_dir, train_ids + val_ids)

    train_ds = BraTS2DDataset(
        cache_dir=args.cache_dir,
        manifest=manifest,
        case_ids=train_ids,
        is_train=True,
        aug_cfg=cfg["augmentation"],
        neg_slice_ratio=neg_ratio,
        brain_only=True,
        seed=seed,
    )
    val_ds = BraTS2DDataset(
        cache_dir=args.cache_dir,
        manifest=manifest,
        case_ids=val_ids,
        is_train=False,
        aug_cfg=None,
        neg_slice_ratio=1.0,
        brain_only=False,
        seed=seed,
    )
    print(f"Train slices/epoch: {len(train_ds)} ({len(train_ids)} ca) | Val slices: {len(val_ds)} ({len(val_ids)} ca)")

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    m_cfg = cfg["model"]
    model = build_unetpp_2d(
        encoder_name=m_cfg["encoder_name"],
        encoder_weights=m_cfg["encoder_weights"],
        in_channels=int(m_cfg["in_channels"]),
        classes=int(m_cfg["classes"]),
    ).to(device)

    l_cfg = cfg["loss"]
    criterion = SoftDiceBCELoss(
        bce_weight=float(l_cfg["bce_weight"]),
        dice_weight=float(l_cfg["dice_weight"]),
        smooth=float(l_cfg["dice_smooth"]),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = make_grad_scaler(enabled=use_amp and device.type == "cuda")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_dice = -1.0
    no_improve = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_ds.resample_epoch(epoch)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
        )

        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        val_loss, val_dice, val_std = validate_epoch(
            model, val_loader, criterion, device, use_amp, threshold=threshold
        )
        scheduler.step()
        elapsed = time.time() - t0
        cur_lr = float(optimizer.param_groups[0]["lr"])

        rec = {
            "epoch": epoch,
            "lr": round(cur_lr, 8),
            "train_loss": round(tr_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_dice_3d": round(val_dice, 5),
            "val_dice_std": round(val_std, 5),
            "sec": round(elapsed, 1),
        }
        history.append(rec)
        print(
            f"[Epoch {epoch:02d}/{epochs}] lr={cur_lr:.2e} | "
            f"train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_dice_3d={val_dice:.4f} ± {val_std:.4f} ({elapsed:.1f}s)"
        )

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_dice_3d": val_dice,
            "config": cfg,
        }
        torch.save(ckpt, args.output_dir / "last_model.pth")

        if val_dice > best_dice:
            best_dice = val_dice
            no_improve = 0
            torch.save(ckpt, args.output_dir / "best_model.pth")
            print(f"  --> Lưu best_model.pth mới (val_dice_3d = {best_dice:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping tại epoch {epoch} (không cải thiện sau {patience} epoch).")
                break

        with (args.output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        save_training_curves(history, args.output_dir / "training_curves.png")

    print(f"Hoàn tất huấn luyện! Best Val 3D Dice = {best_dice:.4f}. Checkpoint lưu tại {args.output_dir}")


if __name__ == "__main__":
    main()
