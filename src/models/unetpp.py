"""Mô hình U-Net++ 2D và hàm Loss dùng chung theo Quy ước v1.1 (Dice + BCE)."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def build_unetpp_2d(
    encoder_name: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 4,
    classes: int = 1,
) -> nn.Module:
    """Khởi tạo U-Net++ 2D xuất logits [B, 1, 224, 224] (không đặt Sigmoid ở lớp cuối)."""
    return smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,
    )


class SoftDiceBCELoss(nn.Module):
    """Loss = bce_weight * BCEWithLogitsLoss + dice_weight * SoftDiceLoss (batch-level)."""

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)

        probs_flat = probs.reshape(-1)
        targets_flat = targets.reshape(-1)
        intersection = (probs_flat * targets_flat).sum()
        denom = probs_flat.sum() + targets_flat.sum()
        soft_dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        dice_loss = 1.0 - soft_dice

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
