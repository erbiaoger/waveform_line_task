"""Loss functions for waveform-line segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossConfig:
    """Weights for segmentation loss terms."""

    bce_weight: float = 1.0
    dice_weight: float = 1.0
    positive_class_weight: float = 6.0


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable dice loss on sigmoid probabilities."""

    prob = torch.sigmoid(logits)
    inter = torch.sum(prob * target, dim=(1, 2, 3))
    denom = torch.sum(prob, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor, config: LossConfig) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute weighted BCE plus Dice for single-channel masks."""

    pos_weight = torch.tensor([float(config.positive_class_weight)], device=logits.device, dtype=logits.dtype)
    loss_bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    loss_dice = soft_dice_loss(logits, target)
    loss = float(config.bce_weight) * loss_bce + float(config.dice_weight) * loss_dice
    metrics = {
        "loss_bce": float(loss_bce.detach().cpu()),
        "loss_dice": float(loss_dice.detach().cpu()),
        "loss": float(loss.detach().cpu()),
    }
    return loss, metrics

