"""Evaluation metrics for waveform-line segmentation."""

from __future__ import annotations

import torch

from .postprocess import logits_to_binary_mask, mask_to_skeleton_tensor


def segmentation_metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    """Compute pixel and skeleton metrics for a batch."""

    pred = logits_to_binary_mask(logits, threshold=threshold)
    target_bin = (target >= 0.5).to(torch.float32)
    tp = torch.sum(pred * target_bin).item()
    fp = torch.sum(pred * (1.0 - target_bin)).item()
    fn = torch.sum((1.0 - pred) * target_bin).item()
    eps = 1e-6
    precision = tp / max(eps, tp + fp)
    recall = tp / max(eps, tp + fn)
    f1 = 2.0 * precision * recall / max(eps, precision + recall)
    iou = tp / max(eps, tp + fp + fn)
    dice = 2.0 * tp / max(eps, 2.0 * tp + fp + fn)

    pred_skel = mask_to_skeleton_tensor(pred).to(target_bin.dtype)
    target_skel = mask_to_skeleton_tensor(target_bin).to(target_bin.dtype)
    sk_tp = torch.sum(pred_skel * target_skel).item()
    sk_fp = torch.sum(pred_skel * (1.0 - target_skel)).item()
    sk_fn = torch.sum((1.0 - pred_skel) * target_skel).item()
    sk_precision = sk_tp / max(eps, sk_tp + sk_fp)
    sk_recall = sk_tp / max(eps, sk_tp + sk_fn)
    sk_f1 = 2.0 * sk_precision * sk_recall / max(eps, sk_precision + sk_recall)

    return {
        "pixel_precision": float(precision),
        "pixel_recall": float(recall),
        "pixel_f1": float(f1),
        "iou": float(iou),
        "dice": float(dice),
        "skeleton_precision": float(sk_precision),
        "skeleton_recall": float(sk_recall),
        "skeleton_f1": float(sk_f1),
    }

