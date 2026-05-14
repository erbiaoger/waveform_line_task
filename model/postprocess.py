"""Mask thresholding and skeleton extraction."""

from __future__ import annotations

import numpy as np
import torch


def logits_to_binary_mask(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Convert logits to binary float mask with values in {0, 1}."""

    prob = torch.sigmoid(logits)
    return (prob >= float(threshold)).to(torch.float32)


def mask_to_skeleton_tensor(mask: torch.Tensor) -> torch.Tensor:
    """Skeletonize a batch or single mask tensor and return {0,1} float output."""

    if mask.ndim == 4:
        items = [mask_to_skeleton_tensor(mask[idx]) for idx in range(int(mask.shape[0]))]
        return torch.stack(items, dim=0)
    if mask.ndim == 3:
        if int(mask.shape[0]) != 1:
            raise ValueError(f"Expected single-channel mask, got {tuple(mask.shape)}")
        skel = skeletonize_binary(mask[0].detach().cpu().numpy() >= 0.5)
        return torch.from_numpy(skel.astype(np.float32))[None, :, :]
    if mask.ndim == 2:
        skel = skeletonize_binary(mask.detach().cpu().numpy() >= 0.5)
        return torch.from_numpy(skel.astype(np.float32))
    raise ValueError(f"Unsupported mask rank: {mask.ndim}")


def skeletonize_binary(mask: np.ndarray) -> np.ndarray:
    """Thin a binary mask with the Zhang-Suen iterative algorithm."""

    image = np.asarray(mask, dtype=np.uint8)
    if image.ndim != 2:
        raise ValueError(f"Expected 2D binary mask, got {image.shape}")
    image = (image > 0).astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            marker = np.zeros_like(image, dtype=bool)
            for y in range(1, image.shape[0] - 1):
                for x in range(1, image.shape[1] - 1):
                    if image[y, x] != 1:
                        continue
                    p2 = image[y - 1, x]
                    p3 = image[y - 1, x + 1]
                    p4 = image[y, x + 1]
                    p5 = image[y + 1, x + 1]
                    p6 = image[y + 1, x]
                    p7 = image[y + 1, x - 1]
                    p8 = image[y, x - 1]
                    p9 = image[y - 1, x - 1]
                    neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                    count = int(sum(neighbors))
                    if count < 2 or count > 6:
                        continue
                    transitions = int(
                        (p2 == 0 and p3 == 1)
                        + (p3 == 0 and p4 == 1)
                        + (p4 == 0 and p5 == 1)
                        + (p5 == 0 and p6 == 1)
                        + (p6 == 0 and p7 == 1)
                        + (p7 == 0 and p8 == 1)
                        + (p8 == 0 and p9 == 1)
                        + (p9 == 0 and p2 == 1)
                    )
                    if transitions != 1:
                        continue
                    if step == 0:
                        if p2 * p4 * p6 != 0:
                            continue
                        if p4 * p6 * p8 != 0:
                            continue
                    else:
                        if p2 * p4 * p8 != 0:
                            continue
                        if p2 * p6 * p8 != 0:
                            continue
                    marker[y, x] = True
            if np.any(marker):
                image[marker] = 0
                changed = True
    return image.astype(bool)

