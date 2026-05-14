"""Batch inference for the independent waveform-line segmentation model.

Purpose:
    Load a trained waveform-line U-Net checkpoint and run batch PNG inference:

        images/ -> pred_masks/, pred_skeletons/, previews/, summary.json

Usage:
    uv run python waveform_line_task/predict_model.py \
      --input-dir waveform_line_task/datasets/sample_check/images \
      --model waveform_line_task/models/unet_v1/checkpoint_best.pt \
      --out-dir waveform_line_task/predictions/sample_check \
      --image-size 512 \
      --device cuda

Notes:
    - `--device auto` is the default and prefers CUDA.
    - CUDA inference enables AMP, pinned host memory, channels-last tensors,
      and non-blocking host-to-device copies.
    - CPU fallback remains available with `--device cpu`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from device_utils import configure_torch_runtime, resolve_runtime_device
    from model.dataset import WaveformLineSegmentationDataset, collate_batch, load_manifest
    from model.network import UNetConfig, WaveformLineUNet
    from model.postprocess import logits_to_binary_mask, mask_to_skeleton_tensor
    from render import save_gray_png, save_preview_png
else:
    from .device_utils import configure_torch_runtime, resolve_runtime_device
    from .model.dataset import WaveformLineSegmentationDataset, collate_batch, load_manifest
    from .model.network import UNetConfig, WaveformLineUNet
    from .model.postprocess import logits_to_binary_mask, mask_to_skeleton_tensor
    from .render import save_gray_png, save_preview_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict waveform-line masks and skeletons from an images directory.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Input images directory under a waveform-line dataset.")
    parser.add_argument("--model", required=True, type=Path, help="Checkpoint path produced by train_model.py.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for masks, skeletons, previews, and summary.json.")
    parser.add_argument("--image-size", type=int, default=512, help="Inference resolution before resizing outputs back to the original size.")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Torch device. 'auto' prefers CUDA, then MPS, then CPU.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for mask binarization.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker count.")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on supported devices. Enabled by default for CUDA.")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision even when CUDA is used.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_args(args)
    runtime_device = resolve_runtime_device(str(args.device))
    configure_torch_runtime(runtime_device)
    device = runtime_device.torch_device
    use_amp = bool(runtime_device.use_cuda and not bool(args.no_amp))
    if bool(args.amp):
        use_amp = bool(runtime_device.use_cuda)
    out_dir = Path(args.out_dir).expanduser()
    _prepare_out_dir(out_dir)

    dataset_root = Path(args.input_dir).expanduser().parent
    records = [record for record in load_manifest(dataset_root) if record.image_path.parent == Path(args.input_dir).expanduser().resolve()]
    if not records:
        raise ValueError(f"No manifest records matched images under {args.input_dir}")
    dataset = WaveformLineSegmentationDataset(records, image_size=int(args.image_size), include_labels=False)
    loader_kwargs: dict[str, Any] = {
        "pin_memory": bool(runtime_device.use_cuda),
        "persistent_workers": bool(int(args.num_workers) > 0),
    }
    if int(args.num_workers) > 0:
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_batch,
        **loader_kwargs,
    )

    checkpoint = torch.load(str(Path(args.model).expanduser()), map_location="cpu", weights_only=False)
    model_config = UNetConfig(**dict(checkpoint.get("model_config", {})))
    model = WaveformLineUNet(model_config).to(device)
    if runtime_device.use_cuda:
        model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    rows: list[dict[str, Any]] = []
    total_time = 0.0
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=bool(runtime_device.use_cuda))
            if runtime_device.use_cuda:
                image = image.to(memory_format=torch.channels_last)
            t0 = time.perf_counter()
            with autocast(device_type="cuda", enabled=bool(use_amp and runtime_device.use_cuda)):
                logits = model(image)
            total_time += float(time.perf_counter() - t0)
            mask = logits_to_binary_mask(logits, threshold=float(args.threshold)).cpu()
            skeleton = mask_to_skeleton_tensor(mask).cpu()
            for idx in range(int(mask.shape[0])):
                stem = str(batch["stem"][idx])
                original_h, original_w = batch["original_size"][idx]
                mask_orig = _resize_binary(mask[idx : idx + 1], original_h, original_w)
                skeleton_orig = _resize_binary(skeleton[idx : idx + 1], original_h, original_w)
                input_orig = _resize_gray(batch["image"][idx : idx + 1].cpu(), original_h, original_w)

                mask_path = out_dir / "pred_masks" / f"{stem}.png"
                skeleton_path = out_dir / "pred_skeletons" / f"{stem}.png"
                preview_path = out_dir / "previews" / f"{stem}_overlay.png"
                save_gray_png(mask_path, (mask_orig[0, 0].numpy() * 255.0).astype("uint8"))
                save_gray_png(skeleton_path, (skeleton_orig[0, 0].numpy() * 255.0).astype("uint8"))
                save_preview_png(preview_path, _build_preview(input_orig[0, 0], mask_orig[0, 0], skeleton_orig[0, 0]))
                rows.append(
                    {
                        "sample_index": int(batch["sample_index"][idx]),
                        "stem": stem,
                        "mask_path": str(mask_path),
                        "skeleton_path": str(skeleton_path),
                        "preview_path": str(preview_path),
                        "positive_ratio": float(mask_orig.mean().item()),
                        "skeleton_ratio": float(skeleton_orig.mean().item()),
                    }
                )

    summary = {
        "input_dir": str(Path(args.input_dir).expanduser()),
        "model": str(Path(args.model).expanduser()),
        "device_requested": runtime_device.requested,
        "device_resolved": runtime_device.resolved,
        "amp": bool(use_amp),
        "threshold": float(args.threshold),
        "image_size": int(args.image_size),
        "num_inputs": int(len(records)),
        "num_success": int(len(rows)),
        "avg_inference_seconds": float(total_time / max(1, len(records))),
        "avg_positive_ratio": float(sum(float(row["positive_ratio"]) for row in rows) / max(1, len(rows))),
        "avg_skeleton_ratio": float(sum(float(row["skeleton_ratio"]) for row in rows) / max(1, len(rows))),
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"done: inputs={len(records)}, success={len(rows)}, out_dir={out_dir}", flush=True)
    return 0


def _build_preview(image: torch.Tensor, mask: torch.Tensor, skeleton: torch.Tensor) -> Any:
    base = (torch.clamp(image, 0.0, 1.0) * 255.0).to(torch.uint8).numpy()
    rgb = np.stack([base, base, base], axis=0)
    mask_np = mask.numpy() >= 0.5
    skel_np = skeleton.numpy() >= 0.5
    rgb[0, mask_np] = 255
    rgb[1, mask_np] = np.minimum(rgb[1, mask_np], 80)
    rgb[2, mask_np] = np.minimum(rgb[2, mask_np], 80)
    rgb[0, skel_np] = 0
    rgb[1, skel_np] = 255
    rgb[2, skel_np] = 255
    return rgb


def _resize_binary(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    out = F.interpolate(mask.to(torch.float32), size=(int(height), int(width)), mode="nearest")
    return (out >= 0.5).to(torch.float32)


def _resize_gray(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return F.interpolate(image.to(torch.float32), size=(int(height), int(width)), mode="bilinear", align_corners=False)


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.image_size) <= 0:
        raise ValueError("--image-size must be > 0")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be > 0")
    if int(args.num_workers) < 0:
        raise ValueError("--num-workers must be >= 0")
    if not (0.0 <= float(args.threshold) <= 1.0):
        raise ValueError("--threshold must be in [0, 1]")
    if bool(args.amp) and bool(args.no_amp):
        raise ValueError("--amp and --no-amp cannot be used together")


def _prepare_out_dir(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "pred_masks").mkdir(parents=True, exist_ok=True)
    (out_dir / "pred_skeletons").mkdir(parents=True, exist_ok=True)
    (out_dir / "previews").mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
