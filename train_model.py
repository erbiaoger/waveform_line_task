"""Train the independent waveform-line U-Net segmentation model.

Purpose:
    Train a standalone single-class segmentation model for the task:

        waveform PNG -> binary vehicle-line mask

    The implementation stays inside `waveform_line_task/` and does not import
    `autotrack` or any legacy training code.

Usage:
    uv run python waveform_line_task/train_model.py \
      --data-dir waveform_line_task/datasets/v1_train \
      --out-dir waveform_line_task/models/unet_v1 \
      --image-size 512 \
      --batch-size 8 \
      --epochs 50 \
      --device cpu

Outputs:
    <out-dir>/checkpoint_last.pt
    <out-dir>/checkpoint_best.pt
    <out-dir>/train_config.json
    <out-dir>/train_history.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model.dataset import WaveformLineSegmentationDataset, collate_batch, load_manifest, split_records
    from model.losses import LossConfig, segmentation_loss
    from model.metrics import segmentation_metrics
    from model.network import UNetConfig, WaveformLineUNet
else:
    from .model.dataset import WaveformLineSegmentationDataset, collate_batch, load_manifest, split_records
    from .model.losses import LossConfig, segmentation_loss
    from .model.metrics import segmentation_metrics
    from .model.network import UNetConfig, WaveformLineUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the independent waveform-line U-Net model.")
    parser.add_argument("--data-dir", required=True, type=Path, help="Dataset directory containing manifest.csv, images/, and labels/.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for checkpoints and logs.")
    parser.add_argument("--image-size", type=int, default=512, help="Training resolution after resizing.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--lr", type=float, default=2e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Tail fraction reserved for validation.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"], help="Torch device.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold used for metrics on sigmoid predictions.")
    parser.add_argument("--positive-class-weight", type=float, default=6.0, help="Positive weight for BCEWithLogits.")
    parser.add_argument("--bce-weight", type=float, default=1.0, help="Weight for BCE term.")
    parser.add_argument("--dice-weight", type=float, default=1.0, help="Weight for Dice term.")
    parser.add_argument("--base-channels", type=int, default=32, help="Base U-Net width.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Optional dropout in conv blocks.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save-every", type=int, default=1, help="Checkpoint frequency in epochs.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing non-empty out-dir.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_args(args)
    _seed_everything(int(args.seed))
    device = _resolve_device(str(args.device))
    out_dir = Path(args.out_dir).expanduser()
    _prepare_out_dir(out_dir, overwrite=bool(args.overwrite))

    records = load_manifest(Path(args.data_dir).expanduser())
    train_records, val_records = split_records(records, float(args.val_fraction))
    train_dataset = WaveformLineSegmentationDataset(train_records, image_size=int(args.image_size), include_labels=True)
    val_dataset = WaveformLineSegmentationDataset(val_records, image_size=int(args.image_size), include_labels=True) if val_records else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=collate_batch,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            collate_fn=collate_batch,
        )

    model_config = UNetConfig(base_channels=int(args.base_channels), dropout=float(args.dropout))
    model = WaveformLineUNet(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loss_config = LossConfig(
        bce_weight=float(args.bce_weight),
        dice_weight=float(args.dice_weight),
        positive_class_weight=float(args.positive_class_weight),
    )

    config_payload = {
        "train_args": _json_ready(vars(args)),
        "model_config": asdict(model_config),
        "loss_config": asdict(loss_config),
        "dataset": {
            "num_records": len(records),
            "train_records": len(train_records),
            "val_records": len(val_records),
        },
    }
    (out_dir / "train_config.json").write_text(json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    best_loss = math.inf
    history_path = out_dir / "train_history.jsonl"
    t0 = time.perf_counter()
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = _run_epoch(
            model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_config=loss_config,
            threshold=float(args.threshold),
            train=True,
        )
        val_metrics = {}
        score_loss = float(train_metrics["loss"])
        if val_loader is not None:
            val_metrics = _run_epoch(
                model,
                loader=val_loader,
                optimizer=None,
                device=device,
                loss_config=loss_config,
                threshold=float(args.threshold),
                train=False,
            )
            score_loss = float(val_metrics["loss"])

        row: dict[str, float | int] = {
            "epoch": int(epoch),
            "elapsed_seconds": float(time.perf_counter() - t0),
        }
        row.update({f"train_{k}": float(v) for k, v in train_metrics.items()})
        row.update({f"val_{k}": float(v) for k, v in val_metrics.items()})
        _append_history_row(history_path, row)

        checkpoint_metrics = {"train": train_metrics, "val": val_metrics}
        if int(epoch) % int(args.save_every) == 0:
            _save_checkpoint(out_dir / "checkpoint_last.pt", model, optimizer, model_config, loss_config, epoch, checkpoint_metrics)
        if score_loss < best_loss:
            best_loss = score_loss
            _save_checkpoint(out_dir / "checkpoint_best.pt", model, optimizer, model_config, loss_config, epoch, checkpoint_metrics)

        print(
            f"epoch {epoch}/{int(args.epochs)} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics.get('loss', float('nan')):.4f}",
            flush=True,
        )

    _save_checkpoint(out_dir / "checkpoint_last.pt", model, optimizer, model_config, loss_config, int(args.epochs), {"train": train_metrics, "val": val_metrics})
    print(f"done: out_dir={out_dir}, elapsed={time.perf_counter() - t0:.1f}s", flush=True)
    return 0


def _run_epoch(
    model: WaveformLineUNet,
    *,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    loss_config: LossConfig,
    threshold: float,
    train: bool,
) -> dict[str, float]:
    model.train(mode=bool(train))
    rows: list[dict[str, float]] = []
    for batch in loader:
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        with torch.set_grad_enabled(bool(train)):
            logits = model(image)
            loss, loss_metrics = segmentation_loss(logits, target, loss_config)
            metrics = segmentation_metrics(logits.detach(), target.detach(), threshold=threshold)
            row = {**loss_metrics, **metrics}
            rows.append(row)
            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
    return _mean_metrics(rows)


def _save_checkpoint(
    path: Path,
    model: WaveformLineUNet,
    optimizer: torch.optim.Optimizer,
    model_config: UNetConfig,
    loss_config: LossConfig,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": asdict(model_config),
        "loss_config": asdict(loss_config),
        "epoch": int(epoch),
        "metrics": metrics,
    }
    torch.save(payload, str(path))


def _append_history_row(path: Path, row: dict[str, float | int]) -> None:
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    out: dict[str, float] = {}
    keys = sorted({k for row in rows for k in row})
    for key in keys:
        values = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
        if values:
            out[key] = float(sum(values) / len(values))
    return out


def _resolve_device(name: str) -> torch.device:
    name = str(name).lower()
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available")
        return torch.device("mps")
    raise ValueError(f"Unsupported device: {name}")


def _prepare_out_dir(out_dir: Path, *, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {out_dir}. Use --overwrite to replace it.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.image_size) <= 0:
        raise ValueError("--image-size must be > 0")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be > 0")
    if int(args.epochs) <= 0:
        raise ValueError("--epochs must be > 0")
    if float(args.lr) <= 0.0:
        raise ValueError("--lr must be > 0")
    if not (0.0 <= float(args.val_fraction) <= 0.9):
        raise ValueError("--val-fraction must be in [0, 0.9]")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())

