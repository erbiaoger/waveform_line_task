"""Dataset helpers for waveform-line PNG segmentation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image


@dataclass(frozen=True)
class SampleRecord:
    """One manifest row resolved relative to a dataset root."""

    sample_index: int
    image_path: Path
    label_path: Path
    preview_path: str
    vehicle_count: int
    visible_point_count: int
    seed: int
    dead_channel_count: int
    missing_block_count: int


def load_manifest(data_dir: Path) -> list[SampleRecord]:
    """Load sample records from a waveform-line dataset manifest."""

    manifest_path = Path(data_dir).expanduser() / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.csv not found: {manifest_path}")
    rows: list[SampleRecord] = []
    with manifest_path.open(newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            rows.append(
                SampleRecord(
                    sample_index=int(row["sample_index"]),
                    image_path=(manifest_path.parent / row["image_path"]).resolve(),
                    label_path=(manifest_path.parent / row["label_path"]).resolve(),
                    preview_path=str(row.get("preview_path", "")),
                    vehicle_count=int(row.get("vehicle_count", 0)),
                    visible_point_count=int(row.get("visible_point_count", 0)),
                    seed=int(row.get("seed", 0)),
                    dead_channel_count=int(row.get("dead_channel_count", 0)),
                    missing_block_count=int(row.get("missing_block_count", 0)),
                )
            )
    if not rows:
        raise ValueError(f"No records found in {manifest_path}")
    return rows


def split_records(records: list[SampleRecord], val_fraction: float) -> tuple[list[SampleRecord], list[SampleRecord]]:
    """Split records by preserving manifest order and reserving the tail for validation."""

    if not records:
        raise ValueError("No records to split")
    frac = max(0.0, min(0.9, float(val_fraction)))
    val_count = int(round(len(records) * frac))
    if val_count <= 0:
        return list(records), []
    val_count = min(len(records) - 1, val_count)
    return list(records[:-val_count]), list(records[-val_count:])


class WaveformLineSegmentationDataset(Dataset):
    """Read grayscale waveform PNGs and binary label PNGs."""

    def __init__(
        self,
        records: list[SampleRecord],
        *,
        image_size: int,
        include_labels: bool,
    ) -> None:
        self.records = list(records)
        self.image_size = int(image_size)
        self.include_labels = bool(include_labels)
        if self.image_size <= 0:
            raise ValueError("image_size must be > 0")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[int(index)]
        image = _read_gray_image(record.image_path)
        original_size = (int(image.shape[-2]), int(image.shape[-1]))
        image = _resize_tensor(image, self.image_size, mode="bilinear")

        item: dict[str, Any] = {
            "image": image.contiguous(),
            "sample_index": int(record.sample_index),
            "image_path": str(record.image_path),
            "original_size": original_size,
            "stem": record.image_path.stem,
        }
        if self.include_labels:
            mask = _read_gray_image(record.label_path)
            mask = (mask >= 0.5).to(torch.float32)
            mask = _resize_tensor(mask, self.image_size, mode="nearest")
            item["mask"] = (mask >= 0.5).to(torch.float32).contiguous()
            item["label_path"] = str(record.label_path)
        return item


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Simple collate that stacks tensor fields and keeps metadata as lists."""

    out: dict[str, Any] = {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "sample_index": [int(item["sample_index"]) for item in batch],
        "image_path": [str(item["image_path"]) for item in batch],
        "original_size": [tuple(item["original_size"]) for item in batch],
        "stem": [str(item["stem"]) for item in batch],
    }
    if "mask" in batch[0]:
        out["mask"] = torch.stack([item["mask"] for item in batch], dim=0)
        out["label_path"] = [str(item["label_path"]) for item in batch]
    return out


def _resize_tensor(tensor: torch.Tensor, image_size: int, *, mode: str) -> torch.Tensor:
    if int(tensor.shape[-2]) == int(image_size) and int(tensor.shape[-1]) == int(image_size):
        return tensor
    tensor = tensor.unsqueeze(0)
    if mode == "nearest":
        resized = F.interpolate(tensor, size=(int(image_size), int(image_size)), mode=mode)
    else:
        resized = F.interpolate(tensor, size=(int(image_size), int(image_size)), mode=mode, align_corners=False)
    return resized.squeeze(0)


def _read_gray_image(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        gray = image.convert("L")
        arr = np.asarray(gray, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)
