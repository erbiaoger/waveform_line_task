"""Device and CUDA runtime helpers for waveform_line_task.

Purpose:
    Centralize device selection and runtime tuning used by dataset
    generation, training, and inference. The project now defaults to
    `auto`, which prefers CUDA, then MPS, and finally CPU.

Usage:
    from waveform_line_task.device_utils import resolve_runtime_device

    device = resolve_runtime_device("auto")

Notes:
    - `auto` resolves to `cuda` when available.
    - CUDA paths enable TF32 and cuDNN benchmark for fixed-size image
      workloads used in this task.
    - CPU remains supported by passing `--device cpu`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RuntimeDevice:
    """Resolved runtime device metadata."""

    torch_device: torch.device
    requested: str
    resolved: str
    use_cuda: bool
    use_mps: bool
    use_cpu: bool


def resolve_runtime_device(device_name: str | None) -> RuntimeDevice:
    """Resolve a requested device string into a concrete torch device."""

    requested = str(device_name or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return RuntimeDevice(torch.device("cuda"), requested, "cuda", True, False, False)
        if torch.backends.mps.is_available():
            return RuntimeDevice(torch.device("mps"), requested, "mps", False, True, False)
        return RuntimeDevice(torch.device("cpu"), requested, "cpu", False, False, True)
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        return RuntimeDevice(torch.device("cuda"), requested, "cuda", True, False, False)
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available")
        return RuntimeDevice(torch.device("mps"), requested, "mps", False, True, False)
    if requested == "cpu":
        return RuntimeDevice(torch.device("cpu"), requested, "cpu", False, False, True)
    raise ValueError(f"Unsupported device: {device_name}. Use auto, cuda, mps, or cpu.")


def configure_torch_runtime(device: RuntimeDevice) -> None:
    """Apply stable runtime settings for the resolved device."""

    if device.use_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

