"""Convert a real DAS `.npy` array into waveform-line model input PNGs.

Purpose:
    This script converts an unlabeled real DAS array into the exact input-image
    style used by the current waveform-line model in this project:

        white background + vertical gray waveform traces + no axes

    The output is written as a minimal dataset directory so it can be inspected
    directly or passed to `predict_model.py` later.

Input:
    - A real `.npy` file stored either as `[time, channel]` or `[channel, time]`.
    - The array is segmented with a sliding time window.
    - Each window is converted into one grayscale PNG under `images/`.

Output:
    <out-dir>/
      README.md
      meta.json
      manifest.csv
      images/sample_000000.png
      labels/sample_000000.png
      previews/sample_000000_overlay.png

    Notes:
    - `labels/` are placeholder all-black masks because this is unlabeled data.
    - `manifest.csv` keeps the same schema as training datasets so
      `predict_model.py` can reuse it without special handling.
    - `previews/` are just image-plus-empty-mask overlays for quick inspection.

Examples:
    Convert the first four 120 s windows from `gauss_section.npy`:

        uv run python convert_real_npy_to_dataset.py \
          --input /Volumes/SanDisk2T4/MyProjects/BaFang/xi/saved_arrays04/gauss_section.npy \
          --out-dir datasets/real_gauss_section_preview \
          --array-layout time_channel \
          --fs 1000 \
          --window-seconds 120 \
          --stride-seconds 60 \
          --num-windows 4 \
          --overwrite

    Start from a later window:

        uv run python convert_real_npy_to_dataset.py \
          --input /path/to/real.npy \
          --out-dir datasets/real_preview \
          --start-window-index 10 \
          --num-windows 6 \
          --overwrite

How to run:
    1. `cd /Volumes/SanDisk2T4/MyProjects/BaFang/KF/waveform_line_task`
    2. `source .venv/bin/activate`
    3. Run one of the commands above, or use the shell wrapper in
       `convert_gauss_section_to_waveform_line.sh`.

Result interpretation:
    - `images/` are the model-ready input PNGs.
    - `meta.json` records the windowing and render settings used.
    - `manifest.csv` records one row per output window.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from render import RenderConfig, render_preview, render_waveform_image, save_gray_png, save_preview_png
else:
    from .render import RenderConfig, render_preview, render_waveform_image, save_gray_png, save_preview_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a real DAS .npy array into waveform-line model input PNGs.")
    parser.add_argument("--input", required=True, type=Path, help="Input `.npy` path.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output dataset directory.")
    parser.add_argument(
        "--array-layout",
        default="time_channel",
        choices=["time_channel", "channel_time"],
        help="Array memory layout inside the `.npy` file.",
    )
    parser.add_argument("--fs", type=float, default=1000.0, help="Sample rate in Hz.")
    parser.add_argument("--window-seconds", type=float, default=120.0, help="Window duration in seconds.")
    parser.add_argument("--stride-seconds", type=float, default=60.0, help="Sliding-window stride in seconds.")
    parser.add_argument("--start-window-index", type=int, default=0, help="First sliding-window index to export.")
    parser.add_argument("--num-windows", type=int, default=4, help="How many windows to export.")
    parser.add_argument("--channel-start", type=int, default=0, help="First channel to keep.")
    parser.add_argument("--channel-count", type=int, default=0, help="Number of channels to keep. `0` means keep all available channels from `channel-start` onward.")
    parser.add_argument(
        "--center-mode",
        default="channel_median",
        choices=["channel_median", "channel_mean", "global_median", "global_mean", "none"],
        help="How to recenter each window before rendering.",
    )
    parser.add_argument("--image-size", type=int, default=512, help="Output PNG size in pixels.")
    parser.add_argument("--waveform-line-width", type=int, default=1, help="Waveform trace width in pixels.")
    parser.add_argument("--wiggle-fraction", type=float, default=0.28, help="Horizontal wiggle amplitude as a fraction of channel spacing.")
    parser.add_argument("--robust-percentile", type=float, default=99.5, help="Percentile for waveform display scaling.")
    parser.add_argument("--preview-count", type=int, default=4, help="How many overlay preview PNGs to write.")
    parser.add_argument("--image-prefix", default="sample", help="Output PNG filename prefix.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_args(args)

    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    _prepare_out_dir(out_dir, overwrite=bool(args.overwrite))

    array = np.load(str(input_path), mmap_mode="r")
    channel_time = _as_channel_time(array, layout=str(args.array_layout))
    selected = _select_channels(channel_time, channel_start=int(args.channel_start), channel_count=int(args.channel_count))

    fs = float(args.fs)
    window_samples = int(round(float(args.window_seconds) * fs))
    stride_samples = int(round(float(args.stride_seconds) * fs))
    total_samples = int(selected.shape[1])
    total_windows = _count_windows(total_samples=total_samples, window_samples=window_samples, stride_samples=stride_samples)
    start_index = int(args.start_window_index)
    end_index = min(total_windows, start_index + int(args.num_windows))
    if start_index >= total_windows:
        raise ValueError(f"--start-window-index={start_index} is out of range; total windows={total_windows}")

    render_config = RenderConfig(
        image_size=int(args.image_size),
        waveform_line_width=int(args.waveform_line_width),
        robust_percentile=float(args.robust_percentile),
        wiggle_fraction=float(args.wiggle_fraction),
    )

    rows: list[dict[str, Any]] = []
    empty_label = np.zeros((int(args.image_size), int(args.image_size)), dtype=np.uint8)
    for sample_index, window_index in enumerate(range(start_index, end_index)):
        start_sample = window_index * stride_samples
        stop_sample = start_sample + window_samples
        window = np.asarray(selected[:, start_sample:stop_sample], dtype=np.float32)
        window = _center_window(window, mode=str(args.center_mode))
        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)

        stem = f"{str(args.image_prefix)}_{sample_index:06d}"
        image_rel = Path("images") / f"{stem}.png"
        label_rel = Path("labels") / f"{stem}.png"
        preview_rel = Path("previews") / f"{stem}_overlay.png"

        waveform_image = render_waveform_image(window, render_config)
        save_gray_png(out_dir / image_rel, waveform_image)
        save_gray_png(out_dir / label_rel, empty_label)

        preview_path = ""
        if sample_index < int(args.preview_count):
            preview = render_preview(waveform_image, empty_label)
            save_preview_png(out_dir / preview_rel, preview)
            preview_path = str(preview_rel)

        rows.append(
            {
                "sample_index": sample_index,
                "image_path": str(image_rel),
                "label_path": str(label_rel),
                "preview_path": preview_path,
                "vehicle_count": 0,
                "visible_point_count": 0,
                "seed": window_index,
                "dead_channel_count": 0,
                "missing_block_count": 0,
                "window_index": window_index,
                "start_sample": start_sample,
                "stop_sample": stop_sample,
                "start_seconds": start_sample / fs,
                "stop_seconds": stop_sample / fs,
            }
        )

    _write_manifest(out_dir / "manifest.csv", rows)
    _write_meta(
        out_dir / "meta.json",
        input_path=input_path,
        args=args,
        array_shape=list(map(int, array.shape)),
        channel_time_shape=list(map(int, selected.shape)),
        total_windows=total_windows,
        exported_windows=len(rows),
    )
    _write_readme(out_dir, args=args, total_windows=total_windows, rows=rows)
    print(f"done: exported={len(rows)}, total_windows={total_windows}, out_dir={out_dir}", flush=True)
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    if float(args.fs) <= 0.0:
        raise ValueError("--fs must be > 0")
    if float(args.window_seconds) <= 0.0:
        raise ValueError("--window-seconds must be > 0")
    if float(args.stride_seconds) <= 0.0:
        raise ValueError("--stride-seconds must be > 0")
    if int(args.start_window_index) < 0:
        raise ValueError("--start-window-index must be >= 0")
    if int(args.num_windows) <= 0:
        raise ValueError("--num-windows must be > 0")
    if int(args.channel_start) < 0:
        raise ValueError("--channel-start must be >= 0")
    if int(args.channel_count) < 0:
        raise ValueError("--channel-count must be >= 0")
    if int(args.image_size) <= 0:
        raise ValueError("--image-size must be > 0")
    if int(args.waveform_line_width) <= 0:
        raise ValueError("--waveform-line-width must be > 0")
    if not (0.0 < float(args.robust_percentile) <= 100.0):
        raise ValueError("--robust-percentile must be in (0, 100]")


def _prepare_out_dir(out_dir: Path, *, overwrite: bool) -> None:
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {out_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    (out_dir / "previews").mkdir(parents=True, exist_ok=True)


def _as_channel_time(array: np.ndarray, *, layout: str) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim != 2:
        raise ValueError(f"Input array must be rank 2, got shape={arr.shape}")
    if layout == "time_channel":
        return arr.T
    if layout == "channel_time":
        return arr
    raise ValueError(f"Unsupported layout: {layout}")


def _select_channels(array: np.ndarray, *, channel_start: int, channel_count: int) -> np.ndarray:
    total_channels = int(array.shape[0])
    start = int(channel_start)
    if start >= total_channels:
        raise ValueError(f"--channel-start={start} exceeds total channels={total_channels}")
    stop = total_channels if int(channel_count) == 0 else min(total_channels, start + int(channel_count))
    if stop <= start:
        raise ValueError("Selected channel range is empty")
    return array[start:stop]


def _count_windows(*, total_samples: int, window_samples: int, stride_samples: int) -> int:
    if window_samples <= 0 or stride_samples <= 0:
        raise ValueError("window_samples and stride_samples must be > 0")
    if total_samples < window_samples:
        raise ValueError(f"Input is shorter than one window: total_samples={total_samples}, window_samples={window_samples}")
    return 1 + (int(total_samples) - int(window_samples)) // int(stride_samples)


def _center_window(window: np.ndarray, *, mode: str) -> np.ndarray:
    arr = np.asarray(window, dtype=np.float32)
    if mode == "none":
        return arr
    if mode == "channel_median":
        offset = np.nanmedian(arr, axis=1, keepdims=True)
        return arr - offset
    if mode == "channel_mean":
        offset = np.nanmean(arr, axis=1, keepdims=True)
        return arr - offset
    if mode == "global_median":
        return arr - np.float32(np.nanmedian(arr))
    if mode == "global_mean":
        return arr - np.float32(np.nanmean(arr))
    raise ValueError(f"Unsupported center mode: {mode}")


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sample_index",
        "image_path",
        "label_path",
        "preview_path",
        "vehicle_count",
        "visible_point_count",
        "seed",
        "dead_channel_count",
        "missing_block_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def _write_meta(
    path: Path,
    *,
    input_path: Path,
    args: argparse.Namespace,
    array_shape: list[int],
    channel_time_shape: list[int],
    total_windows: int,
    exported_windows: int,
) -> None:
    meta = {
        "format": "waveform_line_png_v1_unlabeled",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "array_layout": str(args.array_layout),
        "input_shape": array_shape,
        "selected_channel_time_shape": channel_time_shape,
        "fs": float(args.fs),
        "window_seconds": float(args.window_seconds),
        "stride_seconds": float(args.stride_seconds),
        "start_window_index": int(args.start_window_index),
        "num_windows_requested": int(args.num_windows),
        "num_windows_exported": int(exported_windows),
        "total_windows_available": int(total_windows),
        "channel_start": int(args.channel_start),
        "channel_count": int(args.channel_count),
        "center_mode": str(args.center_mode),
        "image_size": int(args.image_size),
        "waveform_line_width": int(args.waveform_line_width),
        "wiggle_fraction": float(args.wiggle_fraction),
        "robust_percentile": float(args.robust_percentile),
        "preview_count": int(args.preview_count),
    }
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_readme(out_dir: Path, *, args: argparse.Namespace, total_windows: int, rows: list[dict[str, Any]]) -> None:
    exported = len(rows)
    window_span = "n/a"
    if rows:
        start_s = float(rows[0]["start_seconds"])
        stop_s = float(rows[-1]["stop_seconds"])
        window_span = f"{start_s:.3f} s -> {stop_s:.3f} s"
    text = f"""# Real NPY Preview Dataset

This folder was generated from a real unlabeled DAS `.npy` array for the
current waveform-line model.

## Source

- `input`: `{Path(args.input).expanduser()}`
- `array-layout`: `{args.array_layout}`
- `fs`: `{float(args.fs):.6f}` Hz
- `window-seconds`: `{float(args.window_seconds):.6f}` s
- `stride-seconds`: `{float(args.stride_seconds):.6f}` s
- `channel-start`: `{int(args.channel_start)}`
- `channel-count`: `{int(args.channel_count)}`
- `center-mode`: `{args.center_mode}`

## Export Summary

- `total-windows-available`: `{int(total_windows)}`
- `num-windows-exported`: `{int(exported)}`
- `time-span-covered`: `{window_span}`

## Files

- `images/`: model-ready grayscale waveform PNGs.
- `labels/`: placeholder all-black masks for compatibility with `manifest.csv`.
- `previews/`: overlay previews with empty labels.
- `manifest.csv`: dataset rows compatible with `predict_model.py`.
- `meta.json`: full conversion settings.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
