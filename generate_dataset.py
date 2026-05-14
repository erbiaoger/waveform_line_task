"""Generate paired waveform and vehicle-line PNG images.

Purpose:
    Build an independent image dataset for the task:

        original DAS-like waveform image -> vehicle-line-only label image

    All synthesis, labels, and rendering are implemented inside
    `waveform_line_task/`. This script does not import `autotrack` or any
    existing project code.

Usage:
    uv run python waveform_line_task/generate_dataset.py \
        --out-dir waveform_line_task/datasets/v1_train \
        --num-samples 1000 \
        --image-size 1024 \
        --workers 8 \
        --overwrite

Outputs:
    <out-dir>/meta.json
        Dataset format, generator configuration, render configuration, and
        creation metadata.
    <out-dir>/manifest.csv
        One row per sample with image path, label path, vehicle count,
        visible label point count, and seed.
    <out-dir>/images/sample_000000.png
        White-background raw waveform trace image.
    <out-dir>/labels/sample_000000.png
        Black-background binary vehicle-line label image.
    <out-dir>/previews/sample_000000_overlay.png
        Optional RGB overlay preview for manual checking.

Notes:
    Use `--device cpu` for portable CPU generation. `--device cuda` and
    `--device mps` run waveform synthesis math on that torch device when the
    backend is available; PNG rendering and writing remain CPU-bound.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from render import RenderConfig, render_label_image, render_preview, render_waveform_image, save_gray_png, save_preview_png
    from synth import SynthConfig, generate_sample, resolve_torch_device
else:
    from .render import RenderConfig, render_label_image, render_preview, render_waveform_image, save_gray_png, save_preview_png
    from .synth import SynthConfig, generate_sample, resolve_torch_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate independent waveform-to-vehicle-line PNG dataset.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output dataset directory.")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of image/label pairs to generate.")
    parser.add_argument("--image-size", type=int, default=1024, help="Square PNG size in pixels.")
    parser.add_argument("--workers", type=int, default=8, help="CPU worker processes. Forced to 1 when device is cuda/mps.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"], help="Synthesis device. Rendering and PNG writing use CPU.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    parser.add_argument("--preview-count", type=int, default=16, help="Number of overlay preview PNGs to write. Use 0 to disable.")
    parser.add_argument("--image-prefix", default="sample", help="Filename prefix for generated PNGs.")

    parser.add_argument("--n-channels", type=int, default=50, help="Number of DAS channels.")
    parser.add_argument("--time-bins", type=int, default=2048, help="Synthetic time samples per channel before rendering.")
    parser.add_argument("--window-seconds", type=float, default=120.0, help="Synthetic window duration.")
    parser.add_argument("--dx-m", type=float, default=100.0, help="Channel spacing in meters.")
    parser.add_argument("--vehicles-min", type=int, default=24, help="Minimum vehicles per sample.")
    parser.add_argument("--vehicles-max", type=int, default=36, help="Maximum vehicles per sample.")
    parser.add_argument("--speed-min-kmh", type=float, default=70.0, help="Minimum vehicle speed.")
    parser.add_argument("--speed-max-kmh", type=float, default=85.0, help="Maximum vehicle speed.")
    parser.add_argument("--primary-ratio", type=float, default=0.8333333333, help="Fraction of forward-direction vehicles.")
    parser.add_argument("--min-visible-channels", type=int, default=4, help="Minimum visible channels kept per vehicle label.")

    parser.add_argument("--noise-std", type=float, default=0.05, help="White noise standard deviation.")
    parser.add_argument("--colored-noise-std", type=float, default=0.06, help="Time-correlated noise standard deviation.")
    parser.add_argument("--baseline-drift-std", type=float, default=0.035, help="Slow baseline drift standard deviation.")
    parser.add_argument("--isolated-noise-rate", type=float, default=180.0, help="Expected isolated Gaussian noise peaks per sample.")
    parser.add_argument("--missing-block-rate", type=float, default=30.0, help="Expected missing channel-time blocks per enabled sample.")

    parser.add_argument("--waveform-line-width", type=int, default=1, help="Input waveform trace width in pixels.")
    parser.add_argument("--label-line-width", type=int, default=2, help="Binary vehicle label line width in pixels.")
    parser.add_argument("--wiggle-fraction", type=float, default=0.28, help="Trace wiggle amplitude as a fraction of channel spacing.")
    parser.add_argument("--robust-percentile", type=float, default=99.5, help="Percentile for waveform display scaling.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_args(args)

    device_name = str(args.device)
    resolve_torch_device(device_name)
    workers = int(max(1, args.workers))
    if device_name != "cpu" and workers != 1:
        print(f"--device {device_name} uses one process to avoid multiple GPU writers; overriding workers={workers} to 1.", flush=True)
        workers = 1

    out_dir = Path(args.out_dir).expanduser()
    _prepare_out_dir(out_dir, overwrite=bool(args.overwrite))

    synth_config = _build_synth_config(args)
    render_config = _build_render_config(args)
    t0 = time.perf_counter()

    if workers <= 1:
        rows = [_write_one_sample(idx, out_dir, synth_config, render_config, args) for idx in range(int(args.num_samples))]
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_write_one_sample, idx, out_dir, synth_config, render_config, args)
                for idx in range(int(args.num_samples))
            ]
            for completed, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                if completed == 1 or completed == int(args.num_samples) or completed % max(1, int(args.num_samples) // 10) == 0:
                    print(f"generated {completed}/{int(args.num_samples)} samples", flush=True)
    rows = sorted(rows, key=lambda item: int(item["sample_index"]))

    _write_manifest(out_dir / "manifest.csv", rows)
    _write_meta(
        out_dir / "meta.json",
        args=args,
        synth_config=synth_config,
        render_config=render_config,
        rows=rows,
        workers=workers,
        elapsed_seconds=float(time.perf_counter() - t0),
    )
    _write_dataset_readme(out_dir, args)
    print(f"done: samples={len(rows)}, out_dir={out_dir}, elapsed={time.perf_counter() - t0:.1f}s", flush=True)
    return 0


def _build_synth_config(args: argparse.Namespace) -> SynthConfig:
    return SynthConfig(
        n_channels=int(args.n_channels),
        time_bins=int(args.time_bins),
        window_seconds=float(args.window_seconds),
        dx_m=float(args.dx_m),
        vehicles_min=int(args.vehicles_min),
        vehicles_max=int(args.vehicles_max),
        speed_min_kmh=float(args.speed_min_kmh),
        speed_max_kmh=float(args.speed_max_kmh),
        primary_ratio=float(args.primary_ratio),
        min_visible_channels=int(args.min_visible_channels),
        noise_std=float(args.noise_std),
        colored_noise_std=float(args.colored_noise_std),
        baseline_drift_std=float(args.baseline_drift_std),
        isolated_noise_rate=float(args.isolated_noise_rate),
        missing_block_rate=float(args.missing_block_rate),
    )


def _build_render_config(args: argparse.Namespace) -> RenderConfig:
    return RenderConfig(
        image_size=int(args.image_size),
        waveform_line_width=int(args.waveform_line_width),
        label_line_width=int(args.label_line_width),
        robust_percentile=float(args.robust_percentile),
        wiggle_fraction=float(args.wiggle_fraction),
    )


def _write_one_sample(
    index: int,
    out_dir: Path,
    synth_config: SynthConfig,
    render_config: RenderConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    sample = generate_sample(index, synth_config, seed=int(args.seed), device_name=str(args.device))
    stem = f"{str(args.image_prefix)}_{int(index):06d}"
    image_rel = Path("images") / f"{stem}.png"
    label_rel = Path("labels") / f"{stem}.png"
    preview_rel = Path("previews") / f"{stem}_overlay.png"

    waveform_image = render_waveform_image(sample.data, render_config)
    label_image = render_label_image(
        sample.tracks,
        n_channels=int(synth_config.n_channels),
        window_seconds=float(synth_config.window_seconds),
        config=render_config,
    )
    save_gray_png(out_dir / image_rel, waveform_image)
    save_gray_png(out_dir / label_rel, label_image)

    preview_path = ""
    if int(index) < int(args.preview_count):
        preview = render_preview(waveform_image, label_image)
        save_preview_png(out_dir / preview_rel, preview)
        preview_path = str(preview_rel)

    return {
        "sample_index": int(index),
        "image_path": str(image_rel),
        "label_path": str(label_rel),
        "preview_path": preview_path,
        "vehicle_count": int(sample.metadata["vehicle_count"]),
        "visible_point_count": int(sample.metadata["visible_point_count"]),
        "seed": int(sample.metadata["seed"]),
        "dead_channel_count": int(len(sample.metadata["dead_channels"])),
        "missing_block_count": int(sample.metadata["missing_block_count"]),
    }


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
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
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_meta(
    path: Path,
    *,
    args: argparse.Namespace,
    synth_config: SynthConfig,
    render_config: RenderConfig,
    rows: list[dict[str, Any]],
    workers: int,
    elapsed_seconds: float,
) -> None:
    meta = {
        "format": "waveform_line_png_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_samples": int(len(rows)),
        "image_size": int(render_config.image_size),
        "device": str(args.device),
        "workers": int(workers),
        "elapsed_seconds": float(elapsed_seconds),
        "synth_config": asdict(synth_config),
        "render_config": asdict(render_config),
        "generator_args": _json_ready(vars(args)),
        "totals": {
            "vehicle_count": int(sum(int(row["vehicle_count"]) for row in rows)),
            "visible_point_count": int(sum(int(row["visible_point_count"]) for row in rows)),
        },
    }
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_dataset_readme(out_dir: Path, args: argparse.Namespace) -> None:
    text = f"""# Generated Waveform-Line Dataset

This folder was generated by `waveform_line_task/generate_dataset.py`.

## Folders

- `images/`: input waveform PNG files.
- `labels/`: binary vehicle-line label PNG files aligned with `images/`.
- `previews/`: optional RGB overlays for manual checking.

## Files

- `manifest.csv`: image/label path pairs and sample-level counts.
- `meta.json`: generation configuration and summary metadata.
- `README.md`: this generated dataset note.

## Recreate

```sh
uv run python waveform_line_task/generate_dataset.py \\
  --out-dir {out_dir} \\
  --num-samples {int(args.num_samples)} \\
  --image-size {int(args.image_size)} \\
  --workers {int(args.workers)} \\
  --overwrite
```
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def _prepare_out_dir(out_dir: Path, *, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        if not bool(overwrite):
            raise FileExistsError(f"Output directory is not empty: {out_dir}. Use --overwrite to replace it.")
        shutil.rmtree(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    (out_dir / "previews").mkdir(parents=True, exist_ok=True)


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.num_samples) <= 0:
        raise ValueError("--num-samples must be > 0")
    if int(args.image_size) <= 16:
        raise ValueError("--image-size must be > 16")
    if int(args.n_channels) <= 0:
        raise ValueError("--n-channels must be > 0")
    if int(args.time_bins) <= 8:
        raise ValueError("--time-bins must be > 8")
    if float(args.window_seconds) <= 0.0:
        raise ValueError("--window-seconds must be > 0")
    if int(args.vehicles_min) < 0 or int(args.vehicles_max) < int(args.vehicles_min):
        raise ValueError("--vehicles-max must be >= --vehicles-min >= 0")
    if float(args.speed_min_kmh) <= 0.0 or float(args.speed_max_kmh) < float(args.speed_min_kmh):
        raise ValueError("speed range must be positive and ordered")
    if not (0.0 <= float(args.primary_ratio) <= 1.0):
        raise ValueError("--primary-ratio must be in [0, 1]")
    if int(args.label_line_width) <= 0 or int(args.waveform_line_width) <= 0:
        raise ValueError("line widths must be > 0")
    if not (0.0 < float(args.robust_percentile) <= 100.0):
        raise ValueError("--robust-percentile must be in (0, 100]")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
