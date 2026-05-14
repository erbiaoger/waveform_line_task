"""Image rendering utilities for the independent waveform-line task.

The image and label renderers share the same channel/time-to-pixel mapping.
This keeps the input waveform PNG and the binary vehicle-line PNG aligned at
the pixel level.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .synth import VehicleTrack
except ImportError:
    from synth import VehicleTrack


@dataclass(frozen=True)
class RenderConfig:
    """Rendering parameters for paired waveform and vehicle-line images."""

    image_size: int = 512
    waveform_line_width: int = 1
    label_line_width: int = 2
    waveform_color: int = 40
    background_color: int = 255
    label_color: int = 255
    robust_percentile: float = 99.5
    wiggle_fraction: float = 0.28


def render_waveform_image(data: np.ndarray, config: RenderConfig) -> np.ndarray:
    """Render channel-time waveform traces as a grayscale uint8 image."""

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"data must have shape [channel, time], got {arr.shape}")
    n_ch, n_t = int(arr.shape[0]), int(arr.shape[1])
    size = int(config.image_size)
    image = np.full((size, size), int(config.background_color), dtype=np.uint8)
    if n_ch <= 0 or n_t <= 0:
        return image

    y_source = np.linspace(0.0, float(n_t - 1), size, dtype=np.float32)
    x_centers = channel_x_positions(n_ch, size)
    spacing = float(np.median(np.diff(x_centers))) if n_ch >= 2 else float(size) * 0.5
    wiggle = max(1.0, float(config.wiggle_fraction) * spacing)
    finite = arr[np.isfinite(arr)]
    scale = float(np.percentile(np.abs(finite), float(config.robust_percentile))) if finite.size else 1.0
    scale = max(scale, 1e-6)

    for ch in range(n_ch):
        trace = np.interp(y_source, np.arange(n_t, dtype=np.float32), arr[ch]).astype(np.float32)
        trace = np.clip(trace / scale, -1.35, 1.35)
        xs = x_centers[ch] + trace * wiggle
        _draw_vertical_polyline(image, xs, color=int(config.waveform_color), width=int(config.waveform_line_width))
    return image


def render_label_image(
    tracks: list[VehicleTrack],
    *,
    n_channels: int,
    window_seconds: float,
    config: RenderConfig,
) -> np.ndarray:
    """Render vehicle tracks as a black-background binary uint8 label image."""

    size = int(config.image_size)
    mask = np.zeros((size, size), dtype=np.uint8)
    x_centers = channel_x_positions(int(n_channels), size)
    for track in tracks:
        points = []
        for ch in range(int(n_channels)):
            t_s = float(track.times_s[int(ch)])
            if not (0.0 <= t_s <= float(window_seconds)):
                continue
            x = float(x_centers[int(ch)])
            y = t_s / float(window_seconds) * float(size - 1)
            points.append((int(ch), x, y))
        if len(points) < 2:
            continue
        points = sorted(points, key=lambda item: item[0])
        for left, right in zip(points[:-1], points[1:]):
            _draw_line(
                mask,
                int(round(left[1])),
                int(round(left[2])),
                int(round(right[1])),
                int(round(right[2])),
                color=int(config.label_color),
                width=int(config.label_line_width),
            )
    return mask


def render_preview(waveform_image: np.ndarray, label_image: np.ndarray) -> np.ndarray:
    """Create a simple RGB overlay for manual inspection."""

    base = np.asarray(waveform_image, dtype=np.uint8)
    label = np.asarray(label_image, dtype=np.uint8) > 0
    rgb = np.stack([base, base, base], axis=0).astype(np.uint8)
    rgb[0, label] = 255
    rgb[1, label] = np.minimum(rgb[1, label] // 3, 80)
    rgb[2, label] = np.minimum(rgb[2, label] // 3, 80)
    return rgb


def save_gray_png(path: Path, image: np.ndarray) -> None:
    """Write a grayscale PNG from a [H, W] uint8 array."""

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError(f"gray image must have shape [H, W], got {arr.shape}")
    Image.fromarray(arr, mode="L").save(path, format="PNG")


def save_rgb_png(path: Path, image: np.ndarray) -> None:
    """Write an RGB PNG from either [3, H, W] or [H, W, 3] uint8 data."""

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim != 3:
        raise ValueError(f"RGB image must be rank 3, got {arr.shape}")
    if arr.shape[0] == 3:
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] != 3:
        raise ValueError(f"RGB image must have 3 channels, got {arr.shape}")
    Image.fromarray(arr, mode="RGB").save(path, format="PNG")


def save_preview_png(path: Path, image: np.ndarray) -> None:
    """Write an overlay preview through Matplotlib with Times New Roman."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "Times New Roman", "axes.unicode_minus": False})
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim != 3:
        raise ValueError(f"preview image must be rank 3, got {arr.shape}")
    if arr.shape[0] == 3:
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] != 3:
        raise ValueError(f"preview image must have 3 channels, got {arr.shape}")

    dpi = 160
    height, width = int(arr.shape[0]), int(arr.shape[1])
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(arr, interpolation="nearest")
    ax.axis("off")
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3]
    plt.close(fig)
    save_rgb_png(path, rgb)


def channel_x_positions(n_channels: int, image_size: int) -> np.ndarray:
    """Return shared x pixel positions for all DAS channels."""

    n_channels = int(max(1, n_channels))
    image_size = int(image_size)
    margin = max(4.0, float(image_size) / float(n_channels + 1))
    if n_channels == 1:
        return np.array([0.5 * float(image_size - 1)], dtype=np.float32)
    return np.linspace(margin, float(image_size - 1) - margin, n_channels, dtype=np.float32)


def _draw_vertical_polyline(image: np.ndarray, xs: np.ndarray, *, color: int, width: int) -> None:
    prev_x = int(round(float(xs[0])))
    _paint_square(image, prev_x, 0, color=color, width=width)
    for y in range(1, int(image.shape[0])):
        x = int(round(float(xs[y])))
        _draw_line(image, prev_x, y - 1, x, y, color=color, width=width)
        prev_x = x


def _draw_line(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, *, color: int, width: int) -> None:
    dx = abs(int(x1) - int(x0))
    dy = -abs(int(y1) - int(y0))
    sx = 1 if int(x0) < int(x1) else -1
    sy = 1 if int(y0) < int(y1) else -1
    err = dx + dy
    x = int(x0)
    y = int(y0)
    while True:
        _paint_square(image, x, y, color=color, width=width)
        if x == int(x1) and y == int(y1):
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def _paint_square(image: np.ndarray, x: int, y: int, *, color: int, width: int) -> None:
    radius = max(0, int(width) // 2)
    h, w = int(image.shape[0]), int(image.shape[1])
    x0 = max(0, int(x) - radius)
    x1 = min(w, int(x) + radius + 1)
    y0 = max(0, int(y) - radius)
    y1 = min(h, int(y) + radius + 1)
    if x0 < x1 and y0 < y1:
        image[y0:y1, x0:x1] = np.uint8(color)
