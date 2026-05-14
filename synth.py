"""Synthetic DAS-like waveform generator for the independent waveform-line task.

This module intentionally does not import any code from the surrounding
project. It creates one synthetic channel-time waveform array and a list of
vehicle trajectory labels that can be rendered into paired training images.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SynthConfig:
    """Configuration for one synthetic waveform window."""

    n_channels: int = 50
    time_bins: int = 2048
    window_seconds: float = 120.0
    dx_m: float = 100.0
    vehicles_min: int = 24
    vehicles_max: int = 36
    speed_min_kmh: float = 70.0
    speed_max_kmh: float = 85.0
    primary_ratio: float = 0.8333333333
    min_visible_channels: int = 4
    amp_min: float = 5.0
    amp_max: float = 8.0
    sigma_min_s: float = 0.18
    sigma_max_s: float = 0.32
    speed_jitter_frac: float = 0.04
    stop_go_ratio: float = 0.04
    stop_duration_min_s: float = 1.0
    stop_duration_max_s: float = 6.0
    interaction_ratio: float = 0.30
    noise_std: float = 0.05
    colored_noise_std: float = 0.06
    colored_noise_corr_s: float = 0.6
    baseline_drift_std: float = 0.035
    baseline_drift_corr_s: float = 6.0
    channel_bias_std: float = 0.025
    channel_gain_std: float = 0.08
    isolated_noise_rate: float = 180.0
    isolated_noise_amp_min: float = 1.0
    isolated_noise_amp_max: float = 5.5
    isolated_noise_sigma_min_s: float = 0.05
    isolated_noise_sigma_max_s: float = 0.30
    random_dead_channel_ratio: float = 0.25
    random_dead_channel_min: int = 1
    random_dead_channel_max: int = 8
    missing_block_ratio: float = 0.80
    missing_block_rate: float = 30.0
    missing_block_channel_min: int = 1
    missing_block_channel_max: int = 4
    missing_block_duration_min_s: float = 0.5
    missing_block_duration_max_s: float = 4.0


@dataclass
class VehicleTrack:
    """One vehicle trajectory represented as one time value per channel."""

    track_id: int
    direction: int
    speed_kmh: float
    times_s: np.ndarray
    visible: np.ndarray

    @property
    def visible_count(self) -> int:
        return int(np.count_nonzero(self.visible))


@dataclass
class WaveformSample:
    """Generated waveform and labels for one training pair."""

    data: np.ndarray
    tracks: list[VehicleTrack]
    metadata: dict[str, Any]


def resolve_torch_device(device_name: str) -> torch.device:
    """Resolve and validate a torch device name used by this new task."""

    name = str(device_name or "cpu").lower()
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available.")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("--device mps was requested, but MPS is not available.")
        return torch.device("mps")
    raise ValueError(f"Unsupported device: {device_name}. Use cpu, cuda, or mps.")


def generate_sample(sample_index: int, config: SynthConfig, *, seed: int, device_name: str = "cpu") -> WaveformSample:
    """Generate one synthetic waveform sample and its vehicle-line labels."""

    sample_seed = int(seed) + int(sample_index) * 1_000_003
    rng = np.random.default_rng(sample_seed)
    device = resolve_torch_device(device_name)
    cfg = config

    data = torch.zeros((int(cfg.n_channels), int(cfg.time_bins)), dtype=torch.float32, device=device)
    _add_background_noise(data, cfg, rng, device)

    tracks: list[VehicleTrack] = []
    if rng.random() < float(cfg.interaction_ratio):
        tracks.extend(_make_interaction_tracks(cfg, rng, start_id=0))

    target_count = int(rng.integers(int(cfg.vehicles_min), int(cfg.vehicles_max) + 1))
    attempts = 0
    while len(tracks) < target_count and attempts < max(256, target_count * 64):
        attempts += 1
        track = _make_random_track(cfg, rng, track_id=len(tracks))
        if track.visible_count >= int(cfg.min_visible_channels):
            tracks.append(track)

    for track in tracks:
        _add_vehicle_signal(data, cfg, rng, track)

    _add_isolated_noise(data, cfg, rng)
    missing_blocks = _apply_missing_blocks(data, cfg, rng, tracks)
    dead_channels = _apply_dead_channels(data, cfg, rng, tracks)
    _apply_channel_gain(data, cfg, rng, device)

    final_tracks: list[VehicleTrack] = []
    for track in tracks:
        if track.visible_count >= int(cfg.min_visible_channels):
            final_tracks.append(
                VehicleTrack(
                    track_id=len(final_tracks),
                    direction=int(track.direction),
                    speed_kmh=float(track.speed_kmh),
                    times_s=track.times_s.astype(np.float32, copy=True),
                    visible=track.visible.astype(bool, copy=True),
                )
            )

    data_cpu = data.detach().to("cpu", dtype=torch.float32).numpy()
    metadata = {
        "sample_index": int(sample_index),
        "seed": int(sample_seed),
        "vehicle_count": int(len(final_tracks)),
        "visible_point_count": int(sum(track.visible_count for track in final_tracks)),
        "dead_channels": [int(ch) for ch in dead_channels],
        "missing_block_count": int(len(missing_blocks)),
    }
    return WaveformSample(data=data_cpu, tracks=final_tracks, metadata=metadata)


def _make_random_track(cfg: SynthConfig, rng: np.random.Generator, track_id: int) -> VehicleTrack:
    direction = 0 if rng.random() < float(cfg.primary_ratio) else 1
    speed_kmh = float(rng.uniform(float(cfg.speed_min_kmh), float(cfg.speed_max_kmh)))
    anchor_ch = int(rng.integers(0, int(cfg.n_channels)))
    anchor_time = float(rng.uniform(0.0, float(cfg.window_seconds)))
    times = _integrate_track_times(cfg, rng, direction, speed_kmh, anchor_ch, anchor_time)
    visible = (times >= 0.0) & (times <= float(cfg.window_seconds))
    return VehicleTrack(
        track_id=int(track_id),
        direction=int(direction),
        speed_kmh=float(speed_kmh),
        times_s=times.astype(np.float32),
        visible=visible.astype(bool),
    )


def _make_interaction_tracks(cfg: SynthConfig, rng: np.random.Generator, start_id: int) -> list[VehicleTrack]:
    n_ch = int(cfg.n_channels)
    channels = np.arange(n_ch, dtype=np.float32)
    cross_ch = int(rng.integers(0, n_ch))
    cross_time = float(rng.uniform(0.08 * float(cfg.window_seconds), 0.92 * float(cfg.window_seconds)))
    kind = str(rng.choice(["crossing", "overtake", "near_parallel"]))
    speed_a = float(rng.uniform(float(cfg.speed_min_kmh), float(cfg.speed_max_kmh)))
    speed_b = float(rng.uniform(float(cfg.speed_min_kmh), float(cfg.speed_max_kmh)))
    dx = float(cfg.dx_m)

    if kind == "crossing":
        times_a = cross_time + (channels - float(cross_ch)) * dx / (speed_a / 3.6)
        times_b = cross_time - (channels - float(cross_ch)) * dx / (speed_b / 3.6)
        directions = [0, 1]
    else:
        offset = float(rng.uniform(-2.0, 2.0))
        times_a = cross_time + (channels - float(cross_ch)) * dx / (speed_a / 3.6)
        times_b = cross_time + offset + (channels - float(cross_ch)) * dx / (speed_b / 3.6)
        directions = [0, 0]

    out: list[VehicleTrack] = []
    for local_id, (times, direction, speed) in enumerate([(times_a, directions[0], speed_a), (times_b, directions[1], speed_b)]):
        visible = (times >= 0.0) & (times <= float(cfg.window_seconds))
        if int(np.count_nonzero(visible)) < int(cfg.min_visible_channels):
            continue
        out.append(
            VehicleTrack(
                track_id=int(start_id + local_id),
                direction=int(direction),
                speed_kmh=float(speed),
                times_s=times.astype(np.float32),
                visible=visible.astype(bool),
            )
        )
    return out


def _integrate_track_times(
    cfg: SynthConfig,
    rng: np.random.Generator,
    direction: int,
    base_speed_kmh: float,
    anchor_ch: int,
    anchor_time: float,
) -> np.ndarray:
    n_ch = int(cfg.n_channels)
    if n_ch <= 1:
        return np.full((n_ch,), float(anchor_time), dtype=np.float32)

    n_seg = n_ch - 1
    jitter = rng.normal(0.0, float(cfg.speed_jitter_frac), size=n_seg).astype(np.float32)
    if n_seg >= 3:
        kernel = np.ones((min(7, n_seg),), dtype=np.float32)
        kernel /= float(kernel.sum())
        jitter = np.convolve(jitter, kernel, mode="same")
    speed = np.clip(float(base_speed_kmh) * (1.0 + jitter), float(cfg.speed_min_kmh), float(cfg.speed_max_kmh))
    dt = float(cfg.dx_m) / (speed / 3.6)

    times = np.empty((n_ch,), dtype=np.float32)
    anchor_ch = int(np.clip(anchor_ch, 0, n_ch - 1))
    times[anchor_ch] = float(anchor_time)
    sign = 1.0 if int(direction) == 0 else -1.0
    for ch in range(anchor_ch, n_ch - 1):
        times[ch + 1] = times[ch] + sign * float(dt[ch])
    for ch in range(anchor_ch - 1, -1, -1):
        times[ch] = times[ch + 1] - sign * float(dt[ch])

    if rng.random() < float(cfg.stop_go_ratio):
        stop_ch = int(rng.integers(0, n_ch))
        delay = float(rng.uniform(float(cfg.stop_duration_min_s), float(cfg.stop_duration_max_s)))
        if int(direction) == 0:
            times[stop_ch + 1 :] += delay
        else:
            times[:stop_ch] += delay
    return times


def _add_background_noise(
    data: torch.Tensor,
    cfg: SynthConfig,
    rng: np.random.Generator,
    device: torch.device,
) -> None:
    n_ch, n_t = int(data.shape[0]), int(data.shape[1])
    if float(cfg.noise_std) > 0.0:
        white = rng.normal(0.0, float(cfg.noise_std), size=(n_ch, n_t)).astype(np.float32)
        data += torch.from_numpy(white).to(device=device)

    if float(cfg.colored_noise_std) > 0.0:
        colored = rng.normal(0.0, 1.0, size=(n_ch, n_t)).astype(np.float32)
        colored_t = torch.from_numpy(colored).to(device=device)
        corr = _seconds_to_bins(float(cfg.colored_noise_corr_s), cfg)
        colored_t = _smooth_time(colored_t, corr)
        colored_t = colored_t / torch.clamp(colored_t.std(dim=1, keepdim=True, unbiased=False), min=1e-6)
        data += float(cfg.colored_noise_std) * colored_t

    if float(cfg.baseline_drift_std) > 0.0:
        drift = rng.normal(0.0, 1.0, size=(n_ch, n_t)).astype(np.float32)
        drift_t = torch.from_numpy(drift).to(device=device)
        corr = _seconds_to_bins(float(cfg.baseline_drift_corr_s), cfg)
        drift_t = _smooth_time(drift_t, corr)
        drift_t = drift_t / torch.clamp(drift_t.std(dim=1, keepdim=True, unbiased=False), min=1e-6)
        data += float(cfg.baseline_drift_std) * drift_t

    if float(cfg.channel_bias_std) > 0.0:
        bias = rng.normal(0.0, float(cfg.channel_bias_std), size=(n_ch, 1)).astype(np.float32)
        data += torch.from_numpy(bias).to(device=device)


def _smooth_time(x: torch.Tensor, width: int) -> torch.Tensor:
    width = int(max(1, min(int(width), int(x.shape[-1]))))
    if width <= 1:
        return x
    pad_left = width // 2
    pad_right = width - 1 - pad_left
    padded = F.pad(x[:, None, :], (pad_left, pad_right), mode="replicate")
    return F.avg_pool1d(padded, kernel_size=width, stride=1).squeeze(1)


def _add_vehicle_signal(
    data: torch.Tensor,
    cfg: SynthConfig,
    rng: np.random.Generator,
    track: VehicleTrack,
) -> None:
    amp = float(rng.uniform(float(cfg.amp_min), float(cfg.amp_max)))
    sigma_s = float(rng.uniform(float(cfg.sigma_min_s), float(cfg.sigma_max_s)))
    for ch in np.where(track.visible)[0].tolist():
        _add_gaussian_pulse(data, cfg, int(ch), float(track.times_s[int(ch)]), amp, sigma_s)


def _add_isolated_noise(data: torch.Tensor, cfg: SynthConfig, rng: np.random.Generator) -> None:
    rate = max(0.0, float(cfg.isolated_noise_rate))
    count = int(rate)
    if rng.random() < rate - count:
        count += 1
    for _ in range(count):
        ch = int(rng.integers(0, int(cfg.n_channels)))
        center = float(rng.uniform(0.0, float(cfg.window_seconds)))
        amp = float(rng.uniform(float(cfg.isolated_noise_amp_min), float(cfg.isolated_noise_amp_max)))
        sigma_s = float(rng.uniform(float(cfg.isolated_noise_sigma_min_s), float(cfg.isolated_noise_sigma_max_s)))
        _add_gaussian_pulse(data, cfg, ch, center, amp, sigma_s)


def _add_gaussian_pulse(
    data: torch.Tensor,
    cfg: SynthConfig,
    channel: int,
    center_s: float,
    amp: float,
    sigma_s: float,
) -> None:
    n_t = int(data.shape[1])
    center = float(center_s) / float(cfg.window_seconds) * float(max(1, n_t - 1))
    sigma = max(1e-6, float(sigma_s) / float(cfg.window_seconds) * float(max(1, n_t - 1)))
    lo = max(0, int(np.floor(center - 4.0 * sigma)))
    hi = min(n_t, int(np.ceil(center + 4.0 * sigma)) + 1)
    if hi <= lo:
        return
    x = torch.arange(lo, hi, dtype=torch.float32, device=data.device)
    pulse = float(amp) * torch.exp(-0.5 * ((x - float(center)) / float(sigma)) ** 2)
    data[int(channel), lo:hi] += pulse


def _apply_channel_gain(
    data: torch.Tensor,
    cfg: SynthConfig,
    rng: np.random.Generator,
    device: torch.device,
) -> None:
    if float(cfg.channel_gain_std) <= 0.0:
        return
    gain = rng.normal(1.0, float(cfg.channel_gain_std), size=(int(cfg.n_channels), 1)).astype(np.float32)
    gain = np.clip(gain, 0.25, 3.0)
    data *= torch.from_numpy(gain).to(device=device)


def _apply_dead_channels(
    data: torch.Tensor,
    cfg: SynthConfig,
    rng: np.random.Generator,
    tracks: list[VehicleTrack],
) -> list[int]:
    if rng.random() >= float(cfg.random_dead_channel_ratio):
        return []
    lo = int(max(0, cfg.random_dead_channel_min))
    hi = int(max(lo, cfg.random_dead_channel_max))
    count = int(rng.integers(lo, hi + 1)) if hi > 0 else 0
    count = int(min(count, int(cfg.n_channels)))
    if count <= 0:
        return []
    dead = sorted(int(ch) for ch in rng.choice(int(cfg.n_channels), size=count, replace=False).tolist())
    data[dead, :] = 0.0
    for track in tracks:
        track.visible[dead] = False
    return dead


def _apply_missing_blocks(
    data: torch.Tensor,
    cfg: SynthConfig,
    rng: np.random.Generator,
    tracks: list[VehicleTrack],
) -> list[dict[str, int]]:
    if rng.random() >= float(cfg.missing_block_ratio):
        return []
    rate = max(0.0, float(cfg.missing_block_rate))
    count = int(rate)
    if rng.random() < rate - count:
        count += 1
    blocks: list[dict[str, int]] = []
    n_ch, n_t = int(data.shape[0]), int(data.shape[1])
    for _ in range(count):
        width_ch = int(rng.integers(int(cfg.missing_block_channel_min), int(cfg.missing_block_channel_max) + 1))
        width_ch = int(np.clip(width_ch, 1, n_ch))
        start_ch = int(rng.integers(0, max(1, n_ch - width_ch + 1)))
        duration_s = float(rng.uniform(float(cfg.missing_block_duration_min_s), float(cfg.missing_block_duration_max_s)))
        width_t = int(np.clip(round(duration_s / float(cfg.window_seconds) * float(n_t)), 1, n_t))
        start_t = int(rng.integers(0, max(1, n_t - width_t + 1)))
        end_ch = start_ch + width_ch
        end_t = start_t + width_t
        data[start_ch:end_ch, start_t:end_t] = 0.0
        blocks.append({"start_ch": start_ch, "end_ch": end_ch, "start_t": start_t, "end_t": end_t})
        for track in tracks:
            for ch in range(start_ch, end_ch):
                if not bool(track.visible[ch]):
                    continue
                t_idx = int(round(float(track.times_s[ch]) / float(cfg.window_seconds) * float(max(1, n_t - 1))))
                if start_t <= t_idx < end_t:
                    track.visible[ch] = False
    return blocks


def _seconds_to_bins(seconds: float, cfg: SynthConfig) -> int:
    return int(max(1, round(float(seconds) / float(cfg.window_seconds) * float(cfg.time_bins))))

