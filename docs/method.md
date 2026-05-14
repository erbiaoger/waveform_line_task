# Waveform-Line Dataset Method

## Goal

The dataset represents a direct image-to-image task:

```text
raw waveform drawing -> vehicle-line drawing
```

Each training input is a rendered DAS-like waveform image. The matching label is
a binary image containing only the center lines of visible vehicle trajectories.

## Synthetic Waveform Model

The synthetic signal is a channel-time matrix:

```text
X[c, t]
```

where `c` is DAS channel index and `t` is time. A vehicle crossing channel `c`
at time `tau_c` contributes a Gaussian pulse:

```text
A exp(-(t - tau_c)^2 / (2 sigma^2))
```

Multiple vehicles and noise sources are added linearly. The generator includes:

- white noise;
- time-correlated colored noise;
- slow baseline drift;
- channel bias and channel gain changes;
- isolated Gaussian noise peaks;
- random dead channels;
- random missing channel-time blocks.

These terms make the waveform drawing closer to noisy DAS records while keeping
the exact vehicle-line label known.

## Vehicle-Line Geometry

Channels are treated as uniformly spaced positions with spacing `dx_m`. A
vehicle speed `v` gives the time difference between adjacent channels:

```text
Delta t = dx_m / (v / 3.6)
```

Forward vehicles have increasing crossing time as channel index increases.
Reverse vehicles have decreasing crossing time. Small random speed changes make
lines slightly curved. A low-probability stop-go event adds an extra delay after
a selected channel segment.

The generator also creates harder samples:

- crossing vehicles with opposite directions;
- near-parallel vehicles;
- overtaking vehicles with similar direction but different speeds.

## Pixel Mapping

The waveform image and label image use the same mapping:

- x-axis: channel position;
- y-axis: time, with `0 s` at the top and the window end at the bottom.

The input image draws each channel as a vertical waveform trace:

```text
x_pixel = channel_center_x + normalized_amplitude * wiggle_pixels
```

The label image ignores waveform amplitude and draws the vehicle center line at:

```text
x_pixel = channel_center_x
y_pixel = tau_c / window_seconds * (image_size - 1)
```

This keeps the input and target aligned while preserving the visual style of a
raw waveform plot.

## Binary Labels

The label image is a binary mask:

- background: `0`;
- vehicle line: `255`.

If dead channels or missing blocks hide a vehicle point, that point is removed
from the visible label. Label segments are connected only across consecutive
visible channels, so missing regions create gaps instead of false lines.

