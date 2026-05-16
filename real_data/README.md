# Real Data Conversion

This folder contains utilities for converting real unlabeled DAS `.npy` arrays
into the waveform PNG style consumed by the current model in this project.

## Files

- `convert_real_npy_to_dataset.py`: generic CLI that slices a real array into
  windows and renders `images/*.png`, placeholder `labels/*.png`,
  `previews/*.png`, `manifest.csv`, and `meta.json`.
- `convert_gauss_section_to_waveform_line.sh`: preconfigured shell wrapper for
  `/Volumes/SanDisk2T4/MyProjects/BaFang/xi/saved_arrays04/gauss_section.npy`.

## Usage

From the project root:

```sh
sh real_data/convert_gauss_section_to_waveform_line.sh
```

Override the number of exported windows:

```sh
NUM_WINDOWS=6 START_WINDOW_INDEX=8 sh real_data/convert_gauss_section_to_waveform_line.sh
```

Run the generic CLI directly:

```sh
source .venv/bin/activate
uv run python real_data/convert_real_npy_to_dataset.py \
  --input /path/to/real.npy \
  --out-dir datasets/real_preview \
  --array-layout time_channel \
  --fs 1000 \
  --window-seconds 120 \
  --stride-seconds 60 \
  --num-windows 4 \
  --overwrite
```

## Notes

- Default centering uses `channel_median`, which is usually safer for real data
  than rendering raw positive-only amplitudes directly.
- The generated `manifest.csv` is compatible with `predict_model.py`.
- Placeholder labels are intentionally all black because the source data is
  unlabeled.
