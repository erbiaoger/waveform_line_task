# Waveform-Line Task

This folder contains an independent dataset maker for a new recognition task:

```text
raw DAS-like waveform image -> vehicle-line-only binary label image
```

No code in this folder imports `autotrack` or any existing project scripts.

## Folder Layout

- `generate_dataset.py`: CLI entry point. It generates paired PNG images, labels, `manifest.csv`, and `meta.json`.
- `synth.py`: independent DAS-style waveform and vehicle trajectory synthesizer.
- `render.py`: shared pixel mapping and PNG render utilities for waveform images, label masks, and previews.
- `run_install_deps.sh`: `uv` environment creation and dependency installation shortcut.
- `run_generate.sh`: shell shortcut that runs the CLI through `uv run`.
- `model/`: independent waveform-line segmentation package: dataset, U-Net, losses, metrics, and skeleton postprocess.
- `train_model.py`: model training entry point.
- `predict_model.py`: batch inference entry point.
- `run_train.sh`: training shortcut through `uv run`.
- `run_predict.sh`: prediction shortcut through `uv run`.
- `docs/`: method notes for the physical and mathematical assumptions.
- `datasets/`: default generated dataset location. Generated data can be recreated.
- `models/`: default checkpoint output location.
- `predictions/`: default batch prediction output location.

## Install Dependencies

Create the project-local `.venv` and install dependencies with `uv`:

```sh
sh run_install_deps.sh
```

Direct `uv` commands:

```sh
uv venv .venv --python 3.12
uv sync
```

Check the runtime:

```sh
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.backends.mps.is_available())"
```

## Device Policy

- Default device is now `auto`, which prefers `cuda`, then `mps`, then `cpu`.
- Training and inference are tuned first for NVIDIA CUDA.
- CPU remains supported by explicitly passing `--device cpu`.

## Generate Data

From the project root:

```sh
uv run python generate_dataset.py \
  --out-dir datasets/v1_train \
  --num-samples 6000 \
  --image-size 1024 \
  --workers 8 \
  --device cuda \
  --overwrite
```

Or use the shortcut:

```sh
sh run_generate.sh
```

Useful environment overrides:

```sh
NUM_SAMPLES=128 WORKERS=4 DEVICE=cpu OUT_DIR=/tmp/waveform_line_test sh run_generate.sh
```

Recommended larger training set:

```sh
uv run python generate_dataset.py \
  --out-dir datasets/v2_train_large \
  --num-samples 12000 \
  --image-size 1024 \
  --workers 1 \
  --device cuda \
  --overwrite
```

## Train Model

```sh
uv run python train_model.py \
  --data-dir datasets/v1_train \
  --out-dir models/unet_v1 \
  --image-size 512 \
  --batch-size 12 \
  --epochs 60 \
  --device cuda \
  --amp
```

Shortcut:

```sh
sh run_train.sh
```

CPU fallback:

```sh
DEVICE=cpu AMP=0 BATCH_SIZE=4 NUM_WORKERS=0 sh run_train.sh
```

Training outputs:

```text
<out-dir>/
  checkpoint_last.pt
  checkpoint_best.pt
  train_config.json
  train_history.jsonl
```

## Predict

```sh
uv run python predict_model.py \
  --input-dir datasets/sample_check/images \
  --model models/unet_v1/checkpoint_best.pt \
  --out-dir predictions/sample_check \
  --image-size 512 \
  --device cuda \
  --amp
```

Shortcut:

```sh
sh run_predict.sh
```

Prediction outputs:

```text
<out-dir>/
  pred_masks/
  pred_skeletons/
  previews/
  summary.json
```

## Output

```text
<out-dir>/
  README.md
  meta.json
  manifest.csv
  images/sample_000000.png
  labels/sample_000000.png
  previews/sample_000000_overlay.png
```

- `images/`: white-background waveform traces. There are no axes, titles, or labels.
- `labels/`: black-background binary masks. Vehicle lines are white (`255`), background is black (`0`).
- `previews/`: optional red overlays for manual inspection. Preview plotting uses Times New Roman.
- `manifest.csv`: image/label path pairs plus vehicle counts and random seeds.
- `meta.json`: dataset format and full generation configuration.

## Checks

```sh
uv run python -m compileall .
uv run python generate_dataset.py \
  --out-dir /tmp/waveform_line_smoke \
  --num-samples 8 \
  --workers 1 \
  --device cpu \
  --overwrite
uv run python train_model.py \
  --data-dir datasets/sample_check \
  --out-dir /tmp/waveform_line_model_smoke \
  --epochs 1 \
  --batch-size 2 \
  --device cpu \
  --overwrite
uv run python predict_model.py \
  --input-dir datasets/sample_check/images \
  --model /tmp/waveform_line_model_smoke/checkpoint_last.pt \
  --out-dir /tmp/waveform_line_pred_smoke \
  --image-size 512 \
  --device cpu
```

The label PNGs should contain only `0` and `255`, and all image/label pairs
should have the same `1024x1024` shape unless `--image-size` is changed.
