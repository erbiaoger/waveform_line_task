# Generated Datasets

Default output location for datasets made by `waveform_line_task`.

Before running generation, install dependencies:

```sh
sh run_install_deps.sh
```

Run from the project root:

```sh
sh run_generate.sh
```

The default generated folder is `datasets/v1_train/`.

Current default preset is larger than before:

- `6000` samples
- `64` channels
- `3072` time bins
- `auto` device, preferring CUDA

Recommended larger preset for model training:

```sh
uv run python generate_dataset.py \
  --out-dir datasets/v2_train_large \
  --num-samples 12000 \
  --image-size 1024 \
  --workers 1 \
  --device cuda \
  --overwrite
```
