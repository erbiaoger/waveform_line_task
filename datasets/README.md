# Generated Datasets

Default output location for datasets made by `waveform_line_task`.

Run from the project root:

```sh
sh waveform_line_task/run_generate.sh
```

The default generated folder is `waveform_line_task/datasets/v1_train/`.

Current default preset is larger than before:

- `6000` samples
- `64` channels
- `3072` time bins
- `auto` device, preferring CUDA

Recommended larger preset for model training:

```sh
uv run python waveform_line_task/generate_dataset.py \
  --out-dir waveform_line_task/datasets/v2_train_large \
  --num-samples 12000 \
  --image-size 1024 \
  --workers 1 \
  --device cuda \
  --overwrite
```
