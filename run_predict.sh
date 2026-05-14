#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_DIR"

INPUT_DIR=${INPUT_DIR:-waveform_line_task/datasets/sample_check/images}
MODEL=${MODEL:-waveform_line_task/models/unet_v1/checkpoint_best.pt}
OUT_DIR=${OUT_DIR:-waveform_line_task/predictions/sample_check}
IMAGE_SIZE=${IMAGE_SIZE:-512}
BATCH_SIZE=${BATCH_SIZE:-8}
DEVICE=${DEVICE:-auto}
THRESHOLD=${THRESHOLD:-0.5}
NUM_WORKERS=${NUM_WORKERS:-2}
AMP=${AMP:-1}

amp_args=""
if [ "$AMP" = "1" ] || [ "$AMP" = "true" ]; then
  amp_args="--amp"
fi

uv run python waveform_line_task/predict_model.py \
  --input-dir "$INPUT_DIR" \
  --model "$MODEL" \
  --out-dir "$OUT_DIR" \
  --image-size "$IMAGE_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --threshold "$THRESHOLD" \
  --num-workers "$NUM_WORKERS" \
  $amp_args
