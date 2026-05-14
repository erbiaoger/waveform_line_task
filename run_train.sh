#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_DIR"

DATA_DIR=${DATA_DIR:-waveform_line_task/datasets/v1_train}
OUT_DIR=${OUT_DIR:-waveform_line_task/models/unet_v1}
IMAGE_SIZE=${IMAGE_SIZE:-512}
BATCH_SIZE=${BATCH_SIZE:-8}
EPOCHS=${EPOCHS:-50}
DEVICE=${DEVICE:-cpu}
VAL_FRACTION=${VAL_FRACTION:-0.1}
NUM_WORKERS=${NUM_WORKERS:-0}
OVERWRITE=${OVERWRITE:-0}

overwrite_args=""
if [ "$OVERWRITE" = "1" ] || [ "$OVERWRITE" = "true" ]; then
  overwrite_args="--overwrite"
fi

uv run python waveform_line_task/train_model.py \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --image-size "$IMAGE_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --device "$DEVICE" \
  --val-fraction "$VAL_FRACTION" \
  --num-workers "$NUM_WORKERS" \
  $overwrite_args

