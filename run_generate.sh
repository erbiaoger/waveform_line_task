#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_DIR"

OUT_DIR=${OUT_DIR:-waveform_line_task/datasets/v1_train}
NUM_SAMPLES=${NUM_SAMPLES:-1000}
IMAGE_SIZE=${IMAGE_SIZE:-1024}
WORKERS=${WORKERS:-8}
DEVICE=${DEVICE:-cpu}
SEED=${SEED:-42}
PREVIEW_COUNT=${PREVIEW_COUNT:-16}
OVERWRITE=${OVERWRITE:-1}

overwrite_args=""
if [ "$OVERWRITE" = "1" ] || [ "$OVERWRITE" = "true" ]; then
  overwrite_args="--overwrite"
fi

uv run python waveform_line_task/generate_dataset.py \
  --out-dir "$OUT_DIR" \
  --num-samples "$NUM_SAMPLES" \
  --image-size "$IMAGE_SIZE" \
  --workers "$WORKERS" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --preview-count "$PREVIEW_COUNT" \
  $overwrite_args

