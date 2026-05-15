#!/usr/bin/env sh
# Purpose:
#   Shortcut for generating a waveform-line dataset with the project's
#   Python environment through `uv run`.
#
# Parameters (override through environment variables before `sh`):
#   OUT_DIR       Output dataset directory.
#                 Default: datasets/v1_train
#   NUM_SAMPLES   Number of image/label pairs to generate.
#                 Default: 6000
#   IMAGE_SIZE    Output PNG size in pixels.
#                 Default: 512
#   WORKERS       CPU worker count. CUDA generation is internally forced to 1.
#                 Default: 8
#   DEVICE        Device selection: auto, cuda, mps, cpu.
#                 Default: auto
#   SEED          Base random seed.
#                 Default: 42
#   PREVIEW_COUNT Number of overlay preview PNG files to save.
#                 Default: 24
#   OVERWRITE     Replace the output directory when set to 1/true.
#                 Default: 1
#
# Examples:
#   sh run_generate.sh
#   NUM_SAMPLES=12000 DEVICE=cuda WORKERS=1 sh run_generate.sh
#   OUT_DIR=/tmp/waveform_line_test DEVICE=cpu sh run_generate.sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$SCRIPT_DIR
cd "$PROJECT_DIR"

OUT_DIR=${OUT_DIR:-datasets/v1_train}
NUM_SAMPLES=${NUM_SAMPLES:-6000}
IMAGE_SIZE=${IMAGE_SIZE:-512}
WORKERS=${WORKERS:-64}
DEVICE=${DEVICE:-cpu}
SEED=${SEED:-42}
PREVIEW_COUNT=${PREVIEW_COUNT:-24}
OVERWRITE=${OVERWRITE:-1}

overwrite_args=""
if [ "$OVERWRITE" = "1" ] || [ "$OVERWRITE" = "true" ]; then
  overwrite_args="--overwrite"
fi

uv run python generate_dataset.py \
  --out-dir "$OUT_DIR" \
  --num-samples "$NUM_SAMPLES" \
  --image-size "$IMAGE_SIZE" \
  --workers "$WORKERS" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --preview-count "$PREVIEW_COUNT" \
  $overwrite_args
