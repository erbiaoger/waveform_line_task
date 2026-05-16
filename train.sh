#!/usr/bin/env sh
# Purpose:
#   Shortcut for training the waveform-line segmentation model with the
#   project's Python environment through `uv run`.
#
# Parameters (override through environment variables before `sh`):
#   DATA_DIR      Dataset directory containing manifest.csv, images/, labels/.
#                 Default: datasets/v1_train
#   OUT_DIR       Output directory for checkpoints and logs.
#                 Default: models/unet_v1
#   IMAGE_SIZE    Training resize resolution.
#                 Default: 512
#   BATCH_SIZE    Training batch size.
#                 Default: 16
#   EPOCHS        Training epoch count.
#                 Default: 600
#   DEVICE        Device selection: auto, cuda, mps, cpu.
#                 Default: auto
#   VAL_FRACTION  Validation split fraction.
#                 Default: 0.1
#   NUM_WORKERS   PyTorch DataLoader worker count.
#                 Default: 16
#   VAL_SKELETON_EVERY
#                 Run validation skeleton metrics every N epochs.
#                 Default: 10
#                 A validation preview image is also exported on those epochs.
#   AMP           Enable mixed precision when set to 1/true.
#                 Default: 1
#   OVERWRITE     Replace OUT_DIR when set to 1/true.
#                 Default: 0
#
# Examples:
#   sh run_train.sh
#   DEVICE=cuda AMP=1 BATCH_SIZE=16 sh run_train.sh
#   DEVICE=cpu AMP=0 BATCH_SIZE=4 NUM_WORKERS=0 sh run_train.sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$SCRIPT_DIR
cd "$PROJECT_DIR"

DATA_DIR=${DATA_DIR:-datasets/v1_train}
OUT_DIR=${OUT_DIR:-models/unet_v1}
IMAGE_SIZE=${IMAGE_SIZE:-512}
BATCH_SIZE=${BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-600}
DEVICE=${DEVICE:-auto}
VAL_FRACTION=${VAL_FRACTION:-0.1}
NUM_WORKERS=${NUM_WORKERS:-16}
VAL_SKELETON_EVERY=${VAL_SKELETON_EVERY:-10}
AMP=${AMP:-1}
OVERWRITE=${OVERWRITE:-1}

overwrite_args=""
if [ "$OVERWRITE" = "1" ] || [ "$OVERWRITE" = "true" ]; then
  overwrite_args="--overwrite"
fi

amp_args=""
if [ "$AMP" = "1" ] || [ "$AMP" = "true" ]; then
  amp_args="--amp"
fi

uv run python train_model.py \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --image-size "$IMAGE_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --device "$DEVICE" \
  --val-fraction "$VAL_FRACTION" \
  --num-workers "$NUM_WORKERS" \
  --val-skeleton-every "$VAL_SKELETON_EVERY" \
  $amp_args \
  $overwrite_args
