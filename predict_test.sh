#!/usr/bin/env sh
# Purpose:
#   Shortcut for batch prediction with a trained waveform-line model through
#   the project's Python environment and `uv run`.
#
# Parameters (override through environment variables before `sh`):
#   INPUT_DIR     Input image directory to predict.
#                 Default: datasets/sample_check/images
#   MODEL         Trained checkpoint path.
#                 Default: models/unet_v1/checkpoint_best.pt
#   OUT_DIR       Output directory for masks, skeletons, previews, summary.
#                 Default: predictions/sample_check
#   IMAGE_SIZE    Inference resize resolution.
#                 Default: 512
#   BATCH_SIZE    Inference batch size.
#                 Default: 32
#   DEVICE        Device selection: auto, cuda, mps, cpu.
#                 Default: auto
#   THRESHOLD     Sigmoid threshold for binary masks.
#                 Default: 0.5
#   NUM_WORKERS   PyTorch DataLoader worker count.
#                 Default: 2
#   AMP           Enable mixed precision when set to 1/true.
#                 Default: 1
#
# Examples:
#   sh run_predict.sh
#   DEVICE=cuda AMP=1 MODEL=/tmp/unet/checkpoint_best.pt sh run_predict.sh
#   INPUT_DIR=/tmp/test/images OUT_DIR=/tmp/test_pred DEVICE=cpu AMP=0 sh run_predict.sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$SCRIPT_DIR
cd "$PROJECT_DIR"

INPUT_DIR=${INPUT_DIR:-datasets/v1_test/images}
MODEL=${MODEL:-models/unet_v1/checkpoint_last.pt}
OUT_DIR=${OUT_DIR:-predictions/v1_test}
IMAGE_SIZE=${IMAGE_SIZE:-512}
BATCH_SIZE=${BATCH_SIZE:-32}
DEVICE=${DEVICE:-auto}
THRESHOLD=${THRESHOLD:-0.5}
NUM_WORKERS=${NUM_WORKERS:-2}
AMP=${AMP:-1}

amp_args=""
if [ "$AMP" = "1" ] || [ "$AMP" = "true" ]; then
  amp_args="--amp"
fi

uv run python predict_model.py \
  --input-dir "$INPUT_DIR" \
  --model "$MODEL" \
  --out-dir "$OUT_DIR" \
  --image-size "$IMAGE_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --threshold "$THRESHOLD" \
  --num-workers "$NUM_WORKERS" \
  $amp_args
