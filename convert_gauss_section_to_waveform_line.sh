#!/usr/bin/env sh
set -eu

# Convert the real `gauss_section.npy` array into waveform-line model input PNGs.
#
# Default usage:
#   sh convert_gauss_section_to_waveform_line.sh
#
# Example with overrides:
#   OUT_DIR=datasets/real_gauss_section_preview \
#   NUM_WINDOWS=0 \
#   START_WINDOW_INDEX=10 \
#   WINDOW_SECONDS=120 \
#   STRIDE_SECONDS=60 \
#   sh convert_gauss_section_to_waveform_line.sh
#
# Output:
#   <out-dir>/images/*.png
#   <out-dir>/previews/*_overlay.png
#   <out-dir>/manifest.csv
#   <out-dir>/meta.json

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$SCRIPT_DIR
cd "$PROJECT_ROOT"

INPUT=${INPUT:-datasets/real/gauss_section.npy}
OUT_DIR=${OUT_DIR:-datasets/real_gauss_section_preview}
ARRAY_LAYOUT=${ARRAY_LAYOUT:-time_channel}
FS=${FS:-1000}
WINDOW_SECONDS=${WINDOW_SECONDS:-120}
STRIDE_SECONDS=${STRIDE_SECONDS:-60}
START_WINDOW_INDEX=${START_WINDOW_INDEX:-0}
NUM_WINDOWS=${NUM_WINDOWS:-0}
CHANNEL_START=${CHANNEL_START:-0}
CHANNEL_COUNT=${CHANNEL_COUNT:-0}
CENTER_MODE=${CENTER_MODE:-channel_median}
IMAGE_SIZE=${IMAGE_SIZE:-512}
WAVEFORM_LINE_WIDTH=${WAVEFORM_LINE_WIDTH:-1}
WIGGLE_FRACTION=${WIGGLE_FRACTION:-0.28}
ROBUST_PERCENTILE=${ROBUST_PERCENTILE:-99.5}
PREVIEW_COUNT=${PREVIEW_COUNT:-4}
IMAGE_PREFIX=${IMAGE_PREFIX:-sample}
OVERWRITE=${OVERWRITE:-1}

overwrite_args=""
if [ "$OVERWRITE" = "1" ] || [ "$OVERWRITE" = "true" ]; then
  overwrite_args="--overwrite"
fi

uv run python convert_real_npy_to_dataset.py \
  --input "$INPUT" \
  --out-dir "$OUT_DIR" \
  --array-layout "$ARRAY_LAYOUT" \
  --fs "$FS" \
  --window-seconds "$WINDOW_SECONDS" \
  --stride-seconds "$STRIDE_SECONDS" \
  --start-window-index "$START_WINDOW_INDEX" \
  --num-windows "$NUM_WINDOWS" \
  --channel-start "$CHANNEL_START" \
  --channel-count "$CHANNEL_COUNT" \
  --center-mode "$CENTER_MODE" \
  --image-size "$IMAGE_SIZE" \
  --waveform-line-width "$WAVEFORM_LINE_WIDTH" \
  --wiggle-fraction "$WIGGLE_FRACTION" \
  --robust-percentile "$ROBUST_PERCENTILE" \
  --preview-count "$PREVIEW_COUNT" \
  --image-prefix "$IMAGE_PREFIX" \
  $overwrite_args
