#!/bin/bash
# Download annotations + videos for VideoMME, NeXTQA, Charades.
# Defaults: ~2000 VideoMME, ~3000 NeXTQA, ~1000 Charades (enough for good linear probe).
#
# Usage:
#   ./experiments/download_datasets.sh              # all datasets
#   ./experiments/download_datasets.sh video_mme   # VideoMME only
#   ./experiments/download_datasets.sh nextqa      # NeXTQA only
#   ./experiments/download_datasets.sh charades    # Charades only
#
# Optional: MAX_SAMPLES=5000 VIDEO_ZIPS=10 ./experiments/download_datasets.sh

set -e
cd "$(dirname "$0")/.."

MAX_SAMPLES="${MAX_SAMPLES:-}"
VIDEO_ZIPS="${VIDEO_ZIPS:-}"
DS="${1:-all}"

ARGS=""
[ -n "$MAX_SAMPLES" ] && ARGS="$ARGS --max-samples $MAX_SAMPLES"
[ -n "$VIDEO_ZIPS" ] && ARGS="$ARGS --video-zips $VIDEO_ZIPS"

echo "Downloading datasets (DS=$DS)..."
python experiments/download_datasets_full.py "$DS" $ARGS
