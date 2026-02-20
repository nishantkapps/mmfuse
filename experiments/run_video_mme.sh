#!/bin/bash
# Run MMFuse on VideoMME dataset.
# Requires: extdataset/video_mme/ with data, embeddings in embeddings/video_mme/
# Usage: ./experiments/run_video_mme.sh <checkpoint.pt>

set -e
cd "$(dirname "$0")/.."
CHECKPOINT="${1:?Usage: $0 <checkpoint.pt>}"
python experiments/run_dataset.py --dataset video_mme --checkpoint "$CHECKPOINT"
