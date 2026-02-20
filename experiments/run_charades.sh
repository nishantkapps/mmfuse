#!/bin/bash
# Run MMFuse on Charades (ADL-X) dataset.
# Requires: extdataset/charades/ with data, embeddings in embeddings/charades/
# Usage: ./experiments/run_charades.sh <checkpoint.pt>

set -e
cd "$(dirname "$0")/.."
CHECKPOINT="${1:?Usage: $0 <checkpoint.pt>}"
python experiments/run_dataset.py --dataset charades --checkpoint "$CHECKPOINT"
