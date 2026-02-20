#!/bin/bash
# Run MMFuse on NeXTQA dataset.
# Requires: extdataset/nextqa/ with data, embeddings in embeddings/nextqa/
# Usage: ./experiments/run_nextqa.sh <checkpoint.pt>

set -e
cd "$(dirname "$0")/.."
CHECKPOINT="${1:?Usage: $0 <checkpoint.pt>}"
python experiments/run_dataset.py --dataset nextqa --checkpoint "$CHECKPOINT"
