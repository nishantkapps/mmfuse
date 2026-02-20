#!/bin/bash
# Run MMFuse on SData (primary benchmark).
# Requires: embeddings/sdata_viscop/ from scripts/precompute_sdata_embeddings.py
# Usage: ./experiments/run_sdata.sh <checkpoint.pt>

set -e
cd "$(dirname "$0")/.."
CHECKPOINT="${1:?Usage: $0 <checkpoint.pt>}"
python experiments/run_dataset.py --dataset sdata --checkpoint "$CHECKPOINT"
