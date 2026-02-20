#!/bin/bash
# Run MMFuse on VIMA-Bench (simulator-based).
# Requires: extdataset/vima_bench/ with precomputed embeddings or custom eval path.
# Usage: ./experiments/run_vima_bench.sh <checkpoint.pt>

set -e
cd "$(dirname "$0")/.."
CHECKPOINT="${1:?Usage: $0 <checkpoint.pt>}"
python experiments/run_dataset.py --dataset vima_bench --checkpoint "$CHECKPOINT"
