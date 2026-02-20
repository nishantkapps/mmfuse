#!/bin/bash
# Run MMFuse on all cross-dataset benchmarks and show consolidated results.
# Skips datasets without precomputed embeddings.
# Usage: ./experiments/run_all.sh <checkpoint.pt>

set -e
cd "$(dirname "$0")/.."
CHECKPOINT="${1:?Usage: $0 <checkpoint.pt>}"
python experiments/run_all_experiments.py --checkpoint "$CHECKPOINT"
