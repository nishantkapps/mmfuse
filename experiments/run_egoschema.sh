#!/bin/bash
# Run MMFuse on EgoSchema dataset.
# Requires: extdataset/egoschema/ with data, embeddings in embeddings/egoschema/
# Usage: ./experiments/run_egoschema.sh <checkpoint.pt>

set -e
cd "$(dirname "$0")/.."
CHECKPOINT="${1:?Usage: $0 <checkpoint.pt>}"
python experiments/run_dataset.py --dataset egoschema --checkpoint "$CHECKPOINT"
