#!/usr/bin/env bash
# Run sdata training (MultimodalFusionWithAttention + ActionClassifier)
# Usage: ./run_sdata_training.sh [optional: --dataset path/to/sdata]

cd "$(dirname "$0")"
python -m mmfuse.training.train_sdata_attention "$@"
