#!/usr/bin/env bash
# Fine-tune MMFuse on real-world datasets.
# Usage: ./run_finetune.sh --dataset nextqa --max-samples 500 --epochs 5
#        ./run_finetune.sh --dataset vima_bench --max-samples 500 --epochs 5
#        ./run_finetune.sh --dataset vima_bench --checkpoint checkpoints/finetune_nextqa/checkpoint.pt --epochs 3
# Requires: source mmfuse-env/bin/activate (or projects/mmfuse-env)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$(dirname "$SCRIPT_DIR"):${PYTHONPATH}"
exec python -m mmfuse.training.finetune "$@"
