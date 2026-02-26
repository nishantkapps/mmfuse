#!/bin/bash
# Full pipeline for paper-ready comparison: MMFuse vs CLIP vs VisCoP.
#
# 1. Precompute CLIP embeddings (video_mme, nextqa, charades, sdata)
# 2. Run paper comparison script
#
# Prerequisites:
#   - VisCoP embeddings: embeddings/video_mme, embeddings/nextqa, embeddings/charades, embeddings/sdata_viscop
#   - MMFuse results: experiments/results/{video_mme,nextqa,charades}/results.json (from run_all_cross_dataset_experiments.sh)
#   - checkpoints/model.pt for MMFuse SData evaluation
#
# Usage:
#   ./experiments/run_all_paper_comparisons.sh
#   MODEL_FILE=path/to/model.pt ./experiments/run_all_paper_comparisons.sh

set -e
cd "$(dirname "$0")/.."
PROJ_ROOT="$(pwd)"

MODEL_FILE="${MODEL_FILE:-checkpoints/model.pt}"

echo "=========================================="
echo "Paper Comparison Pipeline"
echo "=========================================="
echo "Model file: $MODEL_FILE"
echo "=========================================="

# 1. Precompute CLIP embeddings
echo ""
echo "[1/2] Precomputing CLIP embeddings..."
./experiments/run_precompute_clip.sh

# 2. Run paper comparison
echo ""
echo "[2/2] Running paper comparison..."
python experiments/run_paper_comparisons.py --checkpoint "$MODEL_FILE"

echo ""
echo "=========================================="
echo "Done. Output:"
echo "  - experiments/results/paper_comparison.csv"
echo "  - experiments/results/paper_comparison.tex"
echo "=========================================="
