#!/bin/bash
# Full pipeline for paper-ready comparison: MMFuse vs CLIP vs VisCoP.
#
# 1. Precompute CLIP embeddings (video_mme, nextqa, charades, sdata)
# 2. Run paper comparison script
#
# Prerequisites:
#   - VisCoP embeddings: embeddings/video_mme, embeddings/nextqa, embeddings/charades, embeddings/sdata_viscop
#   - MMFuse results: experiments/results/{video_mme,nextqa,charades}/results.json (from run_all_cross_dataset_experiments.sh)
#   - MMFuse checkpoint for SData evaluation (optional)
#
# Usage:
#   ./experiments/run_all_paper_comparisons.sh
#   CHECKPOINT=checkpoints/ckpt_sdata_epoch_5.pt ./experiments/run_all_paper_comparisons.sh

set -e
cd "$(dirname "$0")/.."
PROJ_ROOT="$(pwd)"

# Find latest checkpoint
find_checkpoint() {
  latest=$(ls checkpoints/ckpt_sdata_epoch_*.pt 2>/dev/null | sort -V | tail -1)
  [ -n "$latest" ] && echo "$latest" && return
  latest=$(find runs -name "ckpt_sdata_epoch_*.pt" 2>/dev/null | sort -V | tail -1)
  [ -n "$latest" ] && echo "$latest" && return
  [ -f models/sdata_viscop/pytorch_model.bin ] && echo models/sdata_viscop/pytorch_model.bin && return
  echo ""
}

CHECKPOINT="${CHECKPOINT:-$(find_checkpoint)}"

echo "=========================================="
echo "Paper Comparison Pipeline"
echo "=========================================="
echo "Checkpoint: ${CHECKPOINT:-none}"
echo "=========================================="

# 1. Precompute CLIP embeddings
echo ""
echo "[1/2] Precomputing CLIP embeddings..."
./experiments/run_precompute_clip.sh

# 2. Run paper comparison
echo ""
echo "[2/2] Running paper comparison..."
if [ -n "$CHECKPOINT" ] && [ -f "$CHECKPOINT" ]; then
  python experiments/run_paper_comparisons.py --checkpoint "$CHECKPOINT"
else
  python experiments/run_paper_comparisons.py
fi

echo ""
echo "=========================================="
echo "Done. Output:"
echo "  - experiments/results/paper_comparison.csv"
echo "  - experiments/results/paper_comparison.tex"
echo "=========================================="
