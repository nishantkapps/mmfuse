#!/bin/bash
# Run all ablation experiments. Edit EMBEDDINGS_DIR and EPOCHS as needed.
# Usage: bash scripts/run_all_ablations.sh

set -e
EMBEDDINGS_DIR="${EMBEDDINGS_DIR:-embeddings/sdata_viscop}"
DATASET="${DATASET:-dataset/sdata}"
EPOCHS="${EPOCHS:-10}"

if [ ! -d "$EMBEDDINGS_DIR" ]; then
  echo "Embeddings dir not found: $EMBEDDINGS_DIR"
  echo "Set EMBEDDINGS_DIR or run precompute_sdata_embeddings.py first."
  exit 1
fi

ABLATIONS="full no_vision_cam1 no_vision_cam2 no_audio vision_only vision_cam1_only vision_cam2_only audio_only fusion_concat no_kl_loss no_movement_head"

for ablation in $ABLATIONS; do
  echo "=========================================="
  echo "Running ablation: $ablation"
  echo "=========================================="
  python scripts/run_ablation.py --ablation "$ablation" --embeddings-dir "$EMBEDDINGS_DIR" --dataset "$DATASET" --epochs "$EPOCHS"
done

echo ""
echo "All ablations complete. Collect results with:"
echo "  python scripts/collect_ablation_results.py --ablation-dir ablation_runs --output-dir results"
