#!/bin/bash
# Precompute CLIP embeddings for established datasets and SData.
# Required for paper comparison (run_paper_comparisons.py).
#
# Usage: ./experiments/run_precompute_clip.sh

set -e
cd "$(dirname "$0")/.."

echo "Precomputing CLIP embeddings for paper comparison..."
echo ""

# Cross-dataset: video_mme, nextqa, charades
for ds in video_mme nextqa charades; do
  DATA="extdataset/$ds"
  OUT="embeddings/${ds}_clip"
  if [ ! -d "$DATA" ]; then
    echo "Skip $ds: no $DATA"
    continue
  fi
  if [ -f "$OUT/config.json" ] && [ -n "$(find "$OUT" -maxdepth 1 -name '*.pt' -print -quit 2>/dev/null)" ]; then
    echo "Skip $ds: embeddings exist in $OUT"
    continue
  fi
  echo "Precomputing CLIP for $ds..."
  case $ds in
    video_mme) python experiments/precompute_video_mme.py --data-dir "$DATA" --out-dir "$OUT" --vision-encoder clip --text-encoder clip ;;
    nextqa)    python experiments/precompute_nextqa.py --data-dir "$DATA" --out-dir "$OUT" --vision-encoder clip --text-encoder clip ;;
    charades)  python experiments/precompute_charades.py --data-dir "$DATA" --out-dir "$OUT" --vision-encoder clip --text-encoder clip ;;
  esac
  echo "  Done: $OUT"
done

# SData with CLIP vision
if [ -d "dataset/sdata" ] && [ ! -f "embeddings/sdata_clip/config.json" ]; then
  echo "Precomputing CLIP for SData..."
  python scripts/precompute_sdata_embeddings.py --dataset dataset/sdata --out-dir embeddings/sdata_clip \
    --vision-encoder clip --audio-encoder wav2vec --cross-pair
  echo "  Done: embeddings/sdata_clip"
else
  echo "Skip SData CLIP: dataset/sdata missing or embeddings/sdata_clip exists"
fi

echo ""
echo "Done. Run: python experiments/run_paper_comparisons.py --checkpoint checkpoints/ckpt_sdata_epoch_N.pt"
