#!/bin/bash
# Run precompute + evaluation + paper comparison for VideoMME and NeXTQA only.
# Use when Charades is not yet ready.
#
# Usage: ./experiments/run_experiments_video_mme_nextqa.sh

set -e
cd "$(dirname "$0")/.."
PROJ_ROOT="$(pwd)"

MODEL_FILE="${MODEL_FILE:-checkpoints/model.pt}"

echo "=========================================="
echo "VideoMME + NeXTQA Experiments"
echo "=========================================="
echo "Model file: $MODEL_FILE"
echo "=========================================="

# --- VideoMME ---
VIDEOMME_DIR="extdataset/video_mme"
EMB_VIDEOMME="embeddings/video_mme"

if [ -d "$VIDEOMME_DIR" ] && [ -f "$VIDEOMME_DIR/annotations.json" ]; then
  echo ""
  echo "[1/4] VideoMME: precompute VisCoP + evaluate"
  if [ ! -f "$EMB_VIDEOMME/config.json" ] || [ -z "$(find "$EMB_VIDEOMME" -maxdepth 1 -name "*.pt" -print -quit 2>/dev/null)" ]; then
    python experiments/precompute_video_mme.py --data-dir "$VIDEOMME_DIR" --out-dir "$EMB_VIDEOMME"
  else
    echo "  Embeddings exist, skipping"
  fi
  python experiments/run_dataset.py --dataset video_mme --checkpoint "$MODEL_FILE" --embeddings-dir "$EMB_VIDEOMME" --linear-probe
  echo "  VideoMME done."
else
  echo "[1/4] VideoMME: SKIP (no data)"
fi

# --- NeXTQA ---
NEXTQA_DIR="extdataset/nextqa"
EMB_NEXTQA="embeddings/nextqa"

if [ -f "$NEXTQA_DIR/annotations.json" ]; then
  echo ""
  echo "[2/4] NeXTQA: precompute VisCoP + evaluate"
  if [ ! -f "$EMB_NEXTQA/config.json" ] || [ -z "$(find "$EMB_NEXTQA" -maxdepth 1 -name "*.pt" -print -quit 2>/dev/null)" ]; then
    python experiments/precompute_nextqa.py --data-dir "$NEXTQA_DIR" --out-dir "$EMB_NEXTQA"
  else
    echo "  Embeddings exist, skipping"
  fi
  python experiments/run_dataset.py --dataset nextqa --checkpoint "$MODEL_FILE" --embeddings-dir "$EMB_NEXTQA" --linear-probe
  echo "  NeXTQA done."
else
  echo "[2/4] NeXTQA: SKIP (no data)"
fi

# --- Precompute CLIP for paper comparison ---
echo ""
echo "[3/4] Precomputing CLIP embeddings..."
for ds in video_mme nextqa; do
  DATA="extdataset/$ds"
  OUT="embeddings/${ds}_clip"
  if [ ! -d "$DATA" ] || [ ! -f "$DATA/annotations.json" ]; then
    echo "  Skip $ds: no data"
    continue
  fi
  if [ -f "$OUT/config.json" ] && [ -n "$(find "$OUT" -maxdepth 1 -name '*.pt' -print -quit 2>/dev/null)" ]; then
    echo "  Skip $ds: CLIP embeddings exist"
    continue
  fi
  echo "  Precomputing CLIP for $ds..."
  case $ds in
    video_mme) python experiments/precompute_video_mme.py --data-dir "$DATA" --out-dir "$OUT" --vision-encoder clip --text-encoder clip ;;
    nextqa)    python experiments/precompute_nextqa.py --data-dir "$DATA" --out-dir "$OUT" --vision-encoder clip --text-encoder clip ;;
  esac
done

# --- Paper comparison ---
echo ""
echo "[4/4] Running paper comparison..."
python experiments/run_paper_comparisons.py --checkpoint "$MODEL_FILE"

echo ""
echo "=========================================="
echo "Done. Output: experiments/results/paper_comparison.csv, paper_comparison.tex"
echo "=========================================="
