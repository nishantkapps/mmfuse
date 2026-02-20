#!/bin/bash
# Run full pipeline for all cross-dataset benchmarks: prepare → precompute → evaluate → collect table.
# Datasets: video_mme, nextqa, charades (uses extdataset/video_mme for VideoMME data)
#
# Usage:
#   ./experiments/run_all_cross_dataset_experiments.sh
#   CHECKPOINT=path/to/model.pt ./experiments/run_all_cross_dataset_experiments.sh
#
# Prerequisites:
#   - extdataset/video_mme/ with annotations.json (from download) or parquet + videos_chunked_*.zip (full)
#   - extdataset/nextqa/ with annotations.json (run: python experiments/download_dataset_subsets.py nextqa --max-samples 100)
#   - extdataset/charades/ with annotations.json (run: python experiments/download_dataset_subsets.py charades --max-samples 100)

set -e
cd "$(dirname "$0")/.."
PROJ_ROOT="$(pwd)"

# Find latest checkpoint from model training (excludes ablation_runs)
# Priority: checkpoints/ > runs/*/ > models/sdata_viscop/pytorch_model.bin
find_checkpoint() {
  # 1. Primary: checkpoints/ from training/train_sdata_attention.py
  latest=$(ls checkpoints/ckpt_sdata_epoch_*.pt 2>/dev/null | sort -V | tail -1)
  [ -n "$latest" ] && echo "$latest" && return
  # 2. runs/ subdirs (if training used custom out-dir)
  latest=$(find runs -name "ckpt_sdata_epoch_*.pt" 2>/dev/null | sort -V | tail -1)
  [ -n "$latest" ] && echo "$latest" && return
  # 3. Exported model
  [ -f models/sdata_viscop/pytorch_model.bin ] && echo models/sdata_viscop/pytorch_model.bin && return
  echo ""
}

CHECKPOINT="${CHECKPOINT:-$(find_checkpoint)}"
if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
  echo "No model training checkpoint found."
  echo "  Train first: python training/train_sdata_attention.py --use-precomputed --embeddings-dir embeddings/sdata_viscop"
  echo "  Checkpoints save to checkpoints/ckpt_sdata_epoch_*.pt"
  echo "  Or pass: CHECKPOINT=path/to/ckpt_sdata_epoch_N.pt ./experiments/run_all_cross_dataset_experiments.sh"
  exit 1
fi

echo "=========================================="
echo "MMFuse Cross-Dataset Experiments"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "=========================================="

# --- VideoMME (extdataset/video_mme: annotations from download or parquet+zip) ---
VIDEOMME_DIR="${VIDEOMME_DIR:-extdataset/video_mme}"
EMB_VIDEOMME="embeddings/video_mme"

if [ -d "$VIDEOMME_DIR" ]; then
  echo ""
  echo "[1/6] VideoMME: prepare + precompute + evaluate"
  echo "----------------------------------------------"
  if [ ! -f "$VIDEOMME_DIR/annotations.json" ]; then
    echo "  Preparing annotations from parquet..."
    python experiments/prepare_video_mme_from_parquet.py \
      --data-dir "$VIDEOMME_DIR" \
      --output "$VIDEOMME_DIR/annotations.json" \
      --extract-zips
  else
    echo "  annotations.json exists, skipping prepare"
  fi

  if [ ! -f "$EMB_VIDEOMME/config.json" ] || [ -z "$(find "$EMB_VIDEOMME" -maxdepth 1 -name "*.pt" -print -quit 2>/dev/null)" ]; then
    echo "  Precomputing embeddings..."
    python experiments/precompute_video_mme.py \
      --data-dir "$VIDEOMME_DIR" \
      --out-dir "$EMB_VIDEOMME"
  else
    echo "  Embeddings exist, skipping precompute"
  fi

  echo "  Running evaluation..."
  python experiments/run_dataset.py \
    --dataset video_mme \
    --checkpoint "$CHECKPOINT" \
    --embeddings-dir "$EMB_VIDEOMME"
  echo "  VideoMME done."
else
  echo ""
  echo "[1/6] VideoMME: SKIP (no $VIDEOMME_DIR)"
  echo "  Run: python experiments/download_dataset_subsets.py video_mme --max-samples 100"
  echo "  Or add parquet + videos_chunked_*.zip for full data"
fi

# --- NeXTQA ---
NEXTQA_DIR="extdataset/nextqa"
EMB_NEXTQA="embeddings/nextqa"

if [ -f "$NEXTQA_DIR/annotations.json" ]; then
  echo ""
  echo "[2/6] NeXTQA: precompute + evaluate"
  echo "----------------------------------------------"
  if [ ! -f "$EMB_NEXTQA/config.json" ] || [ -z "$(find "$EMB_NEXTQA" -maxdepth 1 -name "*.pt" -print -quit 2>/dev/null)" ]; then
    echo "  Precomputing embeddings..."
    python experiments/precompute_nextqa.py \
      --data-dir "$NEXTQA_DIR" \
      --out-dir "$EMB_NEXTQA"
  else
    echo "  Embeddings exist, skipping precompute"
  fi

  echo "  Running evaluation..."
  python experiments/run_dataset.py \
    --dataset nextqa \
    --checkpoint "$CHECKPOINT" \
    --embeddings-dir "$EMB_NEXTQA"
  echo "  NeXTQA done."
else
  echo ""
  echo "[2/6] NeXTQA: SKIP (no $NEXTQA_DIR/annotations.json)"
  echo "  Run: python experiments/download_dataset_subsets.py nextqa --max-samples 100"
fi

# --- Charades ---
CHARADES_DIR="extdataset/charades"
EMB_CHARADES="embeddings/charades"

if [ -f "$CHARADES_DIR/annotations.json" ]; then
  echo ""
  echo "[3/6] Charades: precompute + evaluate"
  echo "----------------------------------------------"
  if [ ! -f "$EMB_CHARADES/config.json" ] || [ -z "$(find "$EMB_CHARADES" -maxdepth 1 -name "*.pt" -print -quit 2>/dev/null)" ]; then
    echo "  Precomputing embeddings..."
    python experiments/precompute_charades.py \
      --data-dir "$CHARADES_DIR" \
      --out-dir "$EMB_CHARADES"
  else
    echo "  Embeddings exist, skipping precompute"
  fi

  echo "  Running evaluation..."
  python experiments/run_dataset.py \
    --dataset charades \
    --checkpoint "$CHECKPOINT" \
    --embeddings-dir "$EMB_CHARADES"
  echo "  Charades done."
else
  echo ""
  echo "[3/6] Charades: SKIP (no $CHARADES_DIR/annotations.json)"
  echo "  Run: python experiments/download_dataset_subsets.py charades --max-samples 100"
fi

# --- SData (primary benchmark: A1-A8, movement head) ---
EMB_SDATA="embeddings/sdata_viscop"
if [ -f "$EMB_SDATA/config.json" ] && [ -n "$(find "$EMB_SDATA" -maxdepth 1 -name "*.pt" -print -quit 2>/dev/null)" ]; then
  echo ""
  echo "[4/6] SData: evaluate (zero-shot, populates A1-A8)"
  echo "----------------------------------------------"
  python experiments/run_dataset.py \
    --dataset sdata \
    --checkpoint "$CHECKPOINT" \
    --embeddings-dir "$EMB_SDATA" \
    --zero-shot
  echo "  SData done."
else
  echo ""
  echo "[4/6] SData: SKIP (no $EMB_SDATA - run precompute_sdata_embeddings.py first)"
fi

# --- VIMA-Bench (L1/L2/L3 - requires simulator) ---
EMB_VIMA="embeddings/vima_bench"
if [ -f "$EMB_VIMA/config.json" ] && [ -n "$(find "$EMB_VIMA" -maxdepth 1 -name "*.pt" -print -quit 2>/dev/null)" ]; then
  echo ""
  echo "[5/6] VIMA-Bench: evaluate (populates L1/L2/L3)"
  echo "----------------------------------------------"
  python experiments/run_dataset.py \
    --dataset vima_bench \
    --checkpoint "$CHECKPOINT" \
    --embeddings-dir "$EMB_VIMA"
  echo "  VIMA-Bench done."
else
  echo ""
  echo "[5/6] VIMA-Bench: SKIP (no $EMB_VIMA - requires simulator, see DATASET_DOWNLOAD.md)"
fi

# --- Collect unified table ---
echo ""
echo "[6/6] Collecting cross-dataset results..."
echo "----------------------------------------------"
python experiments/collect_cross_dataset_results.py

echo ""
echo "=========================================="
echo "Done. Primary output (paper-ready, one row):"
echo "  - experiments/results_table.csv"
echo "  - experiments/results_table.tex"
echo "=========================================="
