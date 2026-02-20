#!/bin/bash
# Run VideoMME experiment using the trained model.
# Data: extdataset/video_mme/ (parquet + video zips or annotations from download)
# Model: uses last checkpoint from training (checkpoints/) OR exported .pth/pytorch_model.bin
#
# Usage:
#   ./experiments/run_video_mme_experiment.sh
#
# Override paths:
#   DATA_DIR=extdataset/video_mme CHECKPOINT=path/to/model.pt ./experiments/run_video_mme_experiment.sh

set -e
cd "$(dirname "$0")/.."
PROJ_ROOT="$(pwd)"

# Data dir: extdataset/video_mme (download subset or full parquet+zip)
DATA_DIR="${DATA_DIR:-extdataset/video_mme}"
EMBEDDINGS_DIR="${EMBEDDINGS_DIR:-embeddings/video_mme}"

# Use trained model only: last checkpoint from training OR exported .pth
find_checkpoint() {
  # 1. Last checkpoint from model training (training/train_sdata_attention.py)
  latest=$(ls checkpoints/ckpt_sdata_epoch_*.pt 2>/dev/null | sort -V | tail -1)
  [ -n "$latest" ] && echo "$latest" && return
  # 2. Exported model: .pth or pytorch_model.bin
  [ -f models/sdata_viscop/pytorch_model.bin ] && echo models/sdata_viscop/pytorch_model.bin && return
  [ -f models/sdata_model.pth ] && echo models/sdata_model.pth && return
  [ -f model.pth ] && echo model.pth && return
  echo ""
}

CHECKPOINT="${CHECKPOINT:-$(find_checkpoint)}"
if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
  echo "No trained model found. Use one of:"
  echo "  1. Last checkpoint: checkpoints/ckpt_sdata_epoch_*.pt"
  echo "     (from: python training/train_sdata_attention.py --use-precomputed --embeddings-dir embeddings/sdata_viscop)"
  echo "  2. Exported model: models/sdata_viscop/pytorch_model.bin or model.pth"
  echo "     (from: python scripts/export_sdata_model.py --checkpoint checkpoints/ckpt_sdata_epoch_N.pt --output-dir models/sdata_viscop)"
  echo ""
  echo "Or pass explicitly: CHECKPOINT=path/to/model.pt ./experiments/run_video_mme_experiment.sh"
  exit 1
fi

echo "=========================================="
echo "VideoMME Experiment"
echo "=========================================="
echo "Data dir:      $DATA_DIR"
echo "Embeddings:    $EMBEDDINGS_DIR"
echo "Checkpoint:    $CHECKPOINT"
echo "=========================================="

if [ ! -d "$DATA_DIR" ]; then
  echo "Data dir not found: $DATA_DIR"
  echo "Create extdataset/video_mme/ and add parquet + video zip(s) or run download_dataset_subsets.py video_mme"
  exit 1
fi

# Step 1: Prepare annotations from parquet (skip if annotations.json exists)
if [ ! -f "$DATA_DIR/annotations.json" ]; then
  echo ""
  echo "[1/3] Preparing annotations from parquet..."
  python experiments/prepare_video_mme_from_parquet.py \
    --data-dir "$DATA_DIR" \
    --output "$DATA_DIR/annotations.json" \
    --extract-zips
  if [ ! -f "$DATA_DIR/annotations.json" ]; then
    echo "Failed to create annotations.json. Check parquet file in $DATA_DIR"
    exit 1
  fi
else
  echo ""
  echo "[1/3] Skipping prepare (annotations.json exists)"
fi

# Step 2: Precompute embeddings (skip if embeddings already exist)
if [ -f "$EMBEDDINGS_DIR/config.json" ] && [ -n "$(find "$EMBEDDINGS_DIR" -maxdepth 1 -name "*.pt" -print -quit 2>/dev/null)" ]; then
  echo ""
  echo "[2/3] Skipping precompute (embeddings already exist in $EMBEDDINGS_DIR)"
else
  echo ""
  echo "[2/3] Precomputing embeddings (vision + text)..."
  python experiments/precompute_video_mme.py \
    --data-dir "$DATA_DIR" \
    --out-dir "$EMBEDDINGS_DIR"
  if [ ! -f "$EMBEDDINGS_DIR/config.json" ]; then
    echo "Precompute failed. Check videos are extracted and accessible."
    exit 1
  fi
fi

# Step 3: Run evaluation (dataset key is video_mme for config)
echo ""
echo "[3/3] Running MMFuse evaluation..."
python experiments/run_dataset.py \
  --dataset video_mme \
  --checkpoint "$CHECKPOINT" \
  --embeddings-dir "$EMBEDDINGS_DIR"

echo ""
echo "=========================================="
echo "VideoMME experiment complete."
echo "Results: experiments/results/video_mme/results.json"
echo "=========================================="
