# Cross-Dataset Experiments (VisCoP-Aligned Benchmarks)

Evaluate MMFuse on multiple datasets for the paper. Datasets are stored in `extdataset/<name>/`.

## Dataset Folders (create these)

Create folders for each dataset under `extdataset/`:

```
extdataset/
├── video_mme/    # VideoMME - 900 videos, 2,700 QA pairs
├── nextqa/       # NeXTQA - videos + QA
├── charades/     # Charades (ADL-X) - 9,848 videos
├── egoschema/    # EgoSchema - egocentric QA
├── vima_bench/   # VIMA-Bench - simulator or pre-recorded
└── sdata/        # Optional: symlink or copy of dataset/sdata
```

## Pipeline

### 1. Prepare data

Download each dataset into `extdataset/<name>/` following the format expected by the precompute scripts.

- **Quick subsets:** `python experiments/download_dataset_subsets.py video_mme --max-samples 100`
- **Full guide:** See `experiments/DATASET_DOWNLOAD.md` for sources and subset instructions
- **Format details:** See `experiments/DATASET_FORMATS.md`

### 2. Precompute embeddings

For each dataset, run the precompute script to create embeddings (vision + audio/text):

```bash
# SData (existing)
python scripts/precompute_sdata_embeddings.py --dataset dataset/sdata \
    --out-dir embeddings/sdata_viscop --vision-encoder viscop --audio-encoder wav2vec --cross-pair

# Charades
python experiments/precompute_charades.py --out-dir embeddings/charades

# VideoMME
python experiments/precompute_video_mme.py --out-dir embeddings/video_mme

# NeXTQA
python experiments/precompute_nextqa.py --out-dir embeddings/nextqa

# EgoSchema
python experiments/precompute_egoschema.py --out-dir embeddings/egoschema

# Generic (any dataset with annotations.json: video_path, text, target)
python experiments/precompute_video_text.py --dataset <name> --out-dir embeddings/<name>
```

### 3. Run evaluation

**Full pipeline (recommended):** Prepare + precompute + evaluate + collect table for all datasets:

```bash
./experiments/run_all_cross_dataset_experiments.sh
```

Auto-picks latest checkpoint from `checkpoints/ckpt_sdata_epoch_*.pt`. Requires:
- `embeddings/sdata_viscop/` (for A1–A8: run `precompute_sdata_embeddings.py`)
- `extdataset/video_mme/` with annotations.json or parquet + videos_chunked_*.zip
- `extdataset/nextqa/annotations.json`, `extdataset/charades/annotations.json` (run `download_dataset_subsets.py`)

For L1/L2/L3: VIMA-Bench (simulator). See `PAPER_TABLE_GUIDE.md` for VisCoP vs MMFuse narrative.

**Evaluate only** (embeddings already precomputed):

```bash
python experiments/run_all_experiments.py
# Auto-picks latest checkpoint; or pass: --checkpoint path/to/ckpt_sdata_epoch_N.pt
```

**Single dataset:**
```bash
python experiments/run_dataset.py --dataset video_mme --checkpoint path/to/model.pt
```

### 4. Consolidated results (experiments table)

`run_all_cross_dataset_experiments.sh` and `run_all_experiments.py` both run `collect_cross_dataset_results.py` at the end. You can also run it manually:

```bash
python experiments/collect_cross_dataset_results.py
```

**Outputs (VisCoP-style unified table):**
- `experiments/cross_dataset_summary.csv` – per-dataset summary
- `experiments/cross_dataset_unified.csv` – L1 | L2 | L3 | A1–A8 | EgoSchema | NeXTQA | VideoMME | ADL-X | Avg
- `experiments/cross_dataset_summary.html` – HTML table
- `experiments/cross_dataset_summary.tex` – LaTeX tables
- `experiments/figures/cross_dataset_accuracy.png` – bar chart

## Results Structure

```
experiments/results/
├── video_mme/results.json
├── nextqa/results.json
├── charades/results.json
├── egoschema/results.json
├── vima_bench/results.json
└── sdata/results.json
```

## Embedding Format

Each precomputed `.pt` file must contain:
- `vision_camera1`: tensor (vision_dim,) - use same frame for both cams if single-camera
- `vision_camera2`: tensor (vision_dim,)
- `audio`: tensor (768,) - use text encoder output for video+text datasets
- `target`: int - action class or MCQ option index

`config.json` in embeddings dir:
```json
{"vision_dim": 3584, "audio_dim": 768, "num_classes": N}
```

## Movement Head

SData and (where applicable) robotic datasets support movement head evaluation. Results include `movement_mse` when ground truth is available. This is unique to our model vs. VisCoP.
