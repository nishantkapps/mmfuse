# VideoMME Experiment Setup

## 1. Create directory and add data

```bash
mkdir -p extdataset/video_mme
cd extdataset/video_mme
```

Place in `extdataset/video_mme/`:
- **\*.parquet** – mapping file (video_id, question, options, answer)
- **videos_chunked_01.zip** (and 02, 03… if you have more) – video files
- **subtitle.zip** – optional (skipped during extraction)

## 2. Install pandas (for parquet)

```bash
pip install pandas pyarrow
```

## 3. Run the experiment

```bash
# From project root - uses trained model automatically
./experiments/run_video_mme_experiment.sh
```

The script auto-finds the model (ablation_runs/full, checkpoints/, or models/sdata_viscop).

Override if needed:
```bash
DATA_DIR=extdataset/video_mme CHECKPOINT=path/to/model.pt ./experiments/run_video_mme_experiment.sh
```

## What the script does

1. **Find model** – Uses last checkpoint from training (`checkpoints/ckpt_sdata_epoch_*.pt`) or exported model (`models/sdata_viscop/pytorch_model.bin`, `model.pth`)
2. **Prepare** – Reads parquet, extracts video zips to `videos/`, writes `annotations.json`
3. **Precompute** – Encodes vision (VisCoP) + text (CLIP) → embeddings
4. **Evaluate** – Runs MMFuse on embeddings, reports accuracy

## Output

- Embeddings: `embeddings/video_mme/*.pt`
- Results: `experiments/results/video_mme/results.json`
