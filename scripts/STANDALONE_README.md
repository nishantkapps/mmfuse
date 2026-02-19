# Standalone SData Inference

Run the trained model **without the mmfuse codebase**. Only PyTorch and NumPy required.

## What to share with teammates

1. **`standalone_sdata_inference.py`** – this script (no mmfuse imports)
2. **`model.pth`** – exported model (from `export_sdata_model.py --output-file model.pth`)
3. **Precomputed embeddings** – `.pt` files with keys: `vision_camera1`, `vision_camera2`, `audio`, `target`

## Requirements

```
pip install torch numpy
```

## Usage

```bash
python standalone_sdata_inference.py --checkpoint model.pth --embeddings-dir /path/to/embeddings [--num-samples 10] [--output-json results.json]
```

## Output

For each sample:
- **Action**: predicted class (0–7)
- **Location**: `delta_along`, `delta_lateral`, `magnitude`

## Embedding format

Each `.pt` file must contain:
- `vision_camera1`: tensor of shape `(vision_dim,)` (e.g. 3584 for VisCoP)
- `vision_camera2`: tensor of shape `(vision_dim,)`
- `audio`: tensor of shape `(768,)` (wav2vec)
- `target`: int, ground-truth action class (0–7)

Optional: `config.json` in the embeddings dir with `vision_dim` and `num_classes`.
