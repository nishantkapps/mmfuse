# MMFuse SData Model

Multimodal fusion model (vision + audio) for action classification and movement prediction.

## Model Info

- **Model type**: mmfuse_sdata
- **Inputs**: Vision (2 cameras) + Audio embeddings
- **Outputs**: Action class (0-7) + Movement vector (delta_along, delta_lateral, magnitude)
- **Vision encoder**: viscop
- **Audio encoder**: wav2vec

## Files

- `config.json` - Model configuration
- `pytorch_model.bin` - Model weights (fusion + classifier + movement head)
- `movement_config.yaml` - Action-to-movement mapping
- `inference_example.py` - Minimal script to run inference on your embeddings
- `README.md` - This file

## Loading the Model

```python
import torch
import json
from pathlib import Path

model_dir = Path("models/sdata_viscop")
config = json.load(open(model_dir / "config.json"))
weights = torch.load(model_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)

# Use with mmfuse:
# from mmfuse.run.sdata_robot_control import SDataRobotController
# controller = SDataRobotController(checkpoint_path=str(model_dir / "pytorch_model.bin"), ...)
```

## Quick inference (from model directory)

Place this model folder inside the mmfuse project (e.g. `mmfuse/models/sdata_viscop/`), then:

```bash
python models/sdata_viscop/inference_example.py --embeddings-dir /path/to/your/embeddings --num-samples 10
```

Or from anywhere with `PYTHONPATH` set to the mmfuse project root.

## Testing with Different Datasets

1. **Precompute embeddings** for your dataset:
   ```
   python scripts/precompute_sdata_embeddings.py --dataset your_dataset --out-dir embeddings/your_data --vision-encoder viscop --audio-encoder wav2vec --cross-pair
   ```

2. **Run test script**:
   ```
   python scripts/test_sdata_model.py --checkpoint models/sdata_viscop/pytorch_model.bin --embeddings-dir embeddings/your_data --num-samples 20
   ```

3. **Run evaluation** (accuracy, confusion matrix, ROC):
   ```
   python scripts/evaluate_sdata.py --checkpoint models/sdata_viscop/pytorch_model.bin --embeddings-dir embeddings/your_data --out-dir results/your_data
   ```

## Action Classes (part0..part7)

- 0: Start the Massage
- 1: Focus Here
- 2: Move down a little bit
- 3: Go Back Up
- 4: Stop. Pause for a second
- 5: Move to the Left
- 6: Move to the Right
- 7: Right there, perfect spot

## Requirements

- PyTorch
- mmfuse package (or add project to PYTHONPATH)
- Precomputed embeddings (vision + audio) in same format as training
