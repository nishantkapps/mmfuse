#!/usr/bin/env python3
"""
Export trained sdata model to a shareable format.

Usage (single .pth file):
  python scripts/export_sdata_model.py --checkpoint runs/sdata_viscop/ckpt_sdata_epoch_10.pt \\
      --output-file models/sdata_model.pth

Usage (directory with config, README, etc.):
  python scripts/export_sdata_model.py --checkpoint runs/sdata_viscop/ckpt_sdata_epoch_10.pt \\
      --output-dir models/sdata_viscop
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, help='Path to checkpoint (.pt)')
    p.add_argument('--output-dir', default='models/sdata_viscop',
                   help='Output directory for exported model (default: models/sdata_viscop)')
    p.add_argument('--output-file', metavar='PATH',
                   help='Output a single .pth file to PATH instead of creating a directory')
    p.add_argument('--movement-config', default='config/sdata_movement_config.yaml',
                   help='Movement config to include')
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)

    # Single .pth file output
    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(ckpt), out_path)
        print(f"Saved model to {out_path.absolute()}")
        print(f"Load with: torch.load('{out_path}', map_location='cpu', weights_only=True)")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build config.json (HuggingFace-style)
    config = {
        "model_type": "mmfuse_sdata",
        "num_classes": ckpt.get('num_classes', 8),
        "fusion_dim": ckpt.get('fusion_dim', 256),
        "vision_dim": ckpt.get('vision_dim', 3584),
        "audio_dim": 768,
        "vision_encoder": ckpt.get('vision_encoder', 'viscop'),
        "audio_encoder": ckpt.get('audio_encoder', 'wav2vec'),
        "has_movement_head": 'movement_state' in ckpt,
        "epoch": ckpt.get('epoch', 0),
    }
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Save full checkpoint (pytorch_model.bin) - drop-in replacement for SDataRobotController, test, eval
    model_bin = dict(ckpt)
    torch.save(model_bin, out_dir / 'pytorch_model.bin')

    # Copy movement config
    mov_config = Path(args.movement_config)
    if not mov_config.is_absolute():
        mov_config = Path(__file__).resolve().parent.parent / mov_config
    if mov_config.exists():
        shutil.copy(mov_config, out_dir / 'movement_config.yaml')

    # inference_example.py - minimal script teammates can run from the model dir
    inference_script = '''#!/usr/bin/env python3
"""Minimal inference example - run from this model directory."""
import argparse
import json
import sys
from pathlib import Path

# Add mmfuse project root (parent of models/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings-dir", required=True, help="Path to precomputed embeddings")
    p.add_argument("--num-samples", type=int, default=5)
    args = p.parse_args()

    model_dir = Path(__file__).resolve().parent
    weights_path = model_dir / "pytorch_model.bin"
    config_path = model_dir / "config.json"

    if not weights_path.exists():
        print("pytorch_model.bin not found")
        sys.exit(1)

    config = json.load(open(config_path))
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)

    # Import after path setup
    from mmfuse.encoders.sensor_encoder import PressureSensorEncoder, EMGSensorEncoder
    from mmfuse.fusion.multimodal_fusion import MultimodalFusionWithAttention

    class ActionClassifier(torch.nn.Module):
        def __init__(self, embedding_dim, num_classes):
            super().__init__()
            self.fc = torch.nn.Linear(embedding_dim, num_classes)
        def forward(self, x):
            return self.fc(x)

    class MovementHead(torch.nn.Module):
        def __init__(self, embedding_dim):
            super().__init__()
            self.fc = torch.nn.Linear(embedding_dim, 3)
        def forward(self, x):
            return self.fc(x)

    fusion_dim = config["fusion_dim"]
    num_classes = config["num_classes"]
    vision_dim = config["vision_dim"]

    fusion = MultimodalFusionWithAttention(
        vision_dim=vision_dim, audio_dim=768, pressure_dim=256, emg_dim=256,
        output_dim=fusion_dim, num_heads=8, dropout=0.2
    )
    classifier = ActionClassifier(fusion_dim, num_classes)
    movement_head = MovementHead(fusion_dim) if config.get("has_movement_head") else None

    fusion.load_state_dict(ckpt["fusion_state"])
    classifier.load_state_dict(ckpt["model_state"])
    if movement_head and "movement_state" in ckpt:
        movement_head.load_state_dict(ckpt["movement_state"])

    fusion.eval()
    classifier.eval()
    if movement_head:
        movement_head.eval()

    # Load a few samples
    samples = sorted(Path(args.embeddings_dir).glob("*.pt"))[: args.num_samples]
    if not samples:
        print("No .pt files in", args.embeddings_dir)
        sys.exit(1)

    pressure_enc = PressureSensorEncoder(output_dim=256, input_features=2)
    emg_enc = EMGSensorEncoder(output_dim=256, num_channels=3, input_features=4)

    correct = 0
    for pth in samples:
        d = torch.load(pth, map_location="cpu", weights_only=True)
        v1 = d["vision_camera1"].unsqueeze(0).float()
        v2 = d["vision_camera2"].unsqueeze(0).float()
        a = d["audio"].unsqueeze(0).float()
        t = d["target"].item()
        v1[~torch.isfinite(v1)] = 0.0
        v2[~torch.isfinite(v2)] = 0.0
        a[~torch.isfinite(a)] = 0.0
        p_emb = pressure_enc(torch.zeros(1, 2))
        e_emb = emg_enc(torch.zeros(1, 4))
        emb = {"vision_camera1": v1, "vision_camera2": v2, "audio": a, "pressure": p_emb, "emg": e_emb}
        with torch.no_grad():
            fused = fusion(emb)
            logits = classifier(fused)
            pred = logits.argmax(dim=1).item()
            mov = movement_head(fused).squeeze().tolist() if movement_head else []
        correct += 1 if pred == t else 0
        mov_str = f" movement={mov}" if mov else ""
        print(f"{pth.name}: true={t} pred={pred}{mov_str}")

    print(f"\\nAccuracy on {len(samples)} samples: {correct}/{len(samples)} = {100*correct/len(samples):.1f}%")

if __name__ == "__main__":
    main()
'''
    with open(out_dir / 'inference_example.py', 'w') as f:
        f.write(inference_script)

    # README.md
    readme = """# MMFuse SData Model

Multimodal fusion model (vision + audio) for action classification and movement prediction.

## Model Info

- **Model type**: mmfuse_sdata
- **Inputs**: Vision (2 cameras) + Audio embeddings
- **Outputs**: Action class (0-7) + Movement vector (delta_along, delta_lateral, magnitude)
- **Vision encoder**: {vision_encoder}
- **Audio encoder**: {audio_encoder}

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
""".format(
        vision_encoder=config['vision_encoder'],
        audio_encoder=config['audio_encoder'],
    )
    with open(out_dir / 'README.md', 'w') as f:
        f.write(readme)

    print(f"Exported model to {out_dir.absolute()}")
    print(f"  - config.json")
    print(f"  - pytorch_model.bin")
    print(f"  - movement_config.yaml")
    print(f"  - inference_example.py")
    print(f"  - README.md")
    print(f"\nShare the folder: {out_dir.absolute()}")


if __name__ == '__main__':
    main()
