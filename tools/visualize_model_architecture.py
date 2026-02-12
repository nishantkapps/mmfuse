#!/usr/bin/env python3
"""
Model Architecture Visualization Tool

Visualizes the mmfuse multimodal pipeline using torchviz (computation graph).

Usage:
  python tools/visualize_model_architecture.py --out architecture.png

Requirements:
  pip install torchviz graphviz
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def build_pipeline_model(device: str = "cpu"):
    """
    Build the full pipeline: encoders + fusion + controller.
    Uses the same architecture as training/streaming.
    """
    from encoders.vision_encoder import VisionEncoder
    from encoders.audio_encoder_learnable import AudioEncoder
    from encoders.sensor_encoder import PressureSensorEncoder, EMGSensorEncoder
    from fusion.multimodal_fusion import MultimodalFusion
    from ctrl.robotic_arm_controller import RoboticArmController3DOF

    vision = VisionEncoder(device=device)
    audio = AudioEncoder(device=device, output_dim=768)
    pressure = PressureSensorEncoder(output_dim=256)
    emg = EMGSensorEncoder(output_dim=256, num_channels=3)
    fusion = MultimodalFusion(
        modality_dims={"vision": 512, "audio": 768, "pressure": 256, "emg": 256},
        fusion_dim=512,
        fusion_method="concat_project",
    )
    controller = RoboticArmController3DOF(embedding_dim=512, device=device)

    return {
        "vision": vision,
        "audio": audio,
        "pressure": pressure,
        "emg": emg,
        "fusion": fusion,
        "controller": controller,
    }


def create_dummy_inputs(batch_size: int = 2, device: str = "cpu"):
    """Create dummy inputs matching the pipeline's expected shapes."""
    return {
        "vision": torch.randn(batch_size, 3, 224, 224, device=device),
        "audio": torch.randn(batch_size, 40000, device=device),  # 2.5s @ 16kHz
        "pressure": torch.randn(batch_size, 100, device=device),
        "emg": torch.randn(batch_size, 100, device=device),
    }


def run_forward(models, inputs):
    """Run forward pass and return fused embedding + output."""
    v = models["vision"](inputs["vision"])
    a = models["audio"](inputs["audio"])  # (B, num_samples) - encoder adds channel dim
    p = models["pressure"](inputs["pressure"])
    e = models["emg"](inputs["emg"])

    fused = models["fusion"]({"vision": v, "audio": a, "pressure": p, "emg": e})
    out = models["controller"].decode(fused)
    return fused, out


def visualize_torchviz(models, inputs, out_path: str):
    """
    Use torchviz to visualize the computation graph.
    Requires: pip install torchviz graphviz
    """
    try:
        from torchviz import make_dot
    except ImportError:
        print("torchviz not installed. Run: pip install torchviz graphviz")
        return False

    fused, out = run_forward(models, inputs)
    # Use fused embedding for a cleaner graph (controller adds many nodes)
    dot = make_dot(fused, params=dict(list(models["fusion"].named_parameters())))
    dot.render(out_path.replace(".png", ""), format="png", cleanup=True)
    print(f"Saved computation graph to {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Visualize mmfuse model architecture with torchviz"
    )
    parser.add_argument(
        "--out",
        default="architecture",
        help="Output path (e.g. architecture.png)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device (cpu or cuda)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for dummy inputs",
    )
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Building pipeline on {device}...")

    models = build_pipeline_model(device)
    for m in models.values():
        m.eval()

    inputs = create_dummy_inputs(args.batch_size, device)

    png_path = args.out if args.out.endswith(".png") else args.out + ".png"
    visualize_torchviz(models, inputs, png_path)


if __name__ == "__main__":
    main()
