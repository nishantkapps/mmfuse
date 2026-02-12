#!/usr/bin/env python3
"""
Generate publication-ready architecture diagram for mmfuse using VisualTorch.

Uses the visualtorch library to create layered/graph visualizations of the
model architecture with automatic layer detection and dimension display.

Usage:
  python tools/generate_paper_diagram.py --out paper_architecture.png
  python tools/generate_paper_diagram.py --style graph --out paper_architecture.png

Requirements:
  pip install visualtorch
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn


def build_architecture_model():
    """
    Build an nn.Module that represents the mmfuse architecture for visualization.
    Matches the flow: Projection → KL → Cross-Modal Alignment → Fusion MLP → Controller.
    """
    class CrossModalAlignment(nn.Module):
        """Cross-modal attention (8 heads). Input: (B, 2048) = (B, 4*512)."""

        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=512, num_heads=8, dropout=0.2, batch_first=True
            )

        def forward(self, x):
            # x: (B, 2048) -> (B, 4, 512)
            x = x.view(x.size(0), 4, 512)
            x, _ = self.attn(x, x, x)
            return x.reshape(x.size(0), -1)  # (B, 2048)

    return nn.Sequential(
        # Project to common dimension (512 each × 4 = 2048 concat)
        nn.Linear(2048, 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(),
        # Cross-Modal Alignment (includes KL distillation in practice)
        CrossModalAlignment(),
        # Fusion MLP: 2048 -> 1024 -> 512
        nn.Linear(2048, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, 512),
        # Controller: 512 -> 256 -> 128 -> 4
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 4),
    )


def build_encoder_models():
    """Build simplified encoder models for individual visualization."""
    # Vision encoder (simplified representation of CLIP ViT)
    vision = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 512),
    )
    # Audio encoder (1D-CNN: 4 conv layers + projection)
    audio = nn.Sequential(
        nn.Conv1d(1, 256, kernel_size=15, stride=2, padding=7),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Conv1d(256, 256, kernel_size=15, stride=2, padding=7),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Conv1d(256, 256, kernel_size=15, stride=2, padding=7),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Conv1d(256, 256, kernel_size=15, stride=2, padding=7),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 768),
    )
    # Pressure/EMG (MLP: 3 layers)
    sensor = nn.Sequential(
        nn.Linear(100, 128),
        nn.LayerNorm(128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 128),
        nn.LayerNorm(128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 256),
    )
    return {"vision": vision, "audio": audio, "pressure": sensor, "emg": sensor}


def draw_with_visualtorch(
    out_path: str,
    style: str = "layered",
    dpi: int = 150,
):
    """Generate diagram using visualtorch."""
    try:
        import visualtorch
    except ImportError:
        print("visualtorch not installed. Run: pip install visualtorch")
        return False

    import matplotlib.pyplot as plt

    # Main pipeline: fusion + controller (from concatenated 2048-dim)
    model = build_architecture_model()
    model.eval()
    input_shape = (2, 2048)  # batch=2 for BatchNorm

    if style == "layered":
        img = visualtorch.layered_view(
            model,
            input_shape=input_shape,
            legend=True,
            draw_volume=True,
            spacing=15,
            padding=20,
        )
    else:
        img = visualtorch.graph_view(
            model,
            input_shape=input_shape,
            layer_spacing=200,
            node_size=40,
        )

    # visualtorch returns PIL Image
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(
        "Multimodal Robotic Feedback System\n"
        "Vision + Audio + Pressure + EMG → 3DOF Arm Control",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved architecture diagram to {out_path}")
    return True


def draw_encoders_separately(out_dir: str, dpi: int = 150):
    """Generate separate encoder diagrams and save to output directory."""
    try:
        import visualtorch
    except ImportError:
        print("visualtorch not installed. Run: pip install visualtorch")
        return False

    import matplotlib.pyplot as plt

    encoders = build_encoder_models()
    shapes = {
        "vision": (1, 3, 224, 224),
        "audio": (1, 1, 40000),
        "pressure": (1, 100),
        "emg": (1, 100),
    }

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for name, model in encoders.items():
        try:
            img = visualtorch.layered_view(
                model, input_shape=shapes[name], legend=True, spacing=10
            )
            path = Path(out_dir) / f"encoder_{name}.png"
            plt.figure(figsize=(10, 6))
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"{name.capitalize()} Encoder")
            plt.tight_layout()
            plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
            plt.close()
            print(f"  Saved {path}")
        except Exception as e:
            print(f"  Skipped {name}: {e}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate architecture diagram using VisualTorch"
    )
    parser.add_argument("--out", default="paper_architecture.png", help="Output path")
    parser.add_argument(
        "--style",
        choices=["layered", "graph"],
        default="layered",
        help="Visualization style: layered (3D blocks) or graph (node links)",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Output DPI")
    parser.add_argument(
        "--encoders-dir",
        default=None,
        help="If set, also save individual encoder diagrams to this directory",
    )
    args = parser.parse_args()

    out_path = args.out if args.out.endswith(".png") else args.out + ".png"
    draw_with_visualtorch(out_path, style=args.style, dpi=args.dpi)

    if args.encoders_dir:
        print(f"\nGenerating encoder diagrams...")
        draw_encoders_separately(args.encoders_dir, dpi=args.dpi)


if __name__ == "__main__":
    main()
