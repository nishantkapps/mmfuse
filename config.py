"""
Configuration for Robotic Multimodal Feedback System
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class VisionConfig:
    """Vision encoder configuration"""
    model_name: str = "ViT-B/32"  # CLIP variant
    image_size: tuple = (224, 224)
    frozen: bool = True
    normalize: bool = True


@dataclass
class AudioConfig:
    """Audio encoder configuration"""
    model_name: str = "facebook/wav2vec2-base"
    sampling_rate: int = 16000
    duration: float = 3.0  # seconds
    frozen: bool = True
    normalize: bool = True


@dataclass
class PressureConfig:
    """Pressure sensor configuration"""
    num_channels: int = 1
    output_dim: int = 256
    window_size: int = 50
    normalize: bool = True
    standardize: bool = True


@dataclass
class EMGConfig:
    """EMG sensor configuration"""
    num_channels: int = 8
    output_dim: int = 256
    window_size: int = 50
    normalize: bool = True
    standardize: bool = True


@dataclass
class FusionConfig:
    """Fusion module configuration"""
    fusion_dim: int = 512
    fusion_method: str = "concat_project"  # or "weighted_sum", "bilinear"
    use_attention: bool = False
    num_attention_heads: int = 8
    dropout: float = 0.2


@dataclass
class SystemConfig:
    """Complete system configuration"""
    # Components
    vision: VisionConfig = None
    audio: AudioConfig = None
    pressure: PressureConfig = None
    emg: EMGConfig = None
    fusion: FusionConfig = None
    
    # Runtime
    device: str = "cuda"
    batch_size: int = 8
    num_workers: int = 4
    
    def __post_init__(self):
        """Initialize defaults if not provided"""
        if self.vision is None:
            self.vision = VisionConfig()
        if self.audio is None:
            self.audio = AudioConfig()
        if self.pressure is None:
            self.pressure = PressureConfig()
        if self.emg is None:
            self.emg = EMGConfig()
        if self.fusion is None:
            self.fusion = FusionConfig()


# Preset configurations for different scenarios

# Lightweight configuration for resource-constrained robots
LIGHTWEIGHT_CONFIG = SystemConfig(
    vision=VisionConfig(model_name="ViT-B/32"),
    audio=AudioConfig(model_name="facebook/wav2vec2-base"),
    fusion=FusionConfig(fusion_dim=256, fusion_method="weighted_sum"),
    batch_size=4
)

# Balanced configuration (recommended)
BALANCED_CONFIG = SystemConfig(
    vision=VisionConfig(model_name="ViT-B/32"),
    audio=AudioConfig(model_name="facebook/wav2vec2-base"),
    fusion=FusionConfig(fusion_dim=512, fusion_method="concat_project"),
    batch_size=8
)

# High-capacity configuration for complex tasks
HIGH_CAPACITY_CONFIG = SystemConfig(
    vision=VisionConfig(model_name="ViT-L/14"),
    audio=AudioConfig(model_name="facebook/wav2vec2-large"),
    fusion=FusionConfig(
        fusion_dim=768,
        fusion_method="concat_project",
        use_attention=True,
        num_attention_heads=8
    ),
    batch_size=16
)


def get_config(preset: str = "balanced") -> SystemConfig:
    """
    Get a preset configuration
    
    Args:
        preset: Configuration preset ('lightweight', 'balanced', 'high_capacity')
    
    Returns:
        SystemConfig instance
    """
    presets = {
        "lightweight": LIGHTWEIGHT_CONFIG,
        "balanced": BALANCED_CONFIG,
        "high_capacity": HIGH_CAPACITY_CONFIG
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
    
    return presets[preset]
