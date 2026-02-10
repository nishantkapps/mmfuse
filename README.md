# Robotic Multimodal Feedback System

A complete system for encoding and fusing multimodal inputs from robotic sensors using pre-trained deep learning models.

## Overview

This system processes five different input modalities:
- **2 Cameras** (vision) - encoded with CLIP
- **1 Audio input** (microphone) - encoded with Wav2Vec 2.0
- **1 Pressure sensor** - encoded with neural network
- **1 EMG sensor** (8 channels) - encoded with neural network

All inputs are encoded independently using pre-trained models, then fused into a common embedding space.

## Architecture

### Components

```
Input Modalities
    ├── Camera 1 ──┐
    ├── Camera 2 ──┤
    ├── Audio ─────┤──> [Encoders] ──> [Projection] ──> [Fusion] ──> Fused Embedding
    ├── Pressure ──┤
    └── EMG ───────┘
```

### Encoders

1. **Vision Encoder (CLIP)**
   - Pre-trained on 400M image-text pairs
   - Provides robust visual features
   - Output: 512-dim embeddings (ViT-B/32)

2. **Audio Encoder (Wav2Vec 2.0)**
   - Pre-trained on 960 hours of speech
   - Self-supervised learning on unlabeled audio
   - Output: 768-dim embeddings

3. **Pressure Sensor Encoder**
   - Extracts temporal features (mean, std, min, max, energy)
   - Neural network projection to common space
   - Output: 256-dim embeddings

4. **EMG Sensor Encoder**
   - Handles 8-channel EMG signals
   - Temporal feature extraction
   - Neural network projection to common space
   - Output: 256-dim embeddings

### Fusion Methods

1. **Concatenation + Projection** (Default)
   - Concatenates all modality embeddings
   - Projects to target dimension via MLP
   - Fast and effective

2. **Weighted Sum**
   - Learns attention weights for each modality
   - Lightweight fusion mechanism

3. **Attention-based Fusion**
   - Cross-modal attention mechanisms
   - Models inter-modality relationships
   - More expressive but computationally heavier

## Installation

```bash
# Clone or navigate to the project
cd /path/to/mmfuse

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from robotic_feedback_system import RoboticFeedbackSystem
import torch

# Initialize system
system = RoboticFeedbackSystem(
    fusion_dim=512,
    fusion_method="concat_project",
    use_attention=False,
    device="cuda"
)
system.eval()

# Prepare inputs
camera1 = torch.randn(batch_size, 3, 224, 224)  # RGB image
camera2 = torch.randn(batch_size, 3, 224, 224)
audio = torch.randn(batch_size, 48000)           # 16kHz, 3 seconds
pressure = torch.randn(batch_size, 1000)         # Sequence of readings
emg = torch.randn(batch_size, 8, 1000)           # 8 channels

# Process and fuse
with torch.no_grad():
    fused_embedding = system(
        camera_images={'camera1': camera1, 'camera2': camera2},
        audio=audio,
        pressure=pressure,
        emg=emg
    )

print(fused_embedding.shape)  # (batch_size, 512)
```

### Get Individual Modality Embeddings

```python
with torch.no_grad():
    fused, modality_embeddings = system(
        camera_images={'camera1': camera1, 'camera2': camera2},
        audio=audio,
        pressure=pressure,
        emg=emg,
        return_modality_embeddings=True
    )

# Access individual modality embeddings
vision_emb = modality_embeddings['vision']      # (batch, 512)
audio_emb = modality_embeddings['audio']        # (batch, 768)
pressure_emb = modality_embeddings['pressure']  # (batch, 256)
emg_emb = modality_embeddings['emg']            # (batch, 256)
```

### Use Individual Encoders Only

```python
# Vision only
vision_embedding = system.encode_vision_only({'camera1': cam1, 'camera2': cam2})

# Audio only
audio_embedding = system.encode_audio_only(audio)

# Pressure only
pressure_embedding = system.encode_pressure_only(pressure)

# EMG only
emg_embedding = system.encode_emg_only(emg)
```

### Get System Information

```python
# Check embedding dimensions
dims = system.get_modality_dimensions()
print(dims)
# Output: {'vision': 512, 'audio': 768, 'pressure': 256, 'emg': 256, 'fused': 512}

# Freeze pre-trained encoders
system.freeze_encoders()

# Unfreeze for fine-tuning
system.unfreeze_encoders()
```

## Running Demos

```bash
python demo.py
```

This runs 5 demonstration scripts:
1. Basic multimodal fusion
2. Attention-based fusion
3. Individual modality encoders
4. Batch processing with different sizes
5. Cross-modal similarity analysis

## Project Structure

```
mmfuse/
├── encoders/
│   ├── __init__.py
│   ├── vision_encoder.py      # CLIP-based vision encoder
│   ├── audio_encoder.py        # Wav2Vec 2.0 audio encoder
│   └── sensor_encoder.py       # Neural network sensor encoders
├── fusion/
│   ├── __init__.py
│   └── multimodal_fusion.py    # Fusion modules (concat, weighted, attention)
├── preprocessing/
│   ├── __init__.py
│   └── preprocessor.py         # Preprocessing for each modality
├── robotic_feedback_system.py  # Main integrated system
├── demo.py                      # Demonstration scripts
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Key Features

- ✅ Pre-trained encoders (no training required)
- ✅ Handles variable batch sizes
- ✅ Multiple fusion strategies
- ✅ Modular and extensible
- ✅ GPU support with automatic fallback to CPU
- ✅ Individual modality access
- ✅ Cross-modal analysis capabilities

## Model Details

### Vision: CLIP
- **Architecture**: Vision Transformer (ViT-B/32)
- **Training Data**: 400M image-text pairs
- **Output Dimension**: 512 (ViT-B/32) or 768 (ViT-L/14)

### Audio: Wav2Vec 2.0
- **Architecture**: Transformer-based
- **Training Data**: 960 hours of unlabeled speech
- **Output Dimension**: 768 (base model)

### Sensors: Custom Neural Networks
- **Architecture**: 2-layer MLP with batch normalization
- **Input**: Statistical features (mean, std, min, max, energy)
- **Output Dimension**: 256

## Typical Use Cases

1. **Robot Control**: Fused representation as input to control policies
2. **Anomaly Detection**: Detect unusual robot states from multimodal feedback
3. **Task Learning**: Learn from multimodal demonstrations
4. **Sensor Fusion**: Combine heterogeneous sensor inputs
5. **Situation Awareness**: Understand robot environment from multiple modalities

## Performance Considerations

- **Memory**: ~6GB with CUDA (Vision + Audio encoders + fusion)
- **Inference Time**: ~100-200ms per batch (GPU)
- **Batch Size**: Tested up to batch_size=32

## Future Extensions

- [ ] Support for additional sensor modalities (IMU, LiDAR, etc.)
- [ ] Fine-tuning options for task-specific fusion
- [ ] Temporal fusion for video/sequence inputs
- [ ] Uncertainty quantification
- [ ] Contrastive learning objectives

## References

- CLIP: [Learning Transferable Models for Composable Vision Tasks](https://arxiv.org/abs/2103.14030)
- Wav2Vec 2.0: [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477)

## License

MIT License

## Author
Nishant Killedar
AI Researcher 
CSIS Department
BITS Pilani Hyderabad Campus
Built for robotic multimodal fusion applications.
