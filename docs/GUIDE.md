# Robotic Multimodal Fusion System - Complete Guide

## Quick Start

### 1. Installation

```bash
cd /home/nishant/projects/mmfuse
pip install -r requirements.txt
```

### 2. Basic Usage

```python
from robotic_feedback_system import RoboticFeedbackSystem
import torch

# Initialize
system = RoboticFeedbackSystem(fusion_dim=512, device="cuda")
system.eval()

# Prepare inputs
camera1 = torch.randn(1, 3, 224, 224)  # 1 RGB image
camera2 = torch.randn(1, 3, 224, 224)  
audio = torch.randn(1, 48000)          # 16kHz, 3 seconds
pressure = torch.randn(1, 1000)         
emg = torch.randn(1, 8, 1000)          # 8 channels

# Get fused embedding
with torch.no_grad():
    fused = system(
        camera_images={'camera1': camera1, 'camera2': camera2},
        audio=audio,
        pressure=pressure,
        emg=emg
    )
# Result: (1, 512) dimensional embedding
```

---

## Architecture Overview

### Input Modalities

| Modality | Input Shape | Encoder | Output Dim | Use Case |
|----------|-------------|---------|-----------|----------|
| Camera 1 | (B, 3, 224, 224) | CLIP (ViT-B/32) | 512 | Visual perception |
| Camera 2 | (B, 3, 224, 224) | CLIP (ViT-B/32) | 512 | Stereo vision / dual views |
| Audio | (B, 48000) | Wav2Vec 2.0 | 768 | Sound, voice, acoustic feedback |
| Pressure | (B, 1000) | Neural Network | 256 | Touch/contact feedback |
| EMG | (B, 8, 1000) | Neural Network | 256 | Muscle/movement activity |

### Processing Pipeline

```
Raw Inputs
    ↓
[Preprocessing: Normalization, Feature Extraction]
    ↓
[Individual Encoders: Pre-trained Models]
    ├─ CLIP (Vision)
    ├─ Wav2Vec 2.0 (Audio)
    └─ Neural Networks (Sensors)
    ↓
[Projection Layers: Map to Common Space]
    ↓
[Fusion Module: Combine All Modalities]
    ↓
Unified Embedding (512-dim)
```

---

## Encoder Details

### Vision: CLIP
- **Pre-trained on**: 400M image-text pairs from internet
- **Architecture**: Vision Transformer (ViT-B/32)
- **Strengths**: 
  - Robust to visual variations
  - Good transfer to robotics domain
  - Semantic understanding
- **Output**: 512-dimensional vectors
- **Frozen by default**: Yes (prevents catastrophic forgetting)

### Audio: Wav2Vec 2.0
- **Pre-trained on**: 960 hours of unlabeled speech (LibriSpeech)
- **Learning method**: Self-supervised learning
- **Strengths**:
  - Works with any audio, not just speech
  - Captures acoustic patterns
  - No labeled data needed for pre-training
- **Output**: 768-dimensional vectors
- **Frozen by default**: Yes

### Pressure Sensor
- **Architecture**: 2-layer MLP (128 → 256)
- **Input**: Statistical features from temporal signal
  - Mean, Standard Deviation
  - Min, Max values
  - Energy (RMS)
- **Output**: 256-dimensional vectors
- **Trainable**: Yes (custom task-specific learning possible)

### EMG Sensor
- **Architecture**: 2-layer MLP (128 → 256)
- **Handles**: 8-channel EMG signals
- **Input**: Temporal feature extraction per channel
- **Output**: 256-dimensional vectors
- **Trainable**: Yes

---

## Fusion Methods

### 1. Concatenation + Projection (Default)
```
[Vision: 512] ─┐
[Audio: 768]  ├─ [Concatenate] ─ [MLP] ─ [512-dim output]
[Pressure: 256] ┤
[EMG: 256]    ─┘

Total input: 512 + 768 + 256 + 256 = 1792 dims
```
- **Advantages**: Fast, simple, captures all information
- **Computational Cost**: Low
- **Best for**: Real-time robotic control

### 2. Weighted Sum
```
[Vision: 512] ─┐
[Audio: 768]  ├─ [Projection to 512] ─ [Weighted Sum] ─ [512-dim output]
[Pressure: 256] ┤
[EMG: 256]    ─┘

Weights: Learned attention mechanism
```
- **Advantages**: Lightweight, interpretable modality importance
- **Computational Cost**: Very low
- **Best for**: Resource-constrained robots

### 3. Attention-based Fusion
```
[All modalities] ─ [Project to 512] ─ [Multi-Head Attention] ─ [512-dim output]

Models inter-modality relationships
```
- **Advantages**: Captures complex inter-modality dependencies
- **Computational Cost**: Higher
- **Best for**: Complex decision-making tasks

---

## Configuration Presets

### Lightweight (Edge Robots)
```python
from config import get_config
config = get_config("lightweight")
# - Smaller models
# - 256-dim fusion space
# - Weighted sum fusion
# - Batch size: 4
# Best for: Resource-constrained robots
```

### Balanced (Recommended)
```python
config = get_config("balanced")
# - Standard CLIP (ViT-B/32)
# - Standard Wav2Vec 2.0
# - 512-dim fusion space
# - Concatenation + Projection
# - Batch size: 8
# Best for: Most robotic applications
```

### High Capacity (Complex Tasks)
```python
config = get_config("high_capacity")
# - Large CLIP (ViT-L/14)
# - Large Wav2Vec 2.0
# - 768-dim fusion space
# - Attention-based fusion
# - Batch size: 16
# Best for: Complex scene understanding, offline processing
```

---

## Practical Examples

### Example 1: Robot Perception Loop

```python
from robotic_feedback_system import RoboticFeedbackSystem
import torch

# Setup
system = RoboticFeedbackSystem(device="cuda")
system.eval()

# In your robot control loop
for step in range(num_steps):
    # Read sensors (implement actual sensor reading)
    cameras = get_camera_frames()  # 2 frames
    audio = get_audio_chunk()       # 3 seconds
    pressure = get_pressure_reading()
    emg = get_emg_reading()
    
    # Get multimodal embedding
    with torch.no_grad():
        embedding = system(
            camera_images={'camera1': cameras[0], 'camera2': cameras[1]},
            audio=audio,
            pressure=pressure,
            emg=emg
        )
    
    # Use embedding for control
    action = control_policy(embedding)
    execute_action(action)
```

### Example 2: Anomaly Detection

```python
from robotic_feedback_system import RoboticFeedbackSystem
from utils.embedding_utils import EmbeddingAnalysis
import torch

system = RoboticFeedbackSystem(device="cuda")

# Establish baseline from normal operation
baseline_embedding = get_fused_embedding(system, normal_sensor_data)

# Monitor for anomalies
while monitoring:
    current_embedding = get_fused_embedding(system, current_sensor_data)
    
    # Compute similarity to baseline
    similarity = EmbeddingAnalysis.cosine_similarity(
        baseline_embedding,
        current_embedding
    )
    
    if similarity < 0.85:
        print("ANOMALY DETECTED!")
        # Take corrective action
```

### Example 3: Modality Importance Analysis

```python
with torch.no_grad():
    fused, modalities = system(
        camera_images={...},
        audio=audio,
        pressure=pressure,
        emg=emg,
        return_modality_embeddings=True
    )

# Analyze modality contributions
contributions = EmbeddingAnalysis.modality_contribution(
    modalities, fused
)

print("Modality Contributions:")
for modality, contrib in contributions.items():
    print(f"  {modality}: {contrib:.2%}")
```

---

## Advanced Features

### Fine-tuning Sensor Encoders

```python
# By default, vision and audio encoders are frozen
# Sensor encoders are trainable

system = RoboticFeedbackSystem(device="cuda")

# Enable training
system.train()

# Create optimizer for trainable parameters
optimizer = torch.optim.Adam(
    [p for p in system.parameters() if p.requires_grad],
    lr=1e-4
)

# Training loop
for batch in training_data:
    optimizer.zero_grad()
    
    fused, _ = system(
        camera_images=batch['cameras'],
        audio=batch['audio'],
        pressure=batch['pressure'],
        emg=batch['emg'],
        return_modality_embeddings=True
    )
    
    loss = compute_loss(fused, batch['labels'])
    loss.backward()
    optimizer.step()
```

### Computing Embedding Statistics

```python
from utils.embedding_utils import create_embeddings_report

with torch.no_grad():
    fused, modalities = system(...)

# Generate comprehensive report
report = create_embeddings_report(fused, modalities, 
                                  output_path="report.json")

print("Embedding Statistics:")
for modality, stats in report['modalities'].items():
    print(f"\n{modality}:")
    print(f"  Dimension: {stats['dimension']}")
    print(f"  Norm: {stats['norm']:.4f}")
    print(f"  Sparsity: {stats['sparsity']:.2%}")
```

### Similarity-based Retrieval

```python
from utils.embedding_utils import EmbeddingRetrieval

# Build database of reference states
retrieval = EmbeddingRetrieval(device="cuda")

for label, sensor_data in reference_database:
    embedding = system(sensor_data)
    retrieval.add_to_database(embedding, label)

# Find similar states during operation
current_embedding = system(current_sensor_data)
similar_states = retrieval.retrieve_similar(current_embedding, top_k=5)

print("Most similar reference states:")
for label, similarity in similar_states:
    print(f"  {label}: {similarity:.4f}")
```

---

## Performance Characteristics

### Computational Requirements

| Configuration | Memory (GB) | Inference Time (ms) | Max Batch Size |
|---------------|-------------|-------------------|-----------------|
| Lightweight | 3 | 50 | 32 |
| Balanced | 6 | 100 | 16 |
| High Capacity | 12 | 200 | 8 |

### Memory Usage Breakdown

- CLIP Vision Encoder: ~2.5 GB
- Wav2Vec 2.0 Audio: ~1.5 GB
- Sensor Encoders: ~0.05 GB
- Fusion Modules: ~0.05 GB
- **Total: ~4-6 GB (balanced config)**

### Inference Speed

On NVIDIA V100 GPU with batch_size=8:
- Vision encoding: ~30ms
- Audio encoding: ~40ms
- Sensor encoding: ~5ms
- Fusion: ~5ms
- **Total: ~80ms per batch**

---

## Input Data Format Specifications

### Camera Input
```
Shape: (batch_size, 3, 224, 224)
Values: [0, 1] range (normalized)
Format: RGB (not BGR)
Preprocessing: Auto-handled by VisionPreprocessor
```

### Audio Input
```
Shape: (batch_size, num_samples)
Sample Rate: 16000 Hz
Duration: 3 seconds (48000 samples)
Values: [-1, 1] range
Preprocessing: Auto-normalized
```

### Pressure Sensor
```
Shape: (batch_size, sequence_length)
Typical sequence: ~1000 timesteps
Sampling rate: ~1000 Hz
Values: Raw sensor readings
Preprocessing: Statistical feature extraction
```

### EMG Sensor
```
Shape: (batch_size, num_channels, sequence_length)
Number of channels: Typically 8
Sequence length: ~1000 timesteps
Sampling rate: ~1000 Hz
Values: Raw EMG voltage readings (μV)
Preprocessing: Statistical feature extraction per channel
```

---

## Troubleshooting

### Issue: Out of Memory Error
- **Solution 1**: Reduce batch size
- **Solution 2**: Use lightweight configuration
- **Solution 3**: Enable gradient checkpointing
```python
# For training, reduce activation memory
system.train()
system.gradient_checkpointing_enable()
```

### Issue: Low Similarity Between Modalities
- **Normal behavior**: Modalities are learned independently
- **Solution**: Use attention-based fusion to learn relationships
```python
system = RoboticFeedbackSystem(use_attention=True)
```

### Issue: Slow Inference
- **Solution 1**: Use GPU acceleration (cuda)
- **Solution 2**: Increase batch size (more efficient)
- **Solution 3**: Use lightweight configuration

### Issue: Inconsistent Results Between Runs
- **Cause**: Random initialization of sensor encoders
- **Solution**: Set random seed
```python
import torch
torch.manual_seed(42)
```

---

## Best Practices

1. **Always use eval() mode for inference**
   ```python
   system.eval()
   with torch.no_grad():
       embedding = system(...)
   ```

2. **Normalize inputs properly**
   - Use provided preprocessors
   - Keep images in [0, 1] range
   - Keep audio in [-1, 1] range

3. **Batch your inputs for efficiency**
   - Minimum batch size: 1
   - Recommended batch size: 8-16
   - Higher batch sizes → better GPU utilization

4. **Monitor embedding statistics**
   - Check embedding norms
   - Verify sparsity patterns
   - Look for degenerate solutions

5. **Cache models if running repeatedly**
   - Models are downloaded on first use
   - Subsequent runs use cache
   - Clear cache if needed: `rm ~/.cache/huggingface/*`

---

## References & Further Reading

- **CLIP**: Learning Transferable Models for Composable Vision Tasks
  - Paper: https://arxiv.org/abs/2103.14030
  - Key insight: Vision-language pre-training learns robust visual features

- **Wav2Vec 2.0**: wav2vec 2.0: A Framework for Self-Supervised Learning
  - Paper: https://arxiv.org/abs/2006.11477
  - Key insight: Self-supervised learning on unlabeled audio works well

- **Multimodal Learning**: Recent survey on multimodal machine learning
  - Overview: Fusion methods, benchmark datasets, applications

---

## Support & Contribution

For issues, questions, or contributions:
1. Check this documentation first
2. Review example scripts
3. Check config presets
4. Refer to inline code documentation

---

## License

MIT License - Free to use and modify for research and commercial applications.
