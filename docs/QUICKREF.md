# Quick Reference Card

## Initialization

```python
from robotic_feedback_system import RoboticFeedbackSystem
from config import get_config

# Simple initialization
system = RoboticFeedbackSystem(device="cuda")
system.eval()

# With preset config
config = get_config("balanced")
system = RoboticFeedbackSystem(
    fusion_dim=config.fusion.fusion_dim,
    fusion_method=config.fusion.fusion_method,
    use_attention=config.fusion.use_attention,
    device=config.device
)
```

## Input Preparation

```python
import torch

# Camera inputs (batch=1, 3 channels, 224x224)
camera1 = torch.randn(1, 3, 224, 224)  # [0, 1]
camera2 = torch.randn(1, 3, 224, 224)  # [0, 1]

# Audio (batch=1, 16kHz, 3 seconds = 48000 samples)
audio = torch.randn(1, 48000)           # [-1, 1]

# Pressure sensor (batch=1, 1000 timesteps)
pressure = torch.randn(1, 1000)

# EMG (batch=1, 8 channels, 1000 timesteps)
emg = torch.randn(1, 8, 1000)
```

## Get Fused Embedding

```python
# Basic inference
with torch.no_grad():
    fused = system(
        camera_images={'camera1': camera1, 'camera2': camera2},
        audio=audio,
        pressure=pressure,
        emg=emg
    )
# Output shape: (batch_size, 512)

# With individual modality embeddings
with torch.no_grad():
    fused, modalities = system(
        camera_images={'camera1': camera1, 'camera2': camera2},
        audio=audio,
        pressure=pressure,
        emg=emg,
        return_modality_embeddings=True
    )
# modalities = {'vision': ..., 'audio': ..., 'pressure': ..., 'emg': ...}
```

## Individual Encoders

```python
# Vision only (average of both cameras)
vision_emb = system.encode_vision_only({'camera1': cam1, 'camera2': cam2})
# Output: (batch_size, 512)

# Audio only
audio_emb = system.encode_audio_only(audio)
# Output: (batch_size, 768)

# Pressure only
pressure_emb = system.encode_pressure_only(pressure)
# Output: (batch_size, 256)

# EMG only
emg_emb = system.encode_emg_only(emg)
# Output: (batch_size, 256)
```

## Analysis & Utilities

```python
from utils.embedding_utils import EmbeddingAnalysis

# Cosine similarity
sim = EmbeddingAnalysis.cosine_similarity(emb1, emb2)
# Returns: float [0, 1]

# Euclidean distance
dist = EmbeddingAnalysis.euclidean_distance(emb1, emb2)
# Returns: float

# Modality contribution to fused embedding
contrib = EmbeddingAnalysis.modality_contribution(modality_embeddings, fused)
# Returns: {'vision': 0.25, 'audio': 0.30, 'pressure': 0.20, 'emg': 0.25}

# Statistics
stats = EmbeddingAnalysis.embedding_statistics(embedding)
# Returns: {dimension, norm, mean, std, min, max, sparsity}
```

## Preprocessing

```python
from preprocessing.preprocessor import (
    VisionPreprocessor, AudioPreprocessor, SensorPreprocessor
)

# Vision
vision_prep = VisionPreprocessor(image_size=(224, 224))
processed_img = vision_prep.preprocess(img)  # PIL Image → tensor

# Audio
audio_prep = AudioPreprocessor(sample_rate=16000, duration=3.0)
processed_audio = audio_prep.preprocess(audio)  # array → tensor

# Sensors
sensor_prep = SensorPreprocessor(normalize=True, standardize=True)
processed_sensor = sensor_prep.preprocess(sensor_data)
```

## System Info

```python
# Get all dimensions
dims = system.get_modality_dimensions()
# Returns: {
#   'vision': 512,
#   'audio': 768,
#   'pressure': 256,
#   'emg': 256,
#   'fused': 512
# }

# Freeze encoders (prevent fine-tuning pre-trained models)
system.freeze_encoders()

# Unfreeze for fine-tuning
system.unfreeze_encoders()
```

## Configuration Presets

```python
from config import get_config

# Lightweight (resource-constrained)
config = get_config("lightweight")

# Balanced (recommended)
config = get_config("balanced")

# High capacity (complex tasks)
config = get_config("high_capacity")

# Access config values
print(config.fusion.fusion_dim)          # 512
print(config.fusion.fusion_method)       # "concat_project"
print(config.device)                     # "cuda"
```

## Batch Processing

```python
batch_size = 8

# Stack individual inputs into batches
batch_cameras = {
    'camera1': torch.stack([img1, img2, ...]),  # (8, 3, 224, 224)
    'camera2': torch.stack([img1, img2, ...])
}
batch_audio = torch.stack([audio1, audio2, ...])  # (8, 48000)
batch_pressure = torch.stack([p1, p2, ...])      # (8, 1000)
batch_emg = torch.stack([e1, e2, ...])           # (8, 8, 1000)

# Process batch
with torch.no_grad():
    fused_batch = system(
        camera_images=batch_cameras,
        audio=batch_audio,
        pressure=batch_pressure,
        emg=batch_emg
    )
# Output: (8, 512)
```

## Common Patterns

### Real-time Robot Control Loop
```python
system.eval()
while robot_running:
    # Read sensors
    img1, img2 = camera.read()
    audio = mic.read(duration=3)
    pressure = pressure_sensor.read()
    emg = emg_sensors.read()
    
    # Get embedding
    with torch.no_grad():
        embedding = system(
            camera_images={'camera1': img1.unsqueeze(0), 'camera2': img2.unsqueeze(0)},
            audio=audio.unsqueeze(0),
            pressure=pressure.unsqueeze(0),
            emg=emg.unsqueeze(0)
        )
    
    # Control
    action = policy(embedding)
    robot.execute(action)
```

### Anomaly Detection
```python
# Baseline
baseline = system(normal_data)

# Monitor
while monitoring:
    current = system(sensor_data)
    similarity = F.cosine_similarity(baseline, current)
    
    if similarity < threshold:
        print("Anomaly!")
```

### Batch Inference
```python
embeddings_list = []
for batch in dataloader:
    with torch.no_grad():
        batch_embeddings = system(batch['cameras'], batch['audio'], 
                                  batch['pressure'], batch['emg'])
    embeddings_list.append(batch_embeddings)

all_embeddings = torch.cat(embeddings_list, dim=0)
```

## Data Format Summary

| Modality | Shape | Range | Rate | Notes |
|----------|-------|-------|------|-------|
| Camera 1 | (B, 3, 224, 224) | [0, 1] | N/A | RGB image |
| Camera 2 | (B, 3, 224, 224) | [0, 1] | N/A | RGB image |
| Audio | (B, 48000) | [-1, 1] | 16 kHz | 3 seconds |
| Pressure | (B, 1000) | Any | 1 kHz | Varies per sensor |
| EMG | (B, 8, 1000) | Any | 1 kHz | 8 channels |

## Output Dimensions

| Encoder | Output Dimension |
|---------|-----------------|
| Vision (CLIP) | 512 |
| Audio (Wav2Vec) | 768 |
| Pressure | 256 |
| EMG | 256 |
| **Fused (Default)** | **512** |

## Typical Use Cases

- **Control**: Use embedding as policy input
- **Anomaly Detection**: Monitor similarity to baseline
- **Classification**: Use embedding for downstream classifier
- **Retrieval**: Find similar states in database
- **Analysis**: Study modality contributions
- **Visualization**: Reduce to 2D for inspection

## Device Management

```python
import torch

# Check GPU availability
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())

# Use GPU
system.to("cuda")
embeddings = system(inputs)  # Automatic GPU processing

# Use CPU (fallback)
system.to("cpu")
embeddings = system(inputs.cpu())

# Move data to same device
embeddings = embeddings.to(system.device)
```

## Common Issues

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch size, use lightweight config |
| Slow inference | Use GPU, increase batch size |
| Model not loading | Check internet (for downloading), clear cache |
| Inconsistent results | Use eval() mode, set random seed |
| High memory usage | Use lightweight preset |

---

**For detailed information, see README.md and GUIDE.md**
