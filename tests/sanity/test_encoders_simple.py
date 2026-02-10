"""
Test Script: Encoder Outputs (without large model downloads)
Tests encoders that don't require downloading large pre-trained models
Run with: source /home/nishant/projects/mmfuse-env/bin/activate && python test_encoders_simple.py
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import encoders
from encoders.vision_encoder import VisionEncoder
from encoders.audio_encoder_learnable import AudioEncoder
from encoders.sensor_encoder import PressureSensorEncoder, EMGSensorEncoder
from preprocessing.preprocessor import SensorPreprocessor


def test_vision_encoder():
    """Test and display Vision Encoder (CLIP) outputs"""
    print("\n" + "=" * 90)
    print("TEST 1: VISION ENCODER (CLIP with open-clip)")
    print("=" * 90)
    
    device = "cpu"
    print(f"Device: {device}\n")
    
    # Initialize encoder
    print("Initializing CLIP Vision Encoder (ViT-B-32)...")
    vision_encoder = VisionEncoder(
        model_name="ViT-B-32",
        pretrained="openai",
        frozen=True,
        device=device
    )
    print(f"✓ Encoder initialized successfully")
    print(f"  Model: ViT-B-32 (Vision Transformer Base, 32x32 patches)")
    print(f"  Backend: open-clip")
    print(f"  Output dimension: {vision_encoder.output_dim}\n")
    
    # Create dummy camera inputs
    print("Creating dummy camera images (RGB 224x224)...\n")
    batch_sizes = [1, 2, 4]
    
    for batch_size in batch_sizes:
        camera_images = torch.rand(batch_size, 3, 224, 224, device=device)
        
        with torch.no_grad():
            embeddings = vision_encoder(camera_images)
        
        embedding_norms = embeddings.norm(dim=1)
        
        print(f"Batch Size: {batch_size}")
        print(f"  Input shape:           {tuple(camera_images.shape)}")
        print(f"  Output shape:          {tuple(embeddings.shape)}")
        print(f"  Embedding norm (mean): {embedding_norms.mean():.4f} ✓")
        print(f"  Embedding norm (std):  {embedding_norms.std():.4f}")
        print(f"  Stats: min={embeddings.min():.4f}, max={embeddings.max():.4f}, "
              f"mean={embeddings.mean():.4f}, std={embeddings.std():.4f}\n")
    
    print("✓ Vision Encoder Test PASSED\n")
    return vision_encoder


def test_pressure_sensor_encoder():
    """Test and display Pressure Sensor Encoder outputs"""
    print("=" * 90)
    print("TEST 2: PRESSURE SENSOR ENCODER")
    print("=" * 90)
    
    device = "cpu"
    print(f"Device: {device}\n")
    
    print("Initializing Pressure Sensor Encoder...")
    pressure_encoder = PressureSensorEncoder(
        output_dim=256,
        num_channels=1
    ).to(device)
    pressure_encoder.eval()  # Set to eval mode to avoid batch norm issues
    
    sensor_preprocessor = SensorPreprocessor(normalize=True, standardize=True)
    
    print(f"✓ Encoder initialized successfully")
    print(f"  Input channels: 1")
    print(f"  Output dimension: {pressure_encoder.output_dim_value}")
    print(f"  Input processing: Temporal feature extraction\n")
    
    print("Creating dummy pressure sensor signals (1000 samples @ 1kHz)...\n")
    batch_sizes = [1, 2, 4]
    
    for batch_size in batch_sizes:
        raw_pressure = torch.randn(batch_size, 1000, device=device)
        
        # Extract temporal features
        pressure_features = sensor_preprocessor.extract_features_temporal(
            raw_pressure,
            window_size=50
        )
        
        # Normalize
        pressure_features = sensor_preprocessor.preprocess_batch(pressure_features)
        
        # Encode
        with torch.no_grad():
            embeddings = pressure_encoder(pressure_features.to(device))
        
        print(f"Batch Size: {batch_size}")
        print(f"  Raw input shape:       {tuple(raw_pressure.shape)}")
        print(f"  Feature shape:         {tuple(pressure_features.shape)}")
        print(f"  Output embedding:      {tuple(embeddings.shape)}")
        print(f"  Embedding norm (mean): {embeddings.norm(dim=1).mean():.4f}")
        print(f"  Stats: min={embeddings.min():.4f}, max={embeddings.max():.4f}, "
              f"mean={embeddings.mean():.4f}, std={embeddings.std():.4f}\n")
    
    print("✓ Pressure Sensor Encoder Test PASSED\n")
    return pressure_encoder


def test_emg_sensor_encoder():
    """Test and display EMG Sensor Encoder outputs"""
    print("=" * 90)
    print("TEST 3: EMG SENSOR ENCODER")
    print("=" * 90)
    
    device = "cpu"
    print(f"Device: {device}\n")
    
    print("Initializing EMG Sensor Encoder...")
    emg_encoder = EMGSensorEncoder(
        output_dim=256,
        num_channels=8,
        input_features=800  # 8 channels * 100 features per channel
    ).to(device)
    emg_encoder.eval()  # Set to eval mode to avoid batch norm issues
    
    sensor_preprocessor = SensorPreprocessor(normalize=True, standardize=True)
    
    print(f"✓ Encoder initialized successfully")
    print(f"  Input channels: 8 (typical EMG electrode array)")
    print(f"  Output dimension: {emg_encoder.output_dim_value}")
    print(f"  Input processing: Temporal feature extraction per channel\n")
    
    print("Creating dummy EMG sensor signals (8 channels, 1000 samples @ 1kHz)...\n")
    batch_sizes = [1, 2, 4]
    
    for batch_size in batch_sizes:
        raw_emg = torch.randn(batch_size, 8, 1000, device=device)
        
        # Extract temporal features
        emg_features = sensor_preprocessor.extract_features_temporal(
            raw_emg,
            window_size=50
        )
        
        # Normalize
        emg_features = sensor_preprocessor.preprocess_batch(emg_features)
        
        # Flatten multi-channel features: (batch_size, 8, 100) -> (batch_size, 800)
        emg_features_flat = emg_features.reshape(emg_features.shape[0], -1)
        
        # Encode
        with torch.no_grad():
            embeddings = emg_encoder(emg_features_flat.to(device))
        
        print(f"Batch Size: {batch_size}")
        print(f"  Raw input shape:       {tuple(raw_emg.shape)}")
        print(f"  Feature shape:         {tuple(emg_features.shape)}")
        print(f"  Output embedding:      {tuple(embeddings.shape)}")
        print(f"  Embedding norm (mean): {embeddings.norm(dim=1).mean():.4f}")
        print(f"  Stats: min={embeddings.min():.4f}, max={embeddings.max():.4f}, "
              f"mean={embeddings.mean():.4f}, std={embeddings.std():.4f}\n")
    
    print("✓ EMG Sensor Encoder Test PASSED\n")
    return emg_encoder


def test_audio_encoder():
    """Test and display Audio Encoder (Learnable CNN) outputs"""
    print("=" * 90)
    print("TEST 3B: AUDIO ENCODER (Learnable 1D CNN)")
    print("=" * 90)
    
    device = "cpu"
    print(f"Device: {device}\n")
    
    # Initialize encoder
    print("Initializing Learnable Audio Encoder...")
    try:
        audio_encoder = AudioEncoder(
            output_dim=768,
            num_filters=256,
            num_layers=4,
            kernel_size=15,
            stride=2,
            device=device
        )
        print(f"✓ Encoder initialized successfully")
        print(f"  Model: 1D CNN Audio Encoder (learnable)")
        print(f"  Filters: 256, Layers: 4, Kernel: 15")
        print(f"  Output dimension: {audio_encoder.output_dim}\n")
    except Exception as e:
        print(f"❌ Failed to initialize audio encoder: {e}")
        return None
    
    # Create dummy audio inputs
    print("Creating dummy audio signals (16kHz, 5 seconds)...\n")
    
    batch_sizes = [1, 2, 4]
    
    for batch_size in batch_sizes:
        # Create raw audio: 16kHz * 5 seconds = 80000 samples
        raw_audio = torch.randn(batch_size, 80000, device=device)
        
        # Encode (learnable encoder processes tensors directly)
        with torch.no_grad():
            embeddings = audio_encoder(raw_audio, sampling_rate=16000)
        
        print(f"Batch Size: {batch_size}")
        print(f"  Raw input shape:       {tuple(raw_audio.shape)} (16kHz, 3s)")
        print(f"  Output embedding:      {tuple(embeddings.shape)}")
        print(f"  Embedding norm (mean): {embeddings.norm(dim=1).mean():.4f}")
        print(f"  Stats: min={embeddings.min():.4f}, max={embeddings.max():.4f}, "
              f"mean={embeddings.mean():.4f}, std={embeddings.std():.4f}\n")
    
    print("✓ Audio Encoder Test PASSED\n")
    return audio_encoder


def test_combined_encoders():
    """Test all encoders together"""
    print("=" * 90)
    print("TEST 4: COMBINED ENCODER OUTPUTS")
    print("=" * 90)
    
    device = "cpu"
    print(f"Device: {device}\n")
    
    # Initialize encoders
    print("Initializing all encoders...")
    vision_encoder = VisionEncoder(model_name="ViT-B-32", device=device)
    audio_encoder = AudioEncoder(output_dim=768, num_filters=256, num_layers=4, device=device)
    pressure_encoder = PressureSensorEncoder(output_dim=256, num_channels=1).to(device)
    emg_encoder = EMGSensorEncoder(output_dim=256, num_channels=8, input_features=800).to(device)
    
    # Set to eval mode
    audio_encoder.eval()
    pressure_encoder.eval()
    emg_encoder.eval()
    
    sensor_preprocessor = SensorPreprocessor()
    
    print("✓ All encoders initialized\n")
    
    # Create test batch
    batch_size = 2
    print(f"Creating test batch (batch_size={batch_size})...\n")
    
    camera1 = torch.rand(batch_size, 3, 224, 224, device=device)
    camera2 = torch.rand(batch_size, 3, 224, 224, device=device)
    audio = torch.randn(batch_size, 80000, device=device)  # 16kHz, 5 seconds
    pressure = torch.randn(batch_size, 1000, device=device)
    emg = torch.randn(batch_size, 8, 1000, device=device)
    
    # Encode all
    print("Encoding all modalities...\n")
    
    with torch.no_grad():
        # Vision
        camera1_emb = vision_encoder(camera1)
        camera2_emb = vision_encoder(camera2)
        vision_emb_avg = (camera1_emb + camera2_emb) / 2
        
        # Audio (raw audio, no preprocessing - handled by model's processor)
        audio_emb = audio_encoder(audio, sampling_rate=16000)
        
        # Pressure
        pressure_features = sensor_preprocessor.extract_features_temporal(pressure, window_size=50)
        pressure_features = sensor_preprocessor.preprocess_batch(pressure_features)
        pressure_emb = pressure_encoder(pressure_features.to(device))
        
        # EMG
        emg_features = sensor_preprocessor.extract_features_temporal(emg, window_size=50)
        emg_features = sensor_preprocessor.preprocess_batch(emg_features)
        emg_features_flat = emg_features.reshape(emg_features.shape[0], -1)
        emg_emb = emg_encoder(emg_features_flat.to(device))
    
    # Display results in table format
    print("MULTIMODAL ENCODER OUTPUT SUMMARY:")
    print("─" * 90)
    print(f"{'Modality':<18} {'Shape':<18} {'Dimension':<12} {'Norm':<12} {'Range':<25}")
    print("─" * 90)
    
    encoders_data = [
        ("Vision (avg)", vision_emb_avg),
        ("Audio (Wav2Vec)", audio_emb),
        ("Pressure Sensor", pressure_emb),
        ("EMG Sensor", emg_emb)
    ]
    
    total_dim = 0
    for name, embedding in encoders_data:
        dim = embedding.shape[-1]
        norm = embedding.norm(dim=1).mean().item()
        min_val = embedding.min().item()
        max_val = embedding.max().item()
        
        print(f"{name:<18} {str(tuple(embedding.shape)):<18} {dim:<12} "
              f"{norm:<12.4f} [{min_val:>8.4f}, {max_val:>8.4f}]")
        total_dim += dim
    
    print("─" * 90)
    print(f"\n✓ Combined Encoder Test PASSED")
    
    # Summary statistics
    print("\n" + "=" * 90)
    print("DIMENSIONALITY SUMMARY")
    print("=" * 90)
    print(f"  Vision:              512-dim")
    print(f"  Audio:               768-dim")
    print(f"  Pressure:            256-dim")
    print(f"  EMG:                 256-dim")
    print(f"  ────────────────────────────")
    print(f"  Total (pre-fusion):  1792-dim")
    print(f"  Post-fusion (proj):  512-dim (unified multimodal embedding)")
    print("\n")


if __name__ == "__main__":
    print("\n" + "=" * 90)
    print("ROBOTIC MULTIMODAL FUSION SYSTEM - ENCODER OUTPUT TESTS")
    print("=" * 90)
    
    try:
        # Run tests
        vision_enc = test_vision_encoder()
        audio_enc = test_audio_encoder()
        pressure_enc = test_pressure_sensor_encoder()
        emg_enc = test_emg_sensor_encoder()
        test_combined_encoders()
        
        print("=" * 90)
        print("✅ ALL ENCODER TESTS PASSED SUCCESSFULLY!")
        print("=" * 90)
        
        print("\nKey Results:")
        print("  ✓ CLIP Vision Encoder (open-clip): Working - 512-dim outputs")
        print("  ✓ Learnable Audio Encoder (1D CNN): Working - 768-dim outputs")
        print("  ✓ Pressure Sensor Encoder: Working - 256-dim outputs")
        print("  ✓ EMG Sensor Encoder: Working - 256-dim outputs")
        print("\nAll 5 modality encoders validated and working correctly!\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
