"""
Test Script: Individual Encoder Outputs
Tests each encoder separately and displays output dimensions and characteristics
Run with: source /home/nishant/projects/mmfuse-env/bin/activate && python test_encoders.py
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import encoders
from encoders.vision_encoder import VisionEncoder
from encoders.audio_encoder import AudioEncoder
from encoders.sensor_encoder import PressureSensorEncoder, EMGSensorEncoder
from preprocessing.preprocessor import VisionPreprocessor, AudioPreprocessor, SensorPreprocessor


def test_vision_encoder():
    """Test and display Vision Encoder (CLIP) outputs"""
    print("\n" + "=" * 80)
    print("TEST 1: VISION ENCODER (CLIP with open-clip)")
    print("=" * 80)
    
    device = "cpu"  # Using CPU for this setup
    print(f"Device: {device}\n")
    
    # Initialize encoder
    print("Initializing CLIP Vision Encoder...")
    vision_encoder = VisionEncoder(
        model_name="ViT-B-32",
        pretrained="openai",
        frozen=True,
        device=device
    )
    print(f"✓ Encoder initialized")
    print(f"✓ Model name: ViT-B-32 (open-clip)")
    print(f"✓ Output dimension: {vision_encoder.output_dim}\n")
    
    # Create dummy camera inputs
    print("Creating dummy camera images...")
    batch_sizes = [1, 2, 4, 8]
    
    for batch_size in batch_sizes:
        # Create random RGB images (H, W, C)
        camera_images = torch.rand(batch_size, 3, 224, 224, device=device)
        
        # Encode
        with torch.no_grad():
            embeddings = vision_encoder(camera_images)
        
        print(f"\nBatch Size: {batch_size}")
        print(f"  Input shape:     {tuple(camera_images.shape)}")
        print(f"  Output shape:    {tuple(embeddings.shape)}")
        print(f"  Embedding norm:  {embeddings.norm(dim=1).mean():.4f} (mean)")
        print(f"  Embedding stats: min={embeddings.min():.4f}, max={embeddings.max():.4f}, "
              f"mean={embeddings.mean():.4f}, std={embeddings.std():.4f}")
        
        # Check normalization (should be normalized)
        embedding_norms = embeddings.norm(dim=1)
        print(f"  Embedding norms: {embedding_norms[0]:.4f} (should be ~1.0)")
    
    print("\n✓ Vision Encoder Test PASSED")


def test_audio_encoder():
    """Test and display Audio Encoder (Wav2Vec 2.0) outputs"""
    print("\n" + "=" * 80)
    print("TEST 2: AUDIO ENCODER (Wav2Vec 2.0)")
    print("=" * 80)
    
    device = "cpu"
    print(f"Device: {device}\n")
    
    # Initialize encoder
    print("Initializing Wav2Vec 2.0 Audio Encoder...")
    audio_encoder = AudioEncoder(
        model_name="facebook/wav2vec2-base",
        frozen=True,
        device=device,
        sampling_rate=16000
    )
    print(f"✓ Encoder initialized")
    print(f"✓ Model name: facebook/wav2vec2-base")
    print(f"✓ Output dimension (before pooling): {audio_encoder.output_dim}")
    print(f"✓ Sampling rate: 16000 Hz\n")
    
    # Create dummy audio inputs
    print("Creating dummy audio signals (16kHz, 3 seconds)...")
    batch_sizes = [1, 2, 4, 8]
    
    for batch_size in batch_sizes:
        # Create random audio (16kHz * 3 seconds = 48000 samples)
        audio = torch.randn(batch_size, 48000, device=device)
        
        # Normalize to [-1, 1]
        audio = audio / (audio.abs().max() + 1e-6)
        
        # Encode without pooling (get sequence)
        with torch.no_grad():
            embeddings_seq = audio_encoder.forward(audio)
            embeddings_pooled = audio_encoder.encode(audio, pooling="mean")
        
        print(f"\nBatch Size: {batch_size}")
        print(f"  Input shape:              {tuple(audio.shape)}")
        print(f"  Sequence embeddings:      {tuple(embeddings_seq.shape)}")
        print(f"  Pooled embeddings:        {tuple(embeddings_pooled.shape)}")
        print(f"  Embedding norm (pooled):  {embeddings_pooled.norm(dim=1).mean():.4f} (mean)")
        print(f"  Embedding stats: min={embeddings_pooled.min():.4f}, "
              f"max={embeddings_pooled.max():.4f}, mean={embeddings_pooled.mean():.4f}, "
              f"std={embeddings_pooled.std():.4f}")
    
    print("\n✓ Audio Encoder Test PASSED")


def test_pressure_sensor_encoder():
    """Test and display Pressure Sensor Encoder outputs"""
    print("\n" + "=" * 80)
    print("TEST 3: PRESSURE SENSOR ENCODER")
    print("=" * 80)
    
    device = "cpu"
    print(f"Device: {device}\n")
    
    # Initialize encoder and preprocessor
    print("Initializing Pressure Sensor Encoder...")
    pressure_encoder = PressureSensorEncoder(
        output_dim=256,
        num_channels=1
    ).to(device)
    
    sensor_preprocessor = SensorPreprocessor(normalize=True, standardize=True)
    
    print(f"✓ Encoder initialized")
    print(f"✓ Number of channels: 1")
    print(f"✓ Output dimension: {pressure_encoder.output_dim_value}\n")
    
    # Test with dummy pressure data
    print("Creating dummy pressure sensor signals (1000 samples @ 1kHz)...")
    batch_sizes = [1, 2, 4, 8]
    
    for batch_size in batch_sizes:
        # Create raw pressure readings
        raw_pressure = torch.randn(batch_size, 1000, device=device)
        
        # Extract features
        pressure_features = sensor_preprocessor.extract_features_temporal(
            raw_pressure,
            window_size=50
        )
        
        # Normalize features
        pressure_features = sensor_preprocessor.preprocess_batch(pressure_features)
        
        # Encode
        with torch.no_grad():
            embeddings = pressure_encoder(pressure_features.to(device))
        
        print(f"\nBatch Size: {batch_size}")
        print(f"  Raw input shape:         {tuple(raw_pressure.shape)}")
        print(f"  Extracted features:      {tuple(pressure_features.shape)}")
        print(f"  Output embeddings:       {tuple(embeddings.shape)}")
        print(f"  Embedding norm:          {embeddings.norm(dim=1).mean():.4f} (mean)")
        print(f"  Embedding stats: min={embeddings.min():.4f}, max={embeddings.max():.4f}, "
              f"mean={embeddings.mean():.4f}, std={embeddings.std():.4f}")
    
    print("\n✓ Pressure Sensor Encoder Test PASSED")


def test_emg_sensor_encoder():
    """Test and display EMG Sensor Encoder outputs"""
    print("\n" + "=" * 80)
    print("TEST 4: EMG SENSOR ENCODER")
    print("=" * 80)
    
    device = "cpu"
    print(f"Device: {device}\n")
    
    # Initialize encoder and preprocessor
    print("Initializing EMG Sensor Encoder...")
    emg_encoder = EMGSensorEncoder(
        output_dim=256,
        num_channels=8
    ).to(device)
    
    sensor_preprocessor = SensorPreprocessor(normalize=True, standardize=True)
    
    print(f"✓ Encoder initialized")
    print(f"✓ Number of channels: 8")
    print(f"✓ Output dimension: {emg_encoder.output_dim_value}\n")
    
    # Test with dummy EMG data
    print("Creating dummy EMG sensor signals (8 channels, 1000 samples @ 1kHz)...")
    batch_sizes = [1, 2, 4, 8]
    
    for batch_size in batch_sizes:
        # Create raw EMG readings (8 channels)
        raw_emg = torch.randn(batch_size, 8, 1000, device=device)
        
        # Extract features
        emg_features = sensor_preprocessor.extract_features_temporal(
            raw_emg,
            window_size=50
        )
        
        # Normalize features
        emg_features = sensor_preprocessor.preprocess_batch(emg_features)
        
        # Encode
        with torch.no_grad():
            embeddings = emg_encoder(emg_features.to(device))
        
        print(f"\nBatch Size: {batch_size}")
        print(f"  Raw input shape:         {tuple(raw_emg.shape)}")
        print(f"  Extracted features:      {tuple(emg_features.shape)}")
        print(f"  Output embeddings:       {tuple(embeddings.shape)}")
        print(f"  Embedding norm:          {embeddings.norm(dim=1).mean():.4f} (mean)")
        print(f"  Embedding stats: min={embeddings.min():.4f}, max={embeddings.max():.4f}, "
              f"mean={embeddings.mean():.4f}, std={embeddings.std():.4f}")
    
    print("\n✓ EMG Sensor Encoder Test PASSED")


def test_all_encoders_together():
    """Test all encoders together and compare outputs"""
    print("\n" + "=" * 80)
    print("TEST 5: ALL ENCODERS TOGETHER (Comparison)")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # Initialize all encoders
    print("Initializing all encoders...")
    vision_encoder = VisionEncoder(device=device)
    audio_encoder = AudioEncoder(device=device)
    pressure_encoder = PressureSensorEncoder(output_dim=256, num_channels=1).to(device)
    emg_encoder = EMGSensorEncoder(output_dim=256, num_channels=8).to(device)
    sensor_preprocessor = SensorPreprocessor()
    
    print("✓ All encoders initialized\n")
    
    # Create test data
    batch_size = 2
    print(f"Creating test batch (batch_size={batch_size})...")
    
    camera1 = torch.rand(batch_size, 3, 224, 224, device=device)
    camera2 = torch.rand(batch_size, 3, 224, 224, device=device)
    audio = torch.randn(batch_size, 48000, device=device)
    audio = audio / (audio.abs().max() + 1e-6)
    pressure = torch.randn(batch_size, 1000, device=device)
    emg = torch.randn(batch_size, 8, 1000, device=device)
    
    print("✓ Test data created\n")
    
    # Encode all
    print("Encoding all modalities...\n")
    
    with torch.no_grad():
        vision_emb = vision_encoder(camera1)
        camera2_emb = vision_encoder(camera2)
        vision_emb_avg = (vision_emb + camera2_emb) / 2
        
        audio_emb = audio_encoder.encode(audio, pooling="mean")
        
        pressure_features = sensor_preprocessor.extract_features_temporal(pressure, window_size=50)
        pressure_features = sensor_preprocessor.preprocess_batch(pressure_features)
        pressure_emb = pressure_encoder(pressure_features.to(device))
        
        emg_features = sensor_preprocessor.extract_features_temporal(emg, window_size=50)
        emg_features = sensor_preprocessor.preprocess_batch(emg_features)
        emg_emb = emg_encoder(emg_features.to(device))
    
    # Display results
    print("ENCODER OUTPUTS:")
    print("─" * 80)
    
    encoders_info = [
        ("Vision (Camera avg)", vision_emb_avg),
        ("Audio", audio_emb),
        ("Pressure", pressure_emb),
        ("EMG", emg_emb)
    ]
    
    print(f"{'Modality':<20} {'Shape':<20} {'Norm':<15} {'Min':<10} {'Max':<10} {'Sparsity':<10}")
    print("─" * 80)
    
    for modality_name, embedding in encoders_info:
        norm_val = embedding.norm(dim=1).mean().item()
        min_val = embedding.min().item()
        max_val = embedding.max().item()
        sparsity = (embedding.abs() < 0.01).float().mean().item()
        
        print(f"{modality_name:<20} {str(tuple(embedding.shape)):<20} "
              f"{norm_val:<15.4f} {min_val:<10.4f} {max_val:<10.4f} {sparsity:<10.2%}")
    
    print("─" * 80)
    print(f"\n✓ All Encoders Test PASSED")
    
    # Additional analysis
    print("\nDimensionality Summary:")
    print(f"  Vision:     {vision_emb_avg.shape[-1]}-dim")
    print(f"  Audio:      {audio_emb.shape[-1]}-dim")
    print(f"  Pressure:   {pressure_emb.shape[-1]}-dim")
    print(f"  EMG:        {emg_emb.shape[-1]}-dim")
    print(f"  Total:      {vision_emb_avg.shape[-1] + audio_emb.shape[-1] + pressure_emb.shape[-1] + emg_emb.shape[-1]}-dim (before fusion)")


def test_preprocessing_pipeline():
    """Test preprocessing for each modality"""
    print("\n" + "=" * 80)
    print("TEST 6: PREPROCESSING PIPELINE")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # Test Vision Preprocessing
    print("Vision Preprocessing:")
    print("─" * 40)
    vision_prep = VisionPreprocessor(image_size=(224, 224))
    
    # Simulate different image formats
    image_np = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    image_tensor = torch.rand(480, 640, 3)
    
    print(f"Input (numpy uint8): {image_np.shape}")
    processed1 = vision_prep.preprocess(image_np)
    print(f"Output: {tuple(processed1.shape)}, range=[{processed1.min():.2f}, {processed1.max():.2f}]")
    
    print(f"\nInput (tensor float): {image_tensor.shape}")
    processed2 = vision_prep.preprocess(image_tensor)
    print(f"Output: {tuple(processed2.shape)}, range=[{processed2.min():.2f}, {processed2.max():.2f}]")
    
    # Test Audio Preprocessing
    print("\n\nAudio Preprocessing:")
    print("─" * 40)
    audio_prep = AudioPreprocessor(sample_rate=16000, duration=3.0)
    
    audio_data = np.random.randn(48000).astype(np.float32)
    print(f"Input: {audio_data.shape}, range=[{audio_data.min():.2f}, {audio_data.max():.2f}]")
    
    processed_audio = audio_prep.preprocess(audio_data)
    print(f"Output: {tuple(processed_audio.shape)}, range=[{processed_audio.min():.2f}, {processed_audio.max():.2f}]")
    
    # Test Sensor Preprocessing
    print("\n\nSensor Preprocessing:")
    print("─" * 40)
    sensor_prep = SensorPreprocessor(normalize=True, standardize=True)
    
    sensor_data = torch.randn(1, 1000)
    print(f"Input: {tuple(sensor_data.shape)}, range=[{sensor_data.min():.2f}, {sensor_data.max():.2f}]")
    
    processed_sensor = sensor_prep.preprocess(sensor_data)
    print(f"Output: {tuple(processed_sensor.shape)}, range=[{processed_sensor.min():.2f}, {processed_sensor.max():.2f}]")
    
    # Test temporal feature extraction
    print(f"\nTemporal Feature Extraction (window_size=50):")
    features = sensor_prep.extract_features_temporal(sensor_data, window_size=50)
    print(f"Output: {tuple(features.shape)}")
    print(f"Features computed per window: mean, std, min, max, energy (5 features)")
    
    print("\n✓ Preprocessing Pipeline Test PASSED")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ROBOTIC MULTIMODAL FUSION SYSTEM - ENCODER TESTING")
    print("=" * 80)
    
    try:
        # Run all tests
        test_vision_encoder()
        test_audio_encoder()
        test_pressure_sensor_encoder()
        test_emg_sensor_encoder()
        test_all_encoders_together()
        test_preprocessing_pipeline()
        
        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
        print("=" * 80)
        
        print("\nSummary:")
        print("  ✓ Vision Encoder (CLIP):       512-dim")
        print("  ✓ Audio Encoder (Wav2Vec):     768-dim")
        print("  ✓ Pressure Sensor Encoder:     256-dim")
        print("  ✓ EMG Sensor Encoder:          256-dim")
        print("  ─────────────────────────────")
        print("  → Total before fusion:         1792-dim")
        print("  → After fusion (projected):    512-dim (unified embedding)")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
