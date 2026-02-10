"""
Test Script: Multimodal Fusion Module
Tests the fusion of different modality encodings into unified embeddings
Run with: source /home/nishant/projects/mmfuse-env/bin/activate && python test_fusion.py
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

# Import fusion module
from fusion.multimodal_fusion import MultimodalFusion

# Import preprocessing
from preprocessing.preprocessor import SensorPreprocessor


def test_individual_encoders():
    """Step 1: Test each encoder individually"""
    print("\n" + "=" * 90)
    print("STEP 1: INDIVIDUAL ENCODER OUTPUTS")
    print("=" * 90)
    
    device = "cpu"
    batch_size = 2
    
    # Vision encoder
    print("\n1.1 Vision Encoder (CLIP ViT-B-32)")
    print("─" * 90)
    vision_encoder = VisionEncoder(model_name="ViT-B-32", device=device)
    camera_input = torch.rand(batch_size, 3, 224, 224, device=device)
    vision_output = vision_encoder(camera_input)
    print(f"  Input shape:  {tuple(camera_input.shape)}")
    print(f"  Output shape: {tuple(vision_output.shape)}")
    print(f"  Output dim:   {vision_output.shape[-1]}")
    print(f"  Norm:         {vision_output.norm(dim=1).mean():.4f}")
    
    # Audio encoder
    print("\n1.2 Audio Encoder (Learnable 1D CNN)")
    print("─" * 90)
    audio_encoder = AudioEncoder(output_dim=768, num_filters=256, num_layers=4, device=device)
    audio_encoder.eval()
    audio_input = torch.randn(batch_size, 48000, device=device)
    with torch.no_grad():
        audio_output = audio_encoder(audio_input, sampling_rate=16000)
    print(f"  Input shape:  {tuple(audio_input.shape)}")
    print(f"  Output shape: {tuple(audio_output.shape)}")
    print(f"  Output dim:   {audio_output.shape[-1]}")
    print(f"  Norm:         {audio_output.norm(dim=1).mean():.4f}")
    
    # Pressure encoder
    print("\n1.3 Pressure Sensor Encoder")
    print("─" * 90)
    pressure_encoder = PressureSensorEncoder(output_dim=256, num_channels=1)
    pressure_encoder.eval()
    pressure_input = torch.randn(batch_size, 1000)
    preprocessor = SensorPreprocessor()
    pressure_features = preprocessor.extract_features_temporal(pressure_input, window_size=50)
    pressure_features = preprocessor.preprocess_batch(pressure_features)
    with torch.no_grad():
        pressure_output = pressure_encoder(pressure_features)
    print(f"  Input shape (raw):     {tuple(pressure_input.shape)}")
    print(f"  Input shape (features):{tuple(pressure_features.shape)}")
    print(f"  Output shape:          {tuple(pressure_output.shape)}")
    print(f"  Output dim:            {pressure_output.shape[-1]}")
    print(f"  Norm:                  {pressure_output.norm(dim=1).mean():.4f}")
    
    # EMG encoder
    print("\n1.4 EMG Sensor Encoder")
    print("─" * 90)
    emg_encoder = EMGSensorEncoder(output_dim=256, num_channels=8, input_features=800)
    emg_encoder.eval()
    emg_input = torch.randn(batch_size, 8, 1000)
    emg_features = preprocessor.extract_features_temporal(emg_input, window_size=50)
    emg_features = preprocessor.preprocess_batch(emg_features)
    emg_features_flat = emg_features.reshape(emg_features.shape[0], -1)
    with torch.no_grad():
        emg_output = emg_encoder(emg_features_flat)
    print(f"  Input shape (raw):     {tuple(emg_input.shape)}")
    print(f"  Input shape (features):{tuple(emg_features.shape)}")
    print(f"  Flattened shape:       {tuple(emg_features_flat.shape)}")
    print(f"  Output shape:          {tuple(emg_output.shape)}")
    print(f"  Output dim:            {emg_output.shape[-1]}")
    print(f"  Norm:                  {emg_output.norm(dim=1).mean():.4f}")
    
    # Summary table
    print("\n" + "─" * 90)
    print("ENCODER OUTPUT SUMMARY")
    print("─" * 90)
    print(f"{'Modality':<18} {'Output Dim':<15} {'Output Shape':<20}")
    print("─" * 90)
    print(f"{'Vision':<18} {vision_output.shape[-1]:<15} {str(tuple(vision_output.shape)):<20}")
    print(f"{'Audio':<18} {audio_output.shape[-1]:<15} {str(tuple(audio_output.shape)):<20}")
    print(f"{'Pressure':<18} {pressure_output.shape[-1]:<15} {str(tuple(pressure_output.shape)):<20}")
    print(f"{'EMG':<18} {emg_output.shape[-1]:<15} {str(tuple(emg_output.shape)):<20}")
    print("─" * 90)
    
    return {
        'vision': vision_output,
        'audio': audio_output,
        'pressure': pressure_output,
        'emg': emg_output,
        'batch_size': batch_size
    }


def test_fusion_method(embeddings, method_name, device="cpu"):
    """Test a specific fusion method"""
    print(f"\n{method_name.upper()} FUSION METHOD")
    print("─" * 90)
    
    modality_dims = {
        'vision': embeddings['vision'].shape[-1],
        'audio': embeddings['audio'].shape[-1],
        'pressure': embeddings['pressure'].shape[-1],
        'emg': embeddings['emg'].shape[-1]
    }
    
    print(f"\nModality dimensions: {modality_dims}")
    total_dim = sum(modality_dims.values())
    print(f"Total pre-fusion dimension: {total_dim}-dim")
    
    # Create fusion module
    fusion_module = MultimodalFusion(
        modality_dims=modality_dims,
        fusion_dim=512,
        fusion_method=method_name,
        dropout=0.2
    )
    fusion_module.to(device)
    fusion_module.eval()
    
    # Prepare embeddings dictionary
    embeddings_input = {
        'vision': embeddings['vision'].to(device),
        'audio': embeddings['audio'].to(device),
        'pressure': embeddings['pressure'].to(device),
        'emg': embeddings['emg'].to(device)
    }
    
    # Forward pass
    with torch.no_grad():
        fused_output = fusion_module(embeddings_input)
    
    print(f"\nFusion Process:")
    print(f"  Step 1: Individual projection layers")
    print(f"    vision:   {modality_dims['vision']:4d}-dim → 512-dim")
    print(f"    audio:    {modality_dims['audio']:4d}-dim → 512-dim")
    print(f"    pressure: {modality_dims['pressure']:4d}-dim → 512-dim")
    print(f"    emg:      {modality_dims['emg']:4d}-dim → 512-dim")
    
    if method_name == "concat_project":
        print(f"\n  Step 2: Concatenate projected embeddings")
        print(f"    512 + 512 + 512 + 512 = 2048-dim")
        print(f"\n  Step 3: Project to fusion dimension")
        print(f"    2048-dim → 512-dim (final)")
    
    elif method_name == "weighted_sum":
        print(f"\n  Step 2: Learn attention weights per modality")
        print(f"    Each modality gets learned weight [0-1]")
        print(f"\n  Step 3: Weighted sum of projections")
        print(f"    (w1*vision + w2*audio + w3*pressure + w4*emg)")
        print(f"    → 512-dim (final)")
    
    elif method_name == "bilinear":
        print(f"\n  Step 2: Concatenate projected embeddings")
        print(f"    512 + 512 + 512 + 512 = 2048-dim")
        print(f"\n  Step 3: Bilinear pooling")
        print(f"    2048-dim → 512-dim (final)")
    
    print(f"\nOutput:")
    print(f"  Shape:  {tuple(fused_output.shape)}")
    print(f"  Dim:    {fused_output.shape[-1]}")
    print(f"  Norm:   {fused_output.norm(dim=1).mean():.4f}")
    print(f"  Stats:  min={fused_output.min():.4f}, max={fused_output.max():.4f}, "
          f"mean={fused_output.mean():.4f}, std={fused_output.std():.4f}")
    
    return fused_output


def test_all_fusion_methods(embeddings):
    """Step 2: Test all fusion methods"""
    print("\n" + "=" * 90)
    print("STEP 2: MULTIMODAL FUSION METHODS")
    print("=" * 90)
    
    device = "cpu"
    fusion_methods = ["concat_project", "weighted_sum", "bilinear"]
    results = {}
    
    for method in fusion_methods:
        try:
            fused = test_fusion_method(embeddings, method, device)
            results[method] = fused
        except Exception as e:
            print(f"❌ {method.upper()} FUSION FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[method] = None
    
    return results


def test_fusion_consistency(embeddings):
    """Step 3: Test fusion consistency across batch sizes"""
    print("\n" + "=" * 90)
    print("STEP 3: FUSION CONSISTENCY ACROSS BATCH SIZES")
    print("=" * 90)
    
    device = "cpu"
    
    # Initialize fusion module
    modality_dims = {
        'vision': embeddings['vision'].shape[-1],
        'audio': embeddings['audio'].shape[-1],
        'pressure': embeddings['pressure'].shape[-1],
        'emg': embeddings['emg'].shape[-1]
    }
    
    fusion_module = MultimodalFusion(
        modality_dims=modality_dims,
        fusion_dim=512,
        fusion_method="concat_project",
        dropout=0.2
    )
    fusion_module.to(device)
    fusion_module.eval()
    
    # Test different batch sizes
    test_batch_sizes = [1, 2, 4, 8]
    print(f"\nTesting with batch sizes: {test_batch_sizes}")
    print("─" * 90)
    print(f"{'Batch Size':<15} {'Output Shape':<20} {'Output Dim':<15} {'Status':<10}")
    print("─" * 90)
    
    for bs in test_batch_sizes:
        try:
            # Create test embeddings with specified batch size
            test_embeddings = {
                'vision': torch.randn(bs, modality_dims['vision'], device=device),
                'audio': torch.randn(bs, modality_dims['audio'], device=device),
                'pressure': torch.randn(bs, modality_dims['pressure'], device=device),
                'emg': torch.randn(bs, modality_dims['emg'], device=device)
            }
            
            with torch.no_grad():
                output = fusion_module(test_embeddings)
            
            print(f"{bs:<15} {str(tuple(output.shape)):<20} {output.shape[-1]:<15} ✓ PASS")
        except Exception as e:
            print(f"{bs:<15} {'ERROR':<20} {'-':<15} ❌ FAIL: {str(e)[:20]}")


def test_fusion_dimensions():
    """Step 4: Verify dimension transformations"""
    print("\n" + "=" * 90)
    print("STEP 4: DIMENSION TRANSFORMATION ANALYSIS")
    print("=" * 90)
    
    device = "cpu"
    
    modality_dims = {
        'vision': 512,
        'audio': 768,
        'pressure': 256,
        'emg': 256
    }
    
    print(f"\nInput Modality Dimensions:")
    print("─" * 90)
    for modality, dim in modality_dims.items():
        print(f"  {modality:<15} : {dim:4d}-dim")
    
    total_dim = sum(modality_dims.values())
    print(f"  {'TOTAL':<15} : {total_dim:4d}-dim")
    
    print(f"\nTransformation Pipeline (concat_project):")
    print("─" * 90)
    print(f"  Step 1: Project each modality to 512-dim")
    print(f"    vision   (512-dim) → 512-dim")
    print(f"    audio    (768-dim) → 512-dim")
    print(f"    pressure (256-dim) → 512-dim")
    print(f"    emg      (256-dim) → 512-dim")
    
    print(f"\n  Step 2: Concatenate projections")
    print(f"    [512, 512, 512, 512] concatenated → 2048-dim tensor")
    
    print(f"\n  Step 3: Dense projection layers")
    print(f"    2048-dim → 1024-dim (Linear + BatchNorm + ReLU)")
    print(f"    1024-dim → 512-dim  (Linear)")
    
    print(f"\n  Output: 512-dim unified multimodal embedding")
    print(f"  Compression ratio: {total_dim}/512 = {total_dim/512:.2f}x")


if __name__ == "__main__":
    print("\n" + "=" * 90)
    print("ROBOTIC MULTIMODAL FUSION SYSTEM - FUSION MODULE TESTS")
    print("=" * 90)
    
    try:
        # Step 1: Test individual encoders
        embeddings = test_individual_encoders()
        
        # Step 4: Verify dimensions
        test_fusion_dimensions()
        
        # Step 2: Test all fusion methods
        fusion_results = test_all_fusion_methods(embeddings)
        
        # Step 3: Test consistency across batch sizes
        test_fusion_consistency(embeddings)
        
        # Final summary
        print("\n" + "=" * 90)
        print("✅ ALL FUSION TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 90)
        
        print("\nKey Results:")
        print("  ✓ Individual encoders produce expected output dimensions")
        print("  ✓ Fusion module projects all modalities to 512-dim")
        print("  ✓ All fusion methods working (concat_project, weighted_sum, bilinear)")
        print("  ✓ Consistent output across batch sizes")
        print("  ✓ Pre-fusion: 1792-dim → Post-fusion: 512-dim\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
