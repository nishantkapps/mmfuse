"""
Example usage and demo of the Robotic Multimodal Feedback System
"""

import torch
import numpy as np
from pathlib import Path

from robotic_feedback_system import RoboticFeedbackSystem


def create_dummy_inputs(batch_size: int = 2, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Create dummy inputs for testing the system
    
    Returns:
        Dictionary with dummy camera, audio, and sensor inputs
    """
    
    # Dummy camera inputs (2 cameras, RGB images 224x224)
    camera1 = torch.randn(batch_size, 3, 224, 224, device=device)
    camera2 = torch.randn(batch_size, 3, 224, 224, device=device)
    
    # Normalize to [0, 1] range (typical for images)
    camera1 = (camera1 - camera1.min()) / (camera1.max() - camera1.min())
    camera2 = (camera2 - camera2.min()) / (camera2.max() - camera2.min())
    
    # Dummy audio input (16kHz, 3 seconds = 48000 samples)
    audio = torch.randn(batch_size, 48000, device=device)
    
    # Dummy pressure sensor (sequence of 1000 timesteps)
    pressure = torch.randn(batch_size, 1000, device=device)
    
    # Dummy EMG sensor (8 channels, 1000 timesteps)
    emg = torch.randn(batch_size, 8, 1000, device=device)
    
    return {
        'camera1': camera1,
        'camera2': camera2,
        'audio': audio,
        'pressure': pressure,
        'emg': emg
    }


def demo_basic_fusion():
    """Demo: Basic multimodal fusion with concatenation"""
    print("=" * 80)
    print("DEMO 1: Basic Multimodal Fusion (Concatenation)")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Initialize system
    system = RoboticFeedbackSystem(
        fusion_dim=512,
        fusion_method="concat_project",
        use_attention=False,
        device=device
    )
    system.to(device)
    system.eval()  # Set to evaluation mode since using pre-trained encoders
    
    print("System initialized with following modality dimensions:")
    dims = system.get_modality_dimensions()
    for modality, dim in dims.items():
        print(f"  {modality:15s}: {dim} dims")
    
    # Create dummy inputs
    print("\nCreating dummy inputs...")
    inputs = create_dummy_inputs(batch_size=2, device=device)
    
    # Process inputs
    print("\nProcessing multimodal inputs...")
    with torch.no_grad():
        fused_embedding, modality_embeddings = system(
            camera_images={'camera1': inputs['camera1'], 'camera2': inputs['camera2']},
            audio=inputs['audio'],
            pressure=inputs['pressure'],
            emg=inputs['emg'],
            return_modality_embeddings=True
        )
    
    print(f"\nFused embedding shape: {fused_embedding.shape}")
    print(f"Fused embedding (first sample):\n{fused_embedding[0, :10]}...")
    
    print("\nIndividual modality embeddings:")
    for modality, emb in modality_embeddings.items():
        print(f"  {modality:15s}: {emb.shape}")


def demo_attention_fusion():
    """Demo: Multimodal fusion with attention mechanism"""
    print("\n" + "=" * 80)
    print("DEMO 2: Multimodal Fusion with Attention")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Initialize system with attention
    system = RoboticFeedbackSystem(
        fusion_dim=512,
        use_attention=True,
        device=device
    )
    system.to(device)
    system.eval()
    
    # Create dummy inputs
    inputs = create_dummy_inputs(batch_size=2, device=device)
    
    # Process inputs
    print("Processing with attention-based fusion...")
    with torch.no_grad():
        fused_embedding = system(
            camera_images={'camera1': inputs['camera1'], 'camera2': inputs['camera2']},
            audio=inputs['audio'],
            pressure=inputs['pressure'],
            emg=inputs['emg']
        )
    
    print(f"\nFused embedding shape: {fused_embedding.shape}")
    print(f"Fused embedding (first sample):\n{fused_embedding[0, :10]}...")


def demo_individual_encoders():
    """Demo: Using individual encoders for specific modalities"""
    print("\n" + "=" * 80)
    print("DEMO 3: Individual Modality Encoders")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    system = RoboticFeedbackSystem(device=device)
    system.eval()
    
    inputs = create_dummy_inputs(batch_size=2, device=device)
    
    # Vision only
    print("Encoding vision only...")
    with torch.no_grad():
        vision_emb = system.encode_vision_only(
            {'camera1': inputs['camera1'], 'camera2': inputs['camera2']}
        )
    print(f"  Vision embedding shape: {vision_emb.shape}")
    
    # Audio only
    print("\nEncoding audio only...")
    with torch.no_grad():
        audio_emb = system.encode_audio_only(inputs['audio'])
    print(f"  Audio embedding shape: {audio_emb.shape}")
    
    # Pressure only
    print("\nEncoding pressure sensor only...")
    with torch.no_grad():
        pressure_emb = system.encode_pressure_only(inputs['pressure'])
    print(f"  Pressure embedding shape: {pressure_emb.shape}")
    
    # EMG only
    print("\nEncoding EMG sensor only...")
    with torch.no_grad():
        emg_emb = system.encode_emg_only(inputs['emg'])
    print(f"  EMG embedding shape: {emg_emb.shape}")


def demo_batch_processing():
    """Demo: Processing variable batch sizes"""
    print("\n" + "=" * 80)
    print("DEMO 4: Batch Processing with Different Sizes")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    system = RoboticFeedbackSystem(device=device)
    system.eval()
    
    for batch_size in [1, 4, 8]:
        print(f"\nProcessing batch size: {batch_size}")
        inputs = create_dummy_inputs(batch_size=batch_size, device=device)
        
        with torch.no_grad():
            fused = system(
                camera_images={'camera1': inputs['camera1'], 'camera2': inputs['camera2']},
                audio=inputs['audio'],
                pressure=inputs['pressure'],
                emg=inputs['emg']
            )
        
        print(f"  Fused embedding shape: {fused.shape}")


def demo_similarity_analysis():
    """Demo: Computing similarities between modalities"""
    print("\n" + "=" * 80)
    print("DEMO 5: Cross-Modal Similarity Analysis")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    system = RoboticFeedbackSystem(device=device)
    system.eval()
    
    inputs = create_dummy_inputs(batch_size=4, device=device)
    
    # Get individual modality embeddings
    with torch.no_grad():
        _, modality_embs = system(
            camera_images={'camera1': inputs['camera1'], 'camera2': inputs['camera2']},
            audio=inputs['audio'],
            pressure=inputs['pressure'],
            emg=inputs['emg'],
            return_modality_embeddings=True
        )
    
    # Compute cosine similarity between modalities
    modalities = list(modality_embs.keys())
    print("Cross-modal cosine similarities (first sample):\n")
    
    for i, mod1 in enumerate(modalities):
        for mod2 in modalities[i+1:]:
            emb1 = modality_embs[mod1][0:1]  # First sample
            emb2 = modality_embs[mod2][0:1]
            
            # Normalize
            emb1 = emb1 / emb1.norm(dim=1, keepdim=True)
            emb2 = emb2 / emb2.norm(dim=1, keepdim=True)
            
            # Compute cosine similarity
            similarity = torch.matmul(emb1, emb2.t()).item()
            print(f"  {mod1:12s} <-> {mod2:12s}: {similarity:.4f}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ROBOTIC MULTIMODAL FEEDBACK SYSTEM - DEMONSTRATIONS")
    print("=" * 80)
    
    # Run all demos
    try:
        demo_basic_fusion()
        demo_attention_fusion()
        demo_individual_encoders()
        demo_batch_processing()
        demo_similarity_analysis()
        
        print("\n" + "=" * 80)
        print("All demos completed successfully!")
        print("=" * 80)
    
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()
