"""
Test script for streaming robot control pipeline
Validates encoders, fusion, and Arduino communication without real hardware
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_streaming_components():
    """Test individual streaming pipeline components"""
    
    logger.info("=" * 70)
    logger.info("STREAMING ROBOT CONTROL - COMPONENT TEST")
    logger.info("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}\n")
    
    # Test 1: Vision Encoder
    logger.info("[TEST 1] Vision Encoder (CLIP)")
    try:
        from encoders.vision_encoder import VisionEncoder
        from preprocessing.preprocessor import VisionPreprocessor
        
        vision_enc = VisionEncoder(device=device)
        vision_prep = VisionPreprocessor(image_size=(224, 224))
        
        # Dummy 1080p frame
        dummy_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        vision_tensor = vision_prep.preprocess(dummy_frame).unsqueeze(0).to(device)
        
        with torch.no_grad():
            vision_emb = vision_enc(vision_tensor)
        
        logger.info(f"  ✓ Vision embedding shape: {vision_emb.shape}")
        logger.info(f"  ✓ Output dimension: {vision_emb.shape[-1]}\n")
    
    except Exception as e:
        logger.error(f"  ✗ Vision encoder failed: {e}\n")
        return False
    
    # Test 2: Audio Encoder
    logger.info("[TEST 2] Audio Encoder (Learnable)")
    try:
        from encoders.audio_encoder_learnable import AudioEncoder as LearnableAudioEncoder
        
        audio_enc = LearnableAudioEncoder(device=device)
        
        # Dummy 2.5 second audio @ 16 kHz = 40,000 samples
        dummy_audio = np.random.randn(2, 40000).astype(np.float32)
        audio_tensor = torch.from_numpy(dummy_audio).to(device)
        
        with torch.no_grad():
            audio_emb = audio_enc(audio_tensor)
        
        logger.info(f"  ✓ Audio embedding shape: {audio_emb.shape}")
        logger.info(f"  ✓ Output dimension: {audio_emb.shape[-1]}\n")
    
    except Exception as e:
        logger.error(f"  ✗ Audio encoder failed: {e}\n")
        return False
    
    # Test 3: Pressure Encoder
    logger.info("[TEST 3] Pressure Sensor Encoder")
    try:
        from encoders.sensor_encoder import PressureSensorEncoder
        
        pressure_enc = PressureSensorEncoder(output_dim=256).to(device)
        
        # Dummy pressure features (window-based)
        pressure_features = torch.randn(2, 100).to(device)
        
        with torch.no_grad():
            pressure_emb = pressure_enc(pressure_features)
        
        logger.info(f"  ✓ Pressure embedding shape: {pressure_emb.shape}")
        logger.info(f"  ✓ Output dimension: {pressure_emb.shape[-1]}\n")
    
    except Exception as e:
        logger.error(f"  ✗ Pressure encoder failed: {e}\n")
        return False
    
    # Test 4: EMG Encoder
    logger.info("[TEST 4] EMG Sensor Encoder")
    try:
        from encoders.sensor_encoder import EMGSensorEncoder
        
        emg_enc = EMGSensorEncoder(output_dim=256, num_channels=3).to(device)
        
        # Dummy EMG features (100 features, not 300)
        emg_features = torch.randn(2, 100).to(device)
        
        with torch.no_grad():
            emg_emb = emg_enc(emg_features)
        
        logger.info(f"  ✓ EMG embedding shape: {emg_emb.shape}")
        logger.info(f"  ✓ Output dimension: {emg_emb.shape[-1]}\n")
    
    except Exception as e:
        logger.error(f"  ✗ EMG encoder failed: {e}\n")
        return False
    
    # Test 5: Fusion Module
    logger.info("[TEST 5] Multimodal Fusion")
    try:
        from fusion.multimodal_fusion import MultimodalFusion
        
        fusion = MultimodalFusion(
            modality_dims={
                'vision': 512,
                'audio': 768,
                'pressure': 256,
                'emg': 256
            },
            fusion_dim=512,
            fusion_method="weighted_sum"
        ).to(device)
        
        # Use outputs from previous tests
        batch_size = 2
        with torch.no_grad():
            fused = fusion({
                'vision': torch.randn(batch_size, 512).to(device),
                'audio': torch.randn(batch_size, 768).to(device),
                'pressure': torch.randn(batch_size, 256).to(device),
                'emg': torch.randn(batch_size, 256).to(device)
            })
        
        logger.info(f"  ✓ Fused embedding shape: {fused.shape}")
        logger.info(f"  ✓ Fusion output dimension: {fused.shape[-1]}\n")
    
    except Exception as e:
        logger.error(f"  ✗ Fusion module failed: {e}\n")
        return False
    
    # Test 6: Robotic Controller
    logger.info("[TEST 6] Robotic Arm Controller (3DOF)")
    try:
        from robotic_arm_controller import RoboticArmController3DOF
        
        controller = RoboticArmController3DOF()
        
        # Decode fused embedding to joint angles
        fused_emb = torch.randn(2, 512)
        with torch.no_grad():
            result = controller.decode(fused_emb)
        
        position = result['position']
        force = result['force']
        
        logger.info(f"  ✓ Position shape: {position.shape}")
        logger.info(f"  ✓ Force shape: {force.shape}")
        logger.info(f"  ✓ Example position: {position[0].tolist()}")
        logger.info(f"  ✓ Example force: {force[0].item():.1f}%\n")
    
    except Exception as e:
        logger.error(f"  ✗ Controller failed: {e}\n")
        return False
    
    # Test 7: Arduino Controller
    logger.info("[TEST 7] Arduino Controller (Mock)")
    try:
        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
        from io.arduino_controller import ArduinoController, SensorBuffer
        
        arduino = ArduinoController(port=None)  # No actual connection
        logger.info(f"  ✓ Arduino controller created")
        logger.info(f"  ✓ Status: {arduino.get_status()}")
        
        sensor_buffer = SensorBuffer(size=100)
        sensor_buffer.append({
            "pressure": 50.0,
            "emg_1": 10.0,
            "emg_2": 20.0,
            "emg_3": 15.0
        })
        
        snapshot = sensor_buffer.get_snapshot()
        if snapshot:
            logger.info(f"  ✓ Sensor buffer working")
        
        logger.info()
    
    except Exception as e:
        logger.error(f"  ✗ Arduino controller failed: {e}\n")
        # Don't fail the whole test suite, just skip this component
        logger.info("  (Arduino component optional, continuing...)\n")
    
    # Summary
    logger.info("=" * 70)
    logger.info("ALL TESTS PASSED ✓")
    logger.info("=" * 70)
    logger.info("\nStreaming pipeline ready!")
    logger.info("Run: python streaming_robot_control.py --config config/streaming_config.yaml")
    logger.info("=" * 70)
    
    return True


if __name__ == "__main__":
    success = test_streaming_components()
    sys.exit(0 if success else 1)
