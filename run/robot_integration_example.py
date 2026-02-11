"""
Practical example: Real robot integration with actual sensor data
Demonstrates how to use the multimodal fusion system in a real robotic application
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robotic_feedback_system import RoboticFeedbackSystem
from preprocessing.preprocessor import (
    VisionPreprocessor,
    AudioPreprocessor,
    SensorPreprocessor
)
from config import get_config, SystemConfig


class RoboticSensorInterface:
    """
    Interface for reading from actual robot sensors
    In a real system, this would connect to actual hardware
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize sensor interface
        
        Args:
            config: System configuration
        """
        self.config = config
        self.vision_prep = VisionPreprocessor(
            image_size=tuple(config.vision.image_size),
            normalize=config.vision.normalize
        )
        self.audio_prep = AudioPreprocessor(
            sample_rate=config.audio.sampling_rate,
            duration=config.audio.duration,
            normalize=config.audio.normalize
        )
        self.sensor_prep = SensorPreprocessor(
            normalize=config.pressure.normalize,
            standardize=config.pressure.standardize
        )
    
    def read_camera_frame(
        self,
        camera_id: int = 1
    ) -> torch.Tensor:
        """
        Read frame from camera
        
        In real implementation, would read from:
        - OpenCV (cv2.VideoCapture)
        - ROS camera driver
        - Hardware camera API
        
        Args:
            camera_id: Camera identifier (1 or 2)
        
        Returns:
            Preprocessed image tensor (3, H, W)
        """
        # Dummy implementation: random image
        # Replace with actual camera reading
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Preprocess
        tensor = self.vision_prep.preprocess(frame)
        return tensor
    
    def read_audio_chunk(self, duration: float = 3.0) -> torch.Tensor:
        """
        Read audio chunk from microphone
        
        In real implementation, would read from:
        - pyaudio
        - sounddevice
        - ROS audio subscriber
        
        Args:
            duration: Duration in seconds
        
        Returns:
            Preprocessed audio tensor (num_samples,)
        """
        # Dummy implementation: random audio
        num_samples = int(duration * self.config.audio.sampling_rate)
        audio = np.random.randn(num_samples).astype(np.float32)
        
        # Preprocess
        tensor = self.audio_prep.preprocess(audio)
        return tensor
    
    def read_pressure_sensor(self, duration: float = 1.0) -> torch.Tensor:
        """
        Read pressure sensor readings
        
        In real implementation, would read from:
        - I2C/SPI sensor
        - Pressure sensor driver
        - ROS sensor subscriber
        
        Args:
            duration: Duration of readings in seconds
        
        Returns:
            Sensor readings tensor (sequence_length,)
        """
        # Dummy implementation: simulated pressure readings
        num_samples = int(duration * 1000)  # 1000 Hz sampling
        pressure = np.random.randn(num_samples).astype(np.float32)
        
        # Preprocess
        tensor = self.sensor_prep.preprocess(pressure)
        return tensor
    
    def read_emg_sensors(self, duration: float = 1.0) -> torch.Tensor:
        """
        Read EMG sensor readings (8 channels)
        
        In real implementation, would read from:
        - EMG sensor array
        - EMG amplifier
        - ROS EMG subscriber
        
        Args:
            duration: Duration of readings in seconds
        
        Returns:
            EMG readings tensor (num_channels, sequence_length)
        """
        # Dummy implementation: simulated EMG signals
        num_channels = self.config.emg.num_channels
        num_samples = int(duration * 1000)  # 1000 Hz sampling
        emg = np.random.randn(num_channels, num_samples).astype(np.float32)
        
        # Preprocess
        tensor = self.sensor_prep.preprocess(emg)
        return tensor


class RoboticFeedbackProcessor:
    """
    Real-time processor for robotic multimodal feedback
    """
    
    def __init__(
        self,
        config: SystemConfig,
        device: str = "cuda"
    ):
        """
        Initialize processor
        
        Args:
            config: System configuration
            device: Computation device
        """
        self.config = config
        self.device = device
        
        # Initialize fusion system
        self.system = RoboticFeedbackSystem(
            fusion_dim=config.fusion.fusion_dim,
            fusion_method=config.fusion.fusion_method,
            use_attention=config.fusion.use_attention,
            device=device
        )
        self.system.eval()
        
        # Initialize sensor interface
        self.sensor_interface = RoboticSensorInterface(config)
        
        # Freeze pre-trained encoders
        self.system.freeze_encoders()
    
    def get_multimodal_embedding(
        self,
        return_modality_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Get current multimodal embedding from all sensors
        
        This is the main function to call in a robot control loop
        
        Args:
            return_modality_embeddings: Whether to return individual modalities
        
        Returns:
            Dictionary with:
            - 'fused': Fused multimodal embedding
            - 'modalities': Optional dict with individual modality embeddings
            - 'timestamp': Timestamp of acquisition
        """
        # Read all sensor data
        camera1 = self.sensor_interface.read_camera_frame(1).unsqueeze(0)
        camera2 = self.sensor_interface.read_camera_frame(2).unsqueeze(0)
        audio = self.sensor_interface.read_audio_chunk().unsqueeze(0)
        pressure = self.sensor_interface.read_pressure_sensor().unsqueeze(0)
        emg = self.sensor_interface.read_emg_sensors().unsqueeze(0)
        
        # Move to device
        camera1 = camera1.to(self.device)
        camera2 = camera2.to(self.device)
        audio = audio.to(self.device)
        pressure = pressure.to(self.device)
        emg = emg.to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            if return_modality_embeddings:
                fused, modalities = self.system(
                    camera_images={'camera1': camera1, 'camera2': camera2},
                    audio=audio,
                    pressure=pressure,
                    emg=emg,
                    return_modality_embeddings=True
                )
            else:
                fused = self.system(
                    camera_images={'camera1': camera1, 'camera2': camera2},
                    audio=audio,
                    pressure=pressure,
                    emg=emg
                )
                modalities = None
        
        result = {
            'fused': fused.squeeze(0).cpu(),
            'timestamp': time.time()
        }
        
        if modalities:
            result['modalities'] = {
                k: v.squeeze(0).cpu() for k, v in modalities.items()
            }
        
        return result


def example_robot_control_loop():
    """
    Example of using multimodal embeddings in a robot control loop
    """
    print("=" * 80)
    print("EXAMPLE: Robot Control Loop with Multimodal Feedback")
    print("=" * 80)
    
    # Get configuration
    config = get_config("balanced")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize processor
    processor = RoboticFeedbackProcessor(config, device=device)
    
    print(f"Device: {device}")
    print(f"Fusion dimension: {config.fusion.fusion_dim}")
    print(f"Fusion method: {config.fusion.fusion_method}\n")
    
    # Simulate robot control loop
    print("Running robot control loop (5 iterations)...\n")
    
    for iteration in range(5):
        print(f"Iteration {iteration + 1}")
        
        # Get multimodal embedding
        result = processor.get_multimodal_embedding(
            return_modality_embeddings=True
        )
        
        fused_embedding = result['fused']
        modalities = result['modalities']
        
        print(f"  Fused embedding shape: {fused_embedding.shape}")
        print(f"  Fused embedding norm: {fused_embedding.norm():.4f}")
        
        # In a real system, use fused_embedding for:
        # - Robot control policy
        # - Anomaly detection
        # - Situation understanding
        # - Decision making
        
        # Example: compute modality importance
        if modalities:
            print(f"  Modality norms:")
            for mod, emb in modalities.items():
                print(f"    {mod:12s}: {emb.norm():.4f}")
        
        print()
        
        # In real system, would control robot here
        # e.g., execute action based on fused_embedding
        # time.sleep(control_period)
    
    print("Control loop completed!")


def example_sensor_monitoring():
    """
    Example of monitoring sensor health using multimodal embeddings
    """
    print("\n" + "=" * 80)
    print("EXAMPLE: Sensor Health Monitoring")
    print("=" * 80)
    
    config = get_config("balanced")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = RoboticFeedbackProcessor(config, device=device)
    
    # Get baseline embedding (normal operation)
    print("Acquiring baseline embedding...")
    baseline_result = processor.get_multimodal_embedding(
        return_modality_embeddings=True
    )
    baseline_embedding = baseline_result['fused']
    baseline_modalities = baseline_result['modalities']
    
    print(f"Baseline embedding norm: {baseline_embedding.norm():.4f}\n")
    
    # Simulate anomaly detection
    print("Simulating 5 sensor readings and checking for anomalies...\n")
    
    for iteration in range(5):
        result = processor.get_multimodal_embedding(
            return_modality_embeddings=True
        )
        fused = result['fused']
        
        # Compute similarity to baseline
        similarity = torch.nn.functional.cosine_similarity(
            baseline_embedding.unsqueeze(0),
            fused.unsqueeze(0)
        ).item()
        
        print(f"Reading {iteration + 1}:")
        print(f"  Similarity to baseline: {similarity:.4f}")
        
        # Alert if significant deviation
        if similarity < 0.85:
            print(f"  ⚠️  ANOMALY DETECTED (similarity < 0.85)")
        else:
            print(f"  ✓ Normal operation")
        
        print()


if __name__ == "__main__":
    # Run examples
    try:
        example_robot_control_loop()
        example_sensor_monitoring()
        
        print("=" * 80)
        print("Examples completed successfully!")
        print("=" * 80)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
