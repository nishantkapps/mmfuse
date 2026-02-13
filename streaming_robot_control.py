"""
Real-time streaming multimodal robotic arm control
Continuously processes video, audio, and sensor inputs at 60 Hz
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from collections import deque
import threading
import time
import logging
from typing import Optional, Dict, Tuple
from datetime import datetime
import yaml

from encoders.vision_encoder import VisionEncoder
from encoders.audio_encoder_learnable import AudioEncoder as LearnableAudioEncoder
from encoders.sensor_encoder import PressureSensorEncoder, EMGSensorEncoder
from fusion.multimodal_fusion import MultimodalFusion
from robotic_arm_controller import RoboticArmController3DOF
from preprocessing.preprocessor import VisionPreprocessor, AudioPreprocessor
# Avoid conflict with built-in 'io' module by importing directly from file path
import sys

from arduino_controller import ArduinoController, SensorBuffer

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class StreamingRobotController:
    """
    End-to-end streaming pipeline for multimodal robotic arm control
    
    Pipeline:
    1. Capture: Webcam (1080p @ 60 Hz), Audio (16 kHz, 2.5s rolling buffer)
    2. Preprocess: Resize images, buffer audio, read sensor data
    3. Encode: Vision, Audio, Pressure, EMG -> embeddings
    4. Fuse: Combine modalities -> unified 512-dim embedding
    5. Decode: Embedding -> 3DOF positions + gripper force
    6. Control: Send commands to Arduino Uno
    """
    
    def __init__(
        self,
        config_path: str = "config/streaming_config.yaml",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize Streaming Robot Controller
        
        Args:
            config_path: Path to YAML configuration file
            device: Compute device ("cuda" or "cpu")
        """
        self.device = device
        self.running = False
        self.config = self._load_config(config_path)
        
        # Extract config
        streaming_cfg = self.config.get("streaming", {})
        arduino_cfg = self.config.get("arduino", {})
        
        self.target_fps = streaming_cfg.get("target_fps", 60)
        self.frame_time = 1.0 / self.target_fps
        self.video_resolution = tuple(streaming_cfg.get("video_resolution", [1080, 1920]))
        self.audio_buffer_duration = streaming_cfg.get("audio_buffer_duration", 2.5)
        self.audio_sample_rate = streaming_cfg.get("audio_sample_rate", 16000)
        
        logger.info(f"Initializing streaming pipeline: {self.target_fps} Hz, {self.video_resolution} resolution")
        
        # Initialize components
        self._init_encoders()
        self._init_preprocessors()
        self._init_fusion_and_controller()
        self._init_arduino(arduino_cfg)
        self._init_buffers()
        
        logger.info("Streaming robot controller initialized successfully")
    
    def _load_config(self, config_path: str) -> dict:
        """Load YAML configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config not found at {config_path}, using defaults")
            return {}
    
    def _init_encoders(self):
        """Initialize all modality encoders"""
        logger.info("Initializing encoders...")
        
        self.vision_encoder = VisionEncoder(device=self.device)
        self.audio_encoder = LearnableAudioEncoder(device=self.device)
        self.pressure_encoder = PressureSensorEncoder(output_dim=256)
        self.emg_encoder = EMGSensorEncoder(output_dim=256, num_channels=3)
        
        # Move to device
        self.pressure_encoder.to(self.device)
        self.emg_encoder.to(self.device)
        
        # Eval mode for inference
        self.vision_encoder.model.eval()
        self.audio_encoder.eval()
        self.pressure_encoder.eval()
        self.emg_encoder.eval()
        
        logger.info("✓ Encoders initialized")
    
    def _init_preprocessors(self):
        """Initialize preprocessors for each modality"""
        logger.info("Initializing preprocessors...")
        
        # Vision: 1080p -> 224x224 (CLIP input size)
        self.vision_preprocessor = VisionPreprocessor(
            image_size=(224, 224),
            normalize=True
        )
        
        # Audio: 2.5 second rolling buffer @ 16 kHz
        self.audio_preprocessor = AudioPreprocessor(
            sample_rate=self.audio_sample_rate,
            duration=self.audio_buffer_duration,
            normalize=True
        )
        
        logger.info("✓ Preprocessors initialized")
    
    def _init_fusion_and_controller(self):
        """Initialize fusion module and robotic controller"""
        logger.info("Initializing fusion and controller...")
        
        self.fusion = MultimodalFusion(
            modality_dims={
                'vision': 512,
                'audio': 768,
                'pressure': 256,
                'emg': 256
            },
            fusion_dim=512,
            fusion_method="weighted_sum"
        )
        self.fusion.to(self.device)
        self.fusion.eval()
        
        self.controller = RoboticArmController3DOF()
        
        logger.info("✓ Fusion and controller initialized")
    
    def _init_arduino(self, arduino_cfg: dict):
        """Initialize Arduino connection"""
        logger.info("Initializing Arduino connection...")
        
        port = arduino_cfg.get("port")
        baud_rate = arduino_cfg.get("baud_rate", 115200)
        
        self.arduino = ArduinoController(
            port=port,
            baud_rate=baud_rate,
            timeout=arduino_cfg.get("timeout", 0.1),
            write_delay=arduino_cfg.get("write_delay", 0.01)
        )
        
        if self.arduino.connect():
            logger.info(f"✓ Arduino connected: {self.arduino.get_status()}")
        else:
            logger.warning("⚠ Arduino connection failed (will run in simulation mode)")
    
    def _init_buffers(self):
        """Initialize data buffers"""
        logger.info("Initializing buffers...")
        
        # Audio buffer: 2.5 seconds @ 16 kHz
        self.audio_samples = int(self.audio_buffer_duration * self.audio_sample_rate)
        self.audio_buffer = deque(maxlen=self.audio_samples)
        
        # Sensor buffer for pressure and EMG
        self.sensor_buffer = SensorBuffer(size=1000)
        
        # Frame stats
        self.frame_count = 0
        self.start_time = time.time()
        
        logger.info(f"✓ Buffers initialized (audio: {self.audio_samples} samples)")
    
    def run(self, webcam_id: int = 0, duration: Optional[float] = None):
        """
        Run the streaming pipeline
        
        Args:
            webcam_id: Webcam device ID (usually 0 for default camera)
            duration: Duration to run in seconds (None = infinite)
        """
        self.running = True
        cap = cv2.VideoCapture(webcam_id)
        
        if not cap.isOpened():
            logger.error("Failed to open webcam")
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_resolution[1])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_resolution[0])
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        logger.info(f"Webcam opened: {self.video_resolution[1]}x{self.video_resolution[0]} @ {self.target_fps} Hz")
        
        # Start audio capture thread
        audio_thread = threading.Thread(
            target=self._audio_capture_thread,
            daemon=True
        )
        audio_thread.start()
        
        # Start sensor read thread
        sensor_thread = threading.Thread(
            target=self._sensor_read_thread,
            daemon=True
        )
        sensor_thread.start()
        
        logger.info("=" * 60)
        logger.info("STREAMING PIPELINE STARTED")
        logger.info("=" * 60)
        
        try:
            while self.running:
                # Control timing
                loop_start = time.time()
                
                # Capture video frame
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    break
                
                # Process frame through pipeline
                self._process_frame(frame)
                
                self.frame_count += 1
                
                # Check duration
                if duration and (time.time() - self.start_time) > duration:
                    logger.info(f"Duration limit reached ({duration}s)")
                    break
                
                # Maintain target FPS
                loop_time = time.time() - loop_start
                if loop_time < self.frame_time:
                    time.sleep(self.frame_time - loop_time)
        
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
        
        finally:
            self.stop(cap)
    
    def _process_frame(self, frame: np.ndarray):
        """
        Process single frame through complete pipeline
        
        Args:
            frame: Video frame (BGR, 1080p)
        """
        with torch.no_grad():
            # 1. Preprocess vision
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            vision_tensor = self.vision_preprocessor.preprocess(frame_rgb)
            vision_tensor = vision_tensor.unsqueeze(0).to(self.device)
            
            # 2. Encode vision
            vision_emb = self.vision_encoder(vision_tensor)
            
            # 3. Encode audio (if buffer ready)
            audio_emb = self._encode_audio()
            
            # 4. Encode sensors (if data available)
            pressure_emb, emg_emb = self._encode_sensors()
            
            # 5. Fuse modalities
            if audio_emb is not None and pressure_emb is not None:
                fused = self.fusion({
                    'vision': vision_emb,
                    'audio': audio_emb,
                    'pressure': pressure_emb,
                    'emg': emg_emb
                })
            elif pressure_emb is not None:
                # Fallback: use vision only (or handle gracefully)
                fused = self.fusion({'vision': vision_emb, 'pressure': pressure_emb})
            else:
                fused = vision_emb  # Minimal fallback
            
            # 6. Decode to robot commands
            result = self.controller.decode(fused)
            joint_angles = result['position'].cpu().numpy()[0]
            gripper_force = result['force'].cpu().item()
            
            # 7. Send to Arduino
            self._send_robot_command(joint_angles, gripper_force)
            
            # 8. Log stats periodically
            if self.frame_count % 60 == 0:  # Log every 60 frames (~1 second at 60 Hz)
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed
                logger.info(
                    f"Frame {self.frame_count} | FPS: {fps:.1f} | "
                    f"Joint angles: ({joint_angles[0]:.1f}°, {joint_angles[1]:.1f}°, {joint_angles[2]:.1f}°) | "
                    f"pressure: {pressure_emb}"
                    f"Gripper: {gripper_force:.1f}%"
                )
    
    def _encode_audio(self) -> Optional[torch.Tensor]:
        """Encode audio from buffer"""
        if len(self.audio_buffer) < self.audio_samples // 2:  # Need at least 50% buffer
            return None
        
        audio_np = np.array(list(self.audio_buffer))
        audio_tensor = torch.from_numpy(audio_np).float().unsqueeze(0).to(self.device)
        
        audio_emb = self.audio_encoder(audio_tensor)
        return audio_emb.mean(dim=1) if audio_emb.dim() > 2 else audio_emb
    
    def _encode_sensors(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:

        """Encode pressure and EMG sensor data"""
        snapshot = self.sensor_buffer.get_snapshot()
        
        if snapshot is None:
            return None, None
        
        pressure, emg1, emg2, emg3 = snapshot
        
        try:
            # Extract temporal features
            pressure_features = self.pressure_encoder.extract_temporal_features(
                torch.from_numpy(pressure).float().unsqueeze(0)
            )
            emg_features = torch.cat([
                self.emg_encoder.extract_temporal_features(
                    torch.from_numpy(e).float().unsqueeze(0)
                )
                for e in [emg1, emg2, emg3]
            ], dim=1)
            
            # Encode through neural networks
            pressure_emb = self.pressure_encoder(pressure_features.to(self.device))
            emg_emb = self.emg_encoder(emg_features.to(self.device))
            
            return pressure_emb, emg_emb
        
        except Exception as e:
            logger.debug(f"Sensor encoding error: {e}")
            return None, None
    
    def _send_robot_command(
        self,
        joint_angles: np.ndarray,
        gripper_force: float
    ):
        """Send command to Arduino"""
        if self.arduino.connected:
            self.arduino.send_command(
                angle1=joint_angles[0],
                angle2=joint_angles[1],
                angle3=joint_angles[2],
                gripper_force=gripper_force
            )
    
    def _audio_capture_thread(self):
        """Capture audio in background thread"""
        try:
            import sounddevice as sd
        except ImportError:
            logger.warning("sounddevice not installed, audio capture disabled")
            return
        
        # Audio capture configuration
        block_size = 1024
        channels = 1
        
        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.debug(f"Audio capture status: {status}")
            # Add audio samples to buffer
            audio_data = indata[:, 0]
            self.audio_buffer.extend(audio_data)
        
        try:
            with sd.InputStream(
                samplerate=self.audio_sample_rate,
                channels=channels,
                blocksize=block_size,
                callback=audio_callback,
                latency='low'
            ):
                while self.running:
                    time.sleep(0.01)
        except Exception as e:
            logger.error(f"Audio capture error: {e}")
    
    def _sensor_read_thread(self):
        """Read Arduino sensors in background thread"""
        while self.running:
            sensor_data = self.arduino.read_sensors()
            if sensor_data:
                self.sensor_buffer.append(sensor_data)
            
            time.sleep(0.01)  # 100 Hz sensor read rate
    
    def stop(self, cap=None):
        """Cleanup and shutdown"""
        self.running = False
        
        if cap:
            cap.release()
        
        if self.arduino.connected:
            self.arduino.disconnect()
        
        elapsed = time.time() - self.start_time
        logger.info("=" * 60)
        logger.info(f"PIPELINE STOPPED")
        logger.info(f"Total frames: {self.frame_count}")
        logger.info(f"Duration: {elapsed:.1f}s")
        logger.info(f"Average FPS: {self.frame_count / elapsed:.1f}")
        logger.info("=" * 60)


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Stream multimodal inputs to robotic arm via Arduino"
    )
    parser.add_argument(
        "--config",
        default="config/streaming_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (cuda or cpu)"
    )
    parser.add_argument(
        "--webcam",
        type=int,
        default=0,
        help="Webcam device ID"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration to run in seconds (None = infinite)"
    )
    
    args = parser.parse_args()
    
    controller = StreamingRobotController(
        config_path=args.config,
        device=args.device
    )
    
    controller.run(
        webcam_id=args.webcam,
        duration=args.duration
    )


if __name__ == "__main__":
    main()
