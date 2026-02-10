"""
Streaming Robot Control - Mock Hardware Demo
Tests the full pipeline with simulated Arduino without real hardware
"""

import sys
import os
import cv2
import torch
import numpy as np
import time
import logging
from pathlib import Path
from typing import Optional

# Add repo root to path
sys.path.insert(0, os.path.dirname(__file__))

from encoders.vision_encoder import VisionEncoder
from encoders.audio_encoder_learnable import AudioEncoder as LearnableAudioEncoder
from encoders.sensor_encoder import PressureSensorEncoder, EMGSensorEncoder
from fusion.multimodal_fusion import MultimodalFusion
from robotic_arm_controller import RoboticArmController3DOF
from preprocessing.preprocessor import VisionPreprocessor, AudioPreprocessor

# Avoid conflict with built-in 'io' module
io_path = os.path.join(os.path.dirname(__file__), 'io')
sys.path.insert(0, io_path)
from mock_arduino_controller import MockArduinoController


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class MockStreamingDemo:
    """
    Demonstrates multimodal streaming pipeline using mock hardware
    
    Captures video and audio, processes through ML pipeline,
    and sends simulated commands to mock Arduino
    """
    
    def __init__(
        self,
        duration: float = 30.0,
        target_fps: int = 60,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize demo
        
        Args:
            duration: How long to run (seconds)
            target_fps: Target frames per second
            device: torch device
        """
        self.duration = duration
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        self.device = device
        
        logger.info("=" * 70)
        logger.info("MOCK STREAMING ROBOT CONTROL DEMO")
        logger.info("=" * 70)
        logger.info(f"Duration: {duration}s | Target FPS: {target_fps} | Device: {device}\n")
        
        # Initialize components
        self._init_encoders()
        self._init_preprocessors()
        self._init_fusion_and_controller()
        self._init_mock_arduino()
        self._init_buffers()
    
    def _init_encoders(self):
        """Initialize ML encoders"""
        logger.info("Initializing encoders...")
        
        self.vision_encoder = VisionEncoder(device=self.device)
        self.audio_encoder = LearnableAudioEncoder(device=self.device)
        self.pressure_encoder = PressureSensorEncoder(output_dim=256)
        self.emg_encoder = EMGSensorEncoder(output_dim=256, num_channels=3)
        
        self.pressure_encoder.to(self.device)
        self.emg_encoder.to(self.device)
        
        for model in [self.vision_encoder.model, self.audio_encoder, 
                     self.pressure_encoder, self.emg_encoder]:
            model.eval()
        
        logger.info("✓ Encoders ready\n")
    
    def _init_preprocessors(self):
        """Initialize preprocessors"""
        self.vision_preprocessor = VisionPreprocessor(
            image_size=(224, 224),
            normalize=True
        )
        
        self.audio_preprocessor = AudioPreprocessor(
            sample_rate=16000,
            duration=2.5,
            normalize=True
        )
    
    def _init_fusion_and_controller(self):
        """Initialize fusion and controller"""
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
        self.fusion.to(self.device).eval()
        
        self.controller = RoboticArmController3DOF()
        
        logger.info("✓ Fusion and controller ready\n")
    
    def _init_mock_arduino(self):
        """Initialize mock Arduino with simulated hardware"""
        logger.info("Initializing Mock Arduino...")
        
        self.arduino = MockArduinoController(noise_level=0.1)
        self.arduino.connect()
        
        logger.info(f"✓ {self.arduino.get_status()}\n")
    
    def _init_buffers(self):
        """Initialize data buffers"""
        from collections import deque
        
        # Audio buffer: 2.5 seconds @ 16 kHz
        self.audio_samples = int(2.5 * 16000)
        self.audio_buffer = deque(maxlen=self.audio_samples)
        
        # Frame tracking
        self.frame_count = 0
        self.start_time = time.time()
    
    def run(self, webcam_ids: list = None, save_video: bool = False):
        """
        Run the demo with single or dual cameras
        
        Args:
            webcam_ids: List of camera IDs [primary, secondary] or single ID (e.g., [0] or [0, 1])
            save_video: Whether to save output video
        """
        if webcam_ids is None:
            webcam_ids = [0]
        elif isinstance(webcam_ids, int):
            webcam_ids = [webcam_ids]
        
        # Open primary camera
        cap_primary = cv2.VideoCapture(webcam_ids[0])
        if not cap_primary.isOpened():
            logger.error(f"Failed to open primary camera (ID: {webcam_ids[0]})")
            return
        
        # Open secondary camera if specified
        cap_secondary = None
        if len(webcam_ids) > 1:
            cap_secondary = cv2.VideoCapture(webcam_ids[1])
            if not cap_secondary.isOpened():
                logger.warning(f"Failed to open secondary camera (ID: {webcam_ids[1]}), using primary only")
                cap_secondary = None
        
        # Set camera properties
        cap_primary.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap_primary.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap_primary.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        if cap_secondary:
            cap_secondary.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap_secondary.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap_secondary.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        # Video writer for output
        writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                'demo_output.mp4',
                fourcc,
                self.target_fps,
                (1920, 1080)
            )
        
        logger.info("=" * 70)
        camera_info = "Dual cameras" if cap_secondary else "Single camera"
        logger.info(f"DEMO RUNNING ({camera_info}) - Press 'q' to quit")
        logger.info("=" * 70 + "\n")
        
        try:
            while self.frame_count < int(self.duration * self.target_fps):
                loop_start = time.time()
                
                # Capture from primary camera
                ret_primary, frame_primary = cap_primary.read()
                if not ret_primary:
                    logger.warning("Failed to capture frame from primary camera")
                    break
                
                # Capture from secondary camera if available
                frame_secondary = None
                if cap_secondary:
                    ret_secondary, frame_secondary = cap_secondary.read()
                    if not ret_secondary:
                        logger.debug("Failed to capture from secondary camera, using primary only")
                        frame_secondary = None
                
                # Process frames
                frame_output = self._process_dual_frames(frame_primary.copy(), frame_secondary.copy() if frame_secondary is not None else None)
                
                # Display
                cv2.imshow('Mock Streaming Demo', frame_output)
                
                # Save
                if writer:
                    writer.write(frame_output)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                self.frame_count += 1
                
                # Maintain FPS
                loop_time = time.time() - loop_start
                if loop_time < self.frame_time:
                    time.sleep(self.frame_time - loop_time)
        
        finally:
            self._cleanup(cap_primary, writer, cap_secondary)
    
    def _process_dual_frames(self, frame_primary: np.ndarray, frame_secondary: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Process dual frames through ML pipeline with vision fusion
        
        Args:
            frame_primary: Primary video frame (BGR, 1080p)
            frame_secondary: Secondary video frame (BGR, 1080p) or None
        
        Returns:
            Annotated frame with overlay info
        """
        with torch.no_grad():
            # 1. Preprocess and encode primary vision
            frame_rgb = cv2.cvtColor(frame_primary, cv2.COLOR_BGR2RGB)
            vision_tensor = self.vision_preprocessor.preprocess(frame_rgb)
            vision_tensor = vision_tensor.unsqueeze(0).to(self.device)
            vision_emb_primary = self.vision_encoder(vision_tensor)
            
            # 2. Preprocess and encode secondary vision (if available)
            if frame_secondary is not None:
                frame_rgb_sec = cv2.cvtColor(frame_secondary, cv2.COLOR_BGR2RGB)
                vision_tensor_sec = self.vision_preprocessor.preprocess(frame_rgb_sec)
                vision_tensor_sec = vision_tensor_sec.unsqueeze(0).to(self.device)
                vision_emb_secondary = self.vision_encoder(vision_tensor_sec)
                
                # Average both vision embeddings
                vision_emb = (vision_emb_primary + vision_emb_secondary) / 2.0
            else:
                vision_emb = vision_emb_primary
            
            # 3. Generate dummy audio (in real scenario, this comes from live audio capture)
            dummy_audio = np.random.randn(2, 40000).astype(np.float32)
            audio_tensor = torch.from_numpy(dummy_audio).to(self.device)
            audio_emb = self.audio_encoder(audio_tensor)
            
            # 4. Read simulated sensors from mock Arduino
            sensor_data = self.arduino.read_sensors()
            if sensor_data:
                pressure_features = torch.randn(1, 100).to(self.device)
                emg_features = torch.randn(1, 100).to(self.device)
                
                pressure_emb = self.pressure_encoder(pressure_features)
                emg_emb = self.emg_encoder(emg_features)
            else:
                pressure_emb = torch.randn(1, 256).to(self.device)
                emg_emb = torch.randn(1, 256).to(self.device)
            
            # 5. Fuse modalities
            fused = self.fusion({
                'vision': vision_emb,
                'audio': audio_emb,
                'pressure': pressure_emb,
                'emg': emg_emb
            })
            
            # 6. Decode to robot commands
            result = self.controller.decode(fused)
            position = result['position'].cpu().numpy()[0]
            gripper_force = result['force'].cpu()[0].item()
            
            # 7. Send to mock Arduino
            self.arduino.send_command(
                angle1=position[0] * 90,
                angle2=position[1] * 90,
                angle3=position[2] * 180,
                gripper_force=gripper_force
            )
        
        # Annotate frame with pipeline info
        frame = self._annotate_frame(frame, position, gripper_force, sensor_data)
        
        return frame
    
    def _annotate_frame(
        self,
        frame: np.ndarray,
        position: np.ndarray,
        gripper_force: float,
        sensor_data: dict
    ) -> np.ndarray:
        """Add text overlay to frame"""
        
        # Frame info
        elapsed = time.time() - self.start_time
        fps = self.frame_count / (elapsed + 1e-6)
        
        h, w = frame.shape[:2]
        
        # Background for text
        cv2.rectangle(frame, (10, 10), (500, 280), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (500, 280), (0, 255, 0), 2)
        
        # Text
        y = 35
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (0, 255, 0)
        thickness = 1
        
        cv2.putText(frame, f"Frame: {self.frame_count} | FPS: {fps:.1f}", (20, y), font, font_scale, color, thickness)
        y += 30
        
        cv2.putText(frame, "Robot Position (m):", (20, y), font, font_scale, color, thickness)
        y += 25
        cv2.putText(frame, f"  X={position[0]:6.3f}  Y={position[1]:6.3f}  Z={position[2]:6.3f}", (30, y), font, font_scale, color, thickness)
        y += 30
        
        cv2.putText(frame, f"Gripper Force: {gripper_force:.1f}%", (20, y), font, font_scale, color, thickness)
        y += 30
        
        if sensor_data:
            cv2.putText(frame, f"Pressure: {sensor_data['pressure']:.0f}", (20, y), font, font_scale, color, thickness)
            y += 25
            cv2.putText(frame, f"EMG: {sensor_data['emg_1']:.0f}, {sensor_data['emg_2']:.0f}, {sensor_data['emg_3']:.0f}", (20, y), font, font_scale, color, thickness)
        
        # Arduino status (bottom right)
        status = self.arduino.get_status()
        status_y = h - 30
        cv2.putText(frame, status, (10, status_y), font, font_scale, (0, 255, 0), 1)
        
        return frame
    
    def _cleanup(self, cap_primary, writer, cap_secondary=None):
        """Clean up resources"""
        cap_primary.release()
        if cap_secondary:
            cap_secondary.release()
        
        if writer:
            writer.release()
        
        cv2.destroyAllWindows()
        self.arduino.disconnect()
        
        elapsed = time.time() - self.start_time
        
        logger.info("\n" + "=" * 70)
        logger.info("DEMO FINISHED")
        logger.info("=" * 70)
        logger.info(f"Total frames: {self.frame_count}")
        logger.info(f"Duration: {elapsed:.1f}s")
        logger.info(f"Average FPS: {self.frame_count / elapsed:.1f}")
        logger.info(f"Total commands: {self.arduino.command_count}")
        logger.info("=" * 70)


def main():
    """Run mock streaming demo"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Mock streaming robot control demo (no hardware required)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Duration to run (seconds)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target FPS (default 30, can go up to 60)"
    )
    parser.add_argument(
        "--webcam",
        type=int,
        nargs="+",
        default=[0],
        help="Webcam device ID(s) - single ID or two IDs for dual camera mode"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (cuda or cpu)"
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save output to demo_output.mp4"
    )
    
    args = parser.parse_args()
    
    demo = MockStreamingDemo(
        duration=args.duration,
        target_fps=args.fps,
        device=args.device
    )
    
    demo.run(
        webcam_ids=args.webcam,
        save_video=args.save_video
    )


if __name__ == "__main__":
    main()
