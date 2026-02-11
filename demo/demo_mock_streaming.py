"""
Streaming Robot Control - Mock Hardware Demo
Tests the full pipeline with simulated Arduino without real hardware
"""

import sys
import os

# Configure Qt / OpenCV environment early to avoid Wayland / plugin errors
# If the user is on Wayland but OpenCV only ships the 'xcb' plugin, prefer 'xcb'.
# For headless environments, fall back to 'offscreen'.
if 'QT_QPA_PLATFORM' not in os.environ:
    # Prefer platform based on environment, but avoid selecting 'wayland' unless plugin is available
    if os.environ.get('XDG_SESSION_TYPE') == 'wayland' or 'WAYLAND_DISPLAY' in os.environ:
        # Many OpenCV builds include only the 'xcb' plugin; use it to prevent plugin load failures
        os.environ['QT_QPA_PLATFORM'] = 'xcb'
    elif 'DISPLAY' in os.environ:
        os.environ['QT_QPA_PLATFORM'] = 'xcb'
    else:
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Reduce OpenCV verbose logs
os.environ.setdefault('OPENCV_LOG_LEVEL', 'ERROR')
# Point Qt to a common font directory if available (adjust for your system)
os.environ.setdefault('QT_QPA_FONTDIR', '/usr/share/fonts/truetype/dejavu')

import cv2
import sounddevice as sd
import torch
import numpy as np
import time
import logging
from pathlib import Path
from typing import Optional

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ..ctrl.robotic_arm_controller import RoboticArmController3DOF
from encoders.vision_encoder import VisionEncoder
from encoders.audio_encoder_learnable import AudioEncoder as LearnableAudioEncoder
from encoders.sensor_encoder import PressureSensorEncoder, EMGSensorEncoder
from fusion.multimodal_fusion import MultimodalFusion
from encoders.asr_vosk import VoskASR, StreamingVosk
from preprocessing.preprocessor import VisionPreprocessor, AudioPreprocessor

# Avoid conflict with built-in 'io' module
io_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'io')
sys.path.insert(0, io_path)
from mock_arduino_controller import create_arduino_controller


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
        duration: float = 15.0,
        target_fps: int = 60,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_mic: bool = False,
        asr_callback=None,
        mic_device: str = None
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
        self.use_mic = use_mic
        # Optional callback(audio_tensor: torch.Tensor) -> str transcript
        self.asr_callback = asr_callback
        self.latest_transcript = ""
        self.mic_device = mic_device
        self._mic_error = False
        self._mic_error_message = ""
        self.latest_rms = 0.0
        
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
        # Use factory which will consult config/streaming_config.yaml
        # to decide between mock and real Arduino and to obtain serial settings.
        self.arduino = create_arduino_controller(noise_level=0.1)
        self.arduino.connect()
        
        logger.info(f"✓ {self.arduino.get_status()}\n")
    
    def _init_buffers(self):
        """Initialize data buffers"""
        from collections import deque
        
        # Audio buffer: 2.5 seconds @ 16 kHz
        self.audio_samples = int(2.5 * 16000)
        self.audio_buffer = deque(maxlen=self.audio_samples)

        # If using microphone, start input stream to fill audio buffer
        self._sd_stream = None
        if self.use_mic:
            try:
                # Log available devices for troubleshooting
                try:
                    devs = sd.query_devices()
                    logger.info(f"Available audio devices: {len(devs)}")
                except Exception:
                    logger.debug("Could not query audio devices via sounddevice")
                def _callback(indata, frames, time_info, status):
                    # indata: (frames, channels)
                    samples = indata[:, 0]
                    self.audio_buffer.extend(samples.tolist())
                # Determine samplerate to use for InputStream. Some devices do not support 16000.
                desired_sr = 16000
                sd_samplerate = desired_sr
                try:
                    dev_info = None
                    # If a device was provided, try to resolve it. Accept integer index or substring of device name.
                    if self.mic_device is not None:
                        # try integer index first
                        try:
                            dev_idx = int(self.mic_device)
                            dev_info = sd.query_devices(dev_idx)
                        except Exception:
                            # string provided: match against available device names (case-insensitive)
                            try:
                                all_devs = sd.query_devices()
                                matches = []
                                query_lower = str(self.mic_device).lower()
                                for i, d in enumerate(all_devs):
                                    name = d.get('name', '') if isinstance(d, dict) else d[0]
                                    if query_lower in str(name).lower() and d.get('max_input_channels', 0) > 0:
                                        matches.append((i, d))
                                if len(matches) == 0:
                                    logger.warning(f"No audio device matched '{self.mic_device}'. Available devices will be listed for debugging.")
                                    for i, d in enumerate(all_devs):
                                        logger.info(f"[{i}] {d.get('name', '')} (inputs={d.get('max_input_channels', 0)})")
                                else:
                                    # pick first match
                                    dev_idx, dev_info = matches[0]
                                    logger.info(f"Selected microphone device index {dev_idx} matching '{self.mic_device}' -> {dev_info.get('name')}")
                            except Exception:
                                dev_info = None
                    else:
                        # use default input device index if available
                        default_dev = sd.default.device[0] if sd.default.device else None
                        if default_dev is not None and default_dev != -1:
                            dev_info = sd.query_devices(default_dev)
                        else:
                            dev_info = None

                    if isinstance(dev_info, dict) and 'default_samplerate' in dev_info:
                        sd_samplerate = int(dev_info['default_samplerate'])
                except Exception:
                    sd_samplerate = desired_sr

                sd_kwargs = dict(samplerate=sd_samplerate, channels=1, callback=_callback, blocksize=1024)
                if self.mic_device is not None:
                    # mic_device may be an index string or a device name; prefer integer index when possible
                    try:
                        sd_kwargs['device'] = int(self.mic_device)
                    except Exception:
                        # try to find matching device index by name
                        try:
                            all_devs = sd.query_devices()
                            query_lower = str(self.mic_device).lower()
                            matched_idx = None
                            for i, d in enumerate(all_devs):
                                name = d.get('name', '')
                                if query_lower in str(name).lower() and d.get('max_input_channels', 0) > 0:
                                    matched_idx = i
                                    break
                            if matched_idx is not None:
                                sd_kwargs['device'] = matched_idx
                            else:
                                sd_kwargs['device'] = self.mic_device
                        except Exception:
                            sd_kwargs['device'] = self.mic_device

                self._sd_stream = sd.InputStream(**sd_kwargs)
                self._sd_stream.start()
                # store actual samplerate for later resampling
                self._sd_samplerate = sd_samplerate
                logger.info(f"Microphone input stream started (device={self.mic_device}, samplerate={sd_samplerate})")
            except Exception as e:
                self._mic_error = True
                self._mic_error_message = str(e)
                logger.warning(f"Could not start microphone input: {e}")
        
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

        # Create resizable OpenCV windows so user can resize camera displays
        # Use WINDOW_NORMAL to allow manual resizing
        cv2.namedWindow('Mock Streaming Demo - Camera 1', cv2.WINDOW_NORMAL)
        # sensible default initial window size (half-FHD)
        try:
            cv2.resizeWindow('Mock Streaming Demo - Camera 1', 960, 540)
        except Exception:
            pass
        if cap_secondary:
            cv2.namedWindow('Mock Streaming Demo - Camera 2', cv2.WINDOW_NORMAL)
            try:
                cv2.resizeWindow('Mock Streaming Demo - Camera 2', 960, 540)
            except Exception:
                pass
        
        # Determine output frame size for video writer
        output_width = 1920 if cap_secondary is None else 1920 * 2
        output_height = 1080
        
        # Video writer for output
        writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                'demo_output.mp4',
                fourcc,
                self.target_fps,
                (output_width, output_height)
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
                annotated_primary, annotated_secondary = self._process_dual_frames(
                    frame_primary.copy(), frame_secondary.copy() if frame_secondary is not None else None)

                # Display each camera in its own window
                cv2.imshow('Mock Streaming Demo - Camera 1', annotated_primary)
                if annotated_secondary is not None:
                    cv2.imshow('Mock Streaming Demo - Camera 2', annotated_secondary)
                cv2.waitKey(1)

                # For video writer, concatenate if both exist
                if annotated_secondary is not None:
                    frame_output = np.concatenate([annotated_primary, annotated_secondary], axis=1)
                else:
                    frame_output = annotated_primary
                
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
    
    def _process_dual_frames(self, frame_primary: np.ndarray, frame_secondary: Optional[np.ndarray] = None):
        """
        Process dual frames through ML pipeline with vision fusion
        Returns annotated frames for both cameras (second may be None)
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
            
            # 3. Get audio: either from microphone buffer or dummy audio
            if getattr(self, 'use_mic', False):
                # Build latest audio buffer of expected length
                buf = list(self.audio_buffer)
                if len(buf) == 0:
                    # no mic data yet
                    audio_np = np.zeros(self.audio_samples, dtype=np.float32)
                    self.latest_rms = 0.0
                else:
                    if len(buf) < self.audio_samples:
                        # pad with zeros
                        padding = [0.0] * (self.audio_samples - len(buf))
                        buf = padding + buf
                    else:
                        buf = buf[-self.audio_samples:]
                    audio_np = np.array(buf, dtype=np.float32)
                    # compute RMS for overlay
                    self.latest_rms = float(np.sqrt((audio_np ** 2).mean()))
                # make batch dimension
                # If samplerate differs from desired 16k, resample
                try:
                    sd_sr = getattr(self, '_sd_samplerate', 16000)
                    if sd_sr != 16000:
                        import librosa
                        audio_np = librosa.resample(audio_np, orig_sr=sd_sr, target_sr=16000)
                except Exception:
                    # if resampling fails, use raw audio
                    pass

                audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).to(self.device)
            else:
                # Generate dummy audio (in real scenario, this comes from live audio capture)
                dummy_audio = np.random.randn(1, self.audio_samples).astype(np.float32)
                audio_tensor = torch.from_numpy(dummy_audio).to(self.device)
            audio_emb = self.audio_encoder(audio_tensor)

            # ----- ASR / Transcript handling -----
            transcript = ""
            try:
                if self.asr_callback is not None:
                    # Pass audio tensor (CPU) to callback; callback should return a string
                    at_cpu = audio_tensor.detach().cpu()
                    transcript = self.asr_callback(at_cpu)
                elif self.use_mic:
                    # Indicate listening when mic is used but no ASR provided
                    transcript = "(listening...)"
            except Exception:
                transcript = ""
            # store for overlay
            self.latest_transcript = transcript
            
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
        
        # Annotate both frames side by side for visualization
        annotated_primary = self._annotate_frame(frame_primary, position, gripper_force, sensor_data, transcript=self.latest_transcript)
        annotated_secondary = None
        if frame_secondary is not None:
            annotated_secondary = self._annotate_frame(frame_secondary, position, gripper_force, sensor_data, transcript=self.latest_transcript)
            annotated_secondary = cv2.resize(annotated_secondary, (annotated_primary.shape[1], annotated_primary.shape[0]))
        return annotated_primary, annotated_secondary
    
    def _annotate_frame(
        self,
        frame: np.ndarray,
        position: np.ndarray,
        gripper_force: float,
        sensor_data: dict,
        transcript: str = ""
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

        # Transcript overlay (bottom-left above status)
        if transcript:
            t_y = status_y - 40
            # Wrap long transcripts across multiple lines if needed
            max_width = 40
            words = transcript.split()
            line = ""
            lines = []
            for w in words:
                if len(line) + len(w) + 1 <= max_width:
                    line = (line + " " + w).strip()
                else:
                    lines.append(line)
                    line = w
            if line:
                lines.append(line)
            for i, ln in enumerate(reversed(lines)):
                cv2.putText(frame, ln, (20, t_y - i * 20), font, 0.6, (255, 255, 0), 1)

        # Microphone status / RMS overlay (bottom-left above transcript)
        if getattr(self, 'use_mic', False):
            if getattr(self, '_mic_error', False):
                cv2.putText(frame, f"MIC ERROR: {self._mic_error_message}", (20, status_y - 80), font, 0.6, (0, 0, 255), 1)
            else:
                rms_text = f"Mic RMS: {self.latest_rms:.4f}"
                cv2.putText(frame, rms_text, (20, status_y - 80), font, 0.6, (255, 255, 0), 1)
        
        return frame
    
    def _cleanup(self, cap_primary, writer, cap_secondary=None):
        """Clean up resources"""
        cap_primary.release()
        if cap_secondary:
            cap_secondary.release()
        
        if writer:
            writer.release()
        
        cv2.destroyAllWindows()
        # Close Arduino and ASR resources
        try:
            self.arduino.disconnect()
        except Exception:
            pass
        if hasattr(self, 'asr_instance') and self.asr_instance is not None:
            try:
                self.asr_instance.close()
            except Exception:
                pass
        
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
    parser.add_argument(
        "--use-mic",
        action="store_true",
        help="Use system microphone for live audio input (requires sounddevice)"
    )
    parser.add_argument(
        "--mic-device",
        type=str,
        default=None,
        help="Sounddevice input device name or index (optional)"
    )
    parser.add_argument(
        "--use-asr",
        action="store_true",
        help="Enable Vosk ASR to generate live transcripts (requires --vosk-model)"
    )
    parser.add_argument(
        "--vosk-model",
        type=str,
        default=None,
        help="Path to Vosk model directory (e.g., vosk-model-small-en-us-0.15)"
    )
    
    args = parser.parse_args()
    
    print(f"[DEBUG] Using webcam_ids: {args.webcam}")
    logging.info(f"[DEBUG] Using webcam_ids: {args.webcam}")
    
    asr_cb = None
    if args.use_asr:
        model_path = args.vosk_model
        if not model_path:
            logger.error("--use-asr requires --vosk-model path to be provided")
        else:
            try:
                asr = StreamingVosk(model_path=model_path, sample_rate=16000)
                asr_cb = asr.feed
                logger.info(f"Vosk streaming ASR initialized from {model_path}")
            except Exception as e:
                logger.error(f"Failed to initialize Vosk ASR: {e}")

    demo = MockStreamingDemo(
        duration=args.duration,
        target_fps=args.fps,
        device=args.device,
        use_mic=args.use_mic,
        asr_callback=asr_cb,
        mic_device=args.mic_device
    )
    # attach asr instance for graceful cleanup
    if args.use_asr and 'asr' in locals():
        demo.asr_instance = asr
    
    demo.run(
        webcam_ids=args.webcam,
        save_video=args.save_video
    )


if __name__ == "__main__":
    main()
