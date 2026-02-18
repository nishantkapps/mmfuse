"""
Real-time robotic control using trained sdata model (MultimodalFusionWithAttention + ActionClassifier).

Uses dual cameras, audio, and optional sensors to predict action class and map to robot commands.
"""

import os
import sys
import time
import logging
import threading
from pathlib import Path
from collections import deque
from typing import Optional, Dict, Tuple

import torch
import numpy as np
import cv2
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmfuse.preprocessing.preprocessor import VisionPreprocessor, AudioPreprocessor
from mmfuse.encoders.vision_encoder import VisionEncoder
from mmfuse.encoders.vision_encoder_viscop import VisCoPVisionEncoder
from mmfuse.encoders.audio_encoder_learnable import AudioEncoder as LearnableAudioEncoder
from mmfuse.encoders.audio_encoder import Wav2VecPooledEncoder
from mmfuse.encoders.audio_encoder_whisper import WhisperAudioEncoder
from mmfuse.encoders.sensor_encoder import PressureSensorEncoder, EMGSensorEncoder
from mmfuse.fusion.multimodal_fusion import MultimodalFusionWithAttention
from mmfuse.io.arduino_controller import ArduinoController, SensorBuffer

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ActionClassifier(torch.nn.Module):
    """Classification head: fused embedding -> num_classes."""
    def __init__(self, embedding_dim=512, num_classes=8):
        super().__init__()
        self.fc = torch.nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class SDataRobotController:
    """
    Streaming controller using trained sdata model.
    Predicts action class from vision (cam1, cam2), audio, sensors -> maps to robot commands.
    """
    def __init__(
        self,
        checkpoint_path: str,
        action_config_path: str = "config/sdata_action_config.yaml",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        audio_encoder: Optional[str] = None,
        vision_encoder: Optional[str] = None,
        viscop_model_path: Optional[str] = None,
    ):
        self.device = device
        self.running = False
        self.checkpoint_path = Path(checkpoint_path)
        self.action_config_path = Path(action_config_path)
        self.audio_encoder_type = audio_encoder  # learnable|wav2vec|whisper; None = from checkpoint
        self.vision_encoder_type = vision_encoder  # clip|viscop; None = from checkpoint
        self.viscop_model_path = viscop_model_path or "viscop_trained_models/viscop_qwen2.5_7b_viscop-lora_egocentric-expert"

        self._load_action_mapping()
        self._load_model()
        self._init_preprocessors()
        self._init_arduino()
        self._init_buffers()

        logger.info("SData robot controller initialized")

    def _load_action_mapping(self):
        """Load action index -> (angle1, angle2, angle3, gripper_force) mapping."""
        try:
            with open(self.action_config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            self.action_commands = [
                (a['angle1'], a['angle2'], a['angle3'], a['gripper_force'])
                for a in cfg.get('actions', [])
            ]
        except Exception as e:
            logger.warning(f"Could not load action config: {e}, using defaults")
            self.action_commands = [(0, 0, 90, 0)] * 8

    def _load_model(self):
        """Load trained sdata checkpoint (fusion + classifier)."""
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        num_classes = ckpt.get('num_classes', 8)
        audio_type = self.audio_encoder_type or ckpt.get('audio_encoder', 'learnable')
        vision_type = self.vision_encoder_type or ckpt.get('vision_encoder', 'clip')
        vision_dim = ckpt.get('vision_dim', 512)

        if vision_type == 'viscop':
            vision = VisCoPVisionEncoder(model_path=self.viscop_model_path, device=self.device)
        else:
            vision = VisionEncoder(device=self.device)
        if audio_type == 'wav2vec':
            audio = Wav2VecPooledEncoder(frozen=True, device=self.device)
        elif audio_type == 'whisper':
            audio = WhisperAudioEncoder(frozen=True, device=self.device)
        else:
            audio = LearnableAudioEncoder(device=self.device)
        pressure = PressureSensorEncoder(output_dim=256, input_features=2)
        emg = EMGSensorEncoder(output_dim=256, num_channels=3, input_features=4)

        modality_dims = {
            'vision_camera1': vision_dim,
            'vision_camera2': vision_dim,
            'audio': 768,
            'pressure': 256,
            'emg': 256,
        }
        fusion_dim = ckpt.get('fusion_dim', 512)
        fusion = MultimodalFusionWithAttention(
            modality_dims=modality_dims,
            fusion_dim=fusion_dim,
            num_heads=8,
            dropout=0.2
        )
        classifier = ActionClassifier(embedding_dim=fusion_dim, num_classes=num_classes)

        fusion.load_state_dict(ckpt['fusion_state'])
        classifier.load_state_dict(ckpt['model_state'])

        for m in [vision, audio, pressure, emg, fusion, classifier]:
            m.to(self.device)
            m.eval()

        self.vision_encoder = vision
        self.vision_encoder_type = vision_type
        self.audio_encoder = audio
        self.pressure_encoder = pressure
        self.emg_encoder = emg
        self.fusion = fusion
        self.classifier = classifier
        self.num_classes = num_classes

    def _init_preprocessors(self):
        self.vision_preprocessor = VisionPreprocessor(image_size=(224, 224), normalize=True)
        self.audio_preprocessor = AudioPreprocessor(
            sample_rate=16000,
            duration=2.5,
            normalize=True
        )
        self.audio_buffer_duration = 2.5
        self.audio_sample_rate = 16000
        self.audio_samples = int(self.audio_buffer_duration * self.audio_sample_rate)
        self.audio_buffer = deque(maxlen=self.audio_samples)

    def _init_arduino(self):
        try:
            with open("config/streaming_config.yaml", 'r') as f:
                cfg = yaml.safe_load(f) or {}
        except FileNotFoundError:
            cfg = {}
        arduino_cfg = cfg.get("arduino", {})
        self.arduino = ArduinoController(
            port=arduino_cfg.get("port"),
            baud_rate=arduino_cfg.get("baud_rate", 115200),
            timeout=arduino_cfg.get("timeout", 0.1),
            write_delay=arduino_cfg.get("write_delay", 0.01)
        )
        if self.arduino.connect():
            logger.info("Arduino connected")
        else:
            logger.warning("Arduino not connected (simulation mode)")

    def _init_buffers(self):
        self.sensor_buffer = SensorBuffer(size=1000)
        self.frame_count = 0
        self.start_time = time.time()

    def _encode_frame(self, frame: np.ndarray) -> torch.Tensor:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.vision_encoder_type == 'viscop':
            # VisCoP expects raw RGB (0-1); resize to 224x224
            resized = cv2.resize(frame_rgb, (224, 224))
            t = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            t = t.to(self.device)
        else:
            t = self.vision_preprocessor.preprocess(frame_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.vision_encoder(t)

    def _encode_audio(self) -> Optional[torch.Tensor]:
        if len(self.audio_buffer) < self.audio_samples // 2:
            return None
        audio_np = np.array(list(self.audio_buffer), dtype=np.float32)
        audio_tensor = torch.from_numpy(audio_np).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.audio_encoder(audio_tensor)
        return emb

    def _encode_sensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return pressure and EMG embeddings (zeros when no sensor data)."""
        snapshot = self.sensor_buffer.get_snapshot()
        if snapshot is None:
            p_feat = torch.zeros(1, 2, device=self.device)
            e_feat = torch.zeros(1, 4, device=self.device)
        else:
            pressure, emg1, emg2, emg3 = snapshot
            p_arr = np.array([np.mean(pressure), np.std(pressure) if len(pressure) > 1 else 0.0], dtype=np.float32)
            e_arr = np.array([np.mean(emg1), np.mean(emg2), np.mean(emg3), 0.0], dtype=np.float32)
            p_feat = torch.tensor(p_arr, dtype=torch.float32).unsqueeze(0).to(self.device)
            e_feat = torch.tensor(e_arr, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            p_emb = self.pressure_encoder(p_feat)
            e_emb = self.emg_encoder(e_feat)
        return p_emb, e_emb

    def _predict_action(
        self,
        frame_primary: np.ndarray,
        frame_secondary: Optional[np.ndarray] = None
    ) -> int:
        """Run inference and return predicted action class."""
        with torch.no_grad():
            v_emb1 = self._encode_frame(frame_primary)
            v_emb2 = self._encode_frame(frame_secondary) if frame_secondary is not None else v_emb1

            a_emb = self._encode_audio()
            if a_emb is None:
                a_emb = torch.zeros(1, 768, device=self.device)

            p_emb, e_emb = self._encode_sensors()

            embeddings = {
                'vision_camera1': v_emb1,
                'vision_camera2': v_emb2,
                'audio': a_emb,
                'pressure': p_emb,
                'emg': e_emb,
            }
            fused, _ = self.fusion(embeddings, return_kl=True)
            logits = self.classifier(fused)
            action = logits.argmax(dim=1).item()
        return action

    def _action_to_command(self, action: int) -> Tuple[float, float, float, float]:
        """Map action index to (angle1, angle2, angle3, gripper_force)."""
        if 0 <= action < len(self.action_commands):
            return self.action_commands[action]
        return self.action_commands[0]

    def _send_command(self, angle1: float, angle2: float, angle3: float, gripper_force: float):
        if self.arduino.connected:
            self.arduino.send_command(
                angle1=angle1,
                angle2=angle2,
                angle3=angle3,
                gripper_force=gripper_force
            )

    def _audio_capture_thread(self):
        try:
            import sounddevice as sd
        except ImportError:
            return

        def callback(indata, frames, time_info, status):
            if status:
                logger.debug(f"Audio: {status}")
            self.audio_buffer.extend(indata[:, 0])

        with sd.InputStream(
            samplerate=self.audio_sample_rate,
            channels=1,
            blocksize=1024,
            callback=callback,
            latency='low'
        ):
            while self.running:
                time.sleep(0.01)

    def _sensor_read_thread(self):
        while self.running:
            data = self.arduino.read_sensors()
            if data:
                self.sensor_buffer.append(data)
            time.sleep(0.01)

    def run(
        self,
        webcam_ids: list = None,
        duration: Optional[float] = None
    ):
        """Run streaming pipeline."""
        if webcam_ids is None:
            webcam_ids = [0, 1]

        cap1 = cv2.VideoCapture(webcam_ids[0])
        if not cap1.isOpened():
            logger.error("Failed to open primary camera (ID %s)", webcam_ids[0])
            return
        logger.info("Camera 1 opened (ID %s)", webcam_ids[0])

        cap2 = None
        if len(webcam_ids) > 1:
            time.sleep(0.5)  # Allow first camera to stabilize before opening second
            cap2 = cv2.VideoCapture(webcam_ids[1])
            if cap2.isOpened():
                logger.info("Camera 2 opened (ID %s)", webcam_ids[1])
            else:
                logger.warning("Camera 2 (ID %s) failed to open, using single camera", webcam_ids[1])
                cap2.release()
                cap2 = None

        threading.Thread(target=self._audio_capture_thread, daemon=True).start()
        threading.Thread(target=self._sensor_read_thread, daemon=True).start()

        logger.info("SData robot control started (dual camera)")
        self.running = True

        try:
            while self.running:
                ret1, frame1 = cap1.read()
                if not ret1:
                    break
                ret2, frame2 = cap2.read() if cap2 else (False, None)
                frame2 = frame2 if ret2 and frame2 is not None else None

                action = self._predict_action(frame1, frame2)
                angle1, angle2, angle3, gripper = self._action_to_command(action)
                self._send_command(angle1, angle2, angle3, gripper)

                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed
                    action_name = f"part{action + 1}" if action < 8 else f"action_{action}"
                    logger.info(f"Frame {self.frame_count} | FPS: {fps:.1f} | Action: {action} ({action_name}) | "
                               f"Angles: ({angle1:.0f}, {angle2:.0f}, {angle3:.0f}) | Gripper: {gripper:.0f}")

                if duration and (time.time() - self.start_time) > duration:
                    break
                time.sleep(0.016)
        except KeyboardInterrupt:
            logger.info("Interrupted")
        finally:
            self.running = False
            cap1.release()
            if cap2:
                cap2.release()
            if self.arduino.connected:
                self.arduino.disconnect()
            logger.info("SData robot control stopped")


def main():
    import argparse
    import glob
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', help='Path to ckpt_sdata_epoch_N.pt (or dir to auto-pick latest)')
    p.add_argument('--action-config', default='config/sdata_action_config.yaml')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--audio-encoder', choices=['learnable', 'wav2vec', 'whisper'], default=None,
                   help='Overrides checkpoint; use same as training (default: from checkpoint)')
    p.add_argument('--vision-encoder', choices=['clip', 'viscop'], default=None,
                   help='Overrides checkpoint (default: from checkpoint)')
    p.add_argument('--viscop-model-path', default=None,
                   help='Path to VisCoP model (when --vision-encoder viscop)')
    p.add_argument('--webcam', type=int, nargs='+', default=[0, 1])
    p.add_argument('--duration', type=float, default=None)
    args = p.parse_args()

    ckpt_path = args.checkpoint
    if not ckpt_path:
        # Default: checkpoints/ckpt_sdata_epoch_*.pt
        candidates = sorted(glob.glob('checkpoints/ckpt_sdata_epoch_*.pt'))
        if not candidates:
            candidates = sorted(glob.glob('../checkpoints/ckpt_sdata_epoch_*.pt'))
        ckpt_path = candidates[-1] if candidates else None
    if not ckpt_path or not os.path.exists(ckpt_path):
        logger.error("No checkpoint found. Use --checkpoint path/to/ckpt_sdata_epoch_N.pt")
        return

    controller = SDataRobotController(
        checkpoint_path=ckpt_path,
        action_config_path=args.action_config,
        device=args.device,
        audio_encoder=args.audio_encoder,
        vision_encoder=args.vision_encoder,
        viscop_model_path=args.viscop_model_path,
    )
    controller.run(webcam_ids=args.webcam, duration=args.duration)


if __name__ == '__main__':
    main()
