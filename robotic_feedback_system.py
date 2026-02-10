"""
Robotic Multimodal Feedback System
Integrates all encoders and fusion for end-to-end multimodal processing
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union
import numpy as np

from encoders.vision_encoder import VisionEncoder
from encoders.audio_encoder import AudioEncoder
from encoders.sensor_encoder import PressureSensorEncoder, EMGSensorEncoder
from fusion.multimodal_fusion import MultimodalFusion, MultimodalFusionWithAttention
from preprocessing.preprocessor import (
    VisionPreprocessor,
    AudioPreprocessor,
    SensorPreprocessor
)


class RoboticFeedbackSystem(nn.Module):
    """
    Complete multimodal feedback system for robotic applications
    
    Takes inputs from:
    - 2 Cameras (vision)
    - 1 Audio input
    - 1 Pressure sensor
    - 1 EMG sensor
    
    Encodes each modality independently using pre-trained models,
    then fuses them into a unified representation.
    """
    
    def __init__(
        self,
        fusion_dim: int = 512,
        fusion_method: str = "concat_project",
        use_attention: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize Robotic Feedback System
        
        Args:
            fusion_dim: Target dimension for fused embeddings
            fusion_method: How to fuse modalities
            use_attention: Whether to use attention-based fusion
            device: Device to load models on
        """
        super().__init__()
        self.device = device
        self.fusion_dim = fusion_dim
        
        # ===== Initialize Encoders =====
        
        # Vision: CLIP encoder for 2 cameras
        self.vision_encoder = VisionEncoder(
            model_name="ViT-B/32",
            frozen=True,
            device=device
        )
        vision_dim = self.vision_encoder.output_dim
        
        # Audio: Wav2Vec 2.0 encoder
        self.audio_encoder = AudioEncoder(
            model_name="facebook/wav2vec2-base",
            frozen=True,
            device=device,
            sampling_rate=16000
        )
        audio_dim = self.audio_encoder.output_dim
        
        # Pressure Sensor: Small neural network encoder
        self.pressure_encoder = PressureSensorEncoder(
            output_dim=256,
            num_channels=1
        ).to(device)
        pressure_dim = self.pressure_encoder.output_dim_value
        
        # EMG Sensor: Small neural network encoder
        self.emg_encoder = EMGSensorEncoder(
            output_dim=256,
            num_channels=8  # Typical EMG has 8 channels
        ).to(device)
        emg_dim = self.emg_encoder.output_dim_value
        
        # ===== Initialize Fusion =====
        modality_dims = {
            'vision': vision_dim,
            'audio': audio_dim,
            'pressure': pressure_dim,
            'emg': emg_dim
        }
        
        if use_attention:
            self.fusion_module = MultimodalFusionWithAttention(
                modality_dims=modality_dims,
                fusion_dim=fusion_dim,
                num_heads=8,
                dropout=0.2
            ).to(device)
        else:
            self.fusion_module = MultimodalFusion(
                modality_dims=modality_dims,
                fusion_dim=fusion_dim,
                fusion_method=fusion_method,
                dropout=0.2
            ).to(device)
        
        # ===== Initialize Preprocessors =====
        self.vision_preprocessor = VisionPreprocessor(
            image_size=(224, 224),
            normalize=True
        )
        
        self.audio_preprocessor = AudioPreprocessor(
            sample_rate=16000,
            duration=3.0,  # 3 seconds
            normalize=True
        )
        
        self.sensor_preprocessor = SensorPreprocessor(
            normalize=True,
            standardize=True
        )
    
    def forward(
        self,
        camera_images: Dict[str, torch.Tensor],
        audio: torch.Tensor,
        pressure: torch.Tensor,
        emg: torch.Tensor,
        return_modality_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Process multimodal input and return fused representation
        
        Args:
            camera_images: Dictionary with keys 'camera1', 'camera2'
                          Each value is tensor of shape (batch_size, 3, H, W)
            audio: Audio tensor of shape (batch_size, num_samples)
            pressure: Pressure sensor readings of shape (batch_size, sequence_length)
            emg: EMG sensor readings of shape (batch_size, num_channels, sequence_length)
            return_modality_embeddings: Whether to also return individual embeddings
        
        Returns:
            fused_embedding: Tensor of shape (batch_size, fusion_dim)
            modality_embeddings: Optional dict with embeddings from each modality
        """
        embeddings = {}
        
        # ===== Process Vision (Average both cameras) =====
        camera1_imgs = camera_images['camera1']  # (batch, 3, H, W)
        camera2_imgs = camera_images['camera2']  # (batch, 3, H, W)
        
        with torch.no_grad():
            camera1_emb = self.vision_encoder(camera1_imgs)  # (batch, vision_dim)
            camera2_emb = self.vision_encoder(camera2_imgs)  # (batch, vision_dim)
        
        # Average embeddings from both cameras
        vision_embedding = (camera1_emb + camera2_emb) / 2
        embeddings['vision'] = vision_embedding
        
        # ===== Process Audio =====
        with torch.no_grad():
            audio_embedding = self.audio_encoder.encode(
                audio,
                sampling_rate=16000,
                pooling="mean"
            )  # (batch, audio_dim)
        embeddings['audio'] = audio_embedding
        
        # ===== Process Pressure Sensor =====
        # Extract temporal features from pressure signal
        pressure_features = self.sensor_preprocessor.extract_features_temporal(
            pressure,
            window_size=50
        )  # (batch, num_features)
        pressure_embedding = self.pressure_encoder(pressure_features)
        embeddings['pressure'] = pressure_embedding
        
        # ===== Process EMG Sensor =====
        # Extract temporal features from EMG signals
        emg_features = self.sensor_preprocessor.extract_features_temporal(
            emg,
            window_size=50
        )  # (batch, num_features)
        emg_embedding = self.emg_encoder(emg_features)
        embeddings['emg'] = emg_embedding
        
        # ===== Fuse all modalities =====
        fused = self.fusion_module(embeddings)
        
        if return_modality_embeddings:
            return fused, embeddings
        else:
            return fused
    
    def encode_vision_only(
        self,
        camera_images: Dict[str, torch.Tensor],
        average: bool = True
    ) -> torch.Tensor:
        """Encode vision inputs only"""
        with torch.no_grad():
            cam1 = self.vision_encoder(camera_images['camera1'])
            cam2 = self.vision_encoder(camera_images['camera2'])
            if average:
                return (cam1 + cam2) / 2
            else:
                return torch.cat([cam1, cam2], dim=1)
    
    def encode_audio_only(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio input only"""
        with torch.no_grad():
            return self.audio_encoder.encode(audio, pooling="mean")
    
    def encode_pressure_only(self, pressure: torch.Tensor) -> torch.Tensor:
        """Encode pressure sensor only"""
        features = self.sensor_preprocessor.extract_features_temporal(
            pressure,
            window_size=50
        )
        return self.pressure_encoder(features)
    
    def encode_emg_only(self, emg: torch.Tensor) -> torch.Tensor:
        """Encode EMG sensor only"""
        features = self.sensor_preprocessor.extract_features_temporal(
            emg,
            window_size=50
        )
        return self.emg_encoder(features)
    
    def get_modality_dimensions(self) -> Dict[str, int]:
        """Get embedding dimensions for each modality"""
        return {
            'vision': self.vision_encoder.output_dim,
            'audio': self.audio_encoder.output_dim,
            'pressure': self.pressure_encoder.output_dim_value,
            'emg': self.emg_encoder.output_dim_value,
            'fused': self.fusion_dim
        }
    
    def freeze_encoders(self):
        """Freeze all pre-trained encoders"""
        self.vision_encoder.eval()
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        self.audio_encoder.eval()
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoders(self):
        """Unfreeze encoders for fine-tuning"""
        self.vision_encoder.train()
        for param in self.vision_encoder.parameters():
            param.requires_grad = True
        
        self.audio_encoder.train()
        for param in self.audio_encoder.parameters():
            param.requires_grad = True
