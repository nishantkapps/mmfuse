"""
Data preprocessing pipelines for different modalities
Handles normalization, resizing, and format conversion
"""

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
from typing import Union, Tuple, Optional
import librosa


class VisionPreprocessor:
    """Preprocessing for camera/vision inputs"""
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        normalize: bool = True
    ):
        """
        Initialize Vision Preprocessor
        
        Args:
            image_size: Target size for images (height, width)
            normalize: Whether to apply normalization
        """
        self.image_size = image_size
        self.normalize = normalize
        
        # Standard ImageNet normalization
        if normalize:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor()
            ])
    
    def preprocess(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Preprocess a single image
        
        Args:
            image: PIL Image, numpy array, or torch tensor
        
        Returns:
            Tensor of shape (3, H, W) with values in [0, 1]
        """
        if isinstance(image, torch.Tensor):
            image = image.numpy() if image.is_cpu else image.cpu().numpy()
        
        if isinstance(image, np.ndarray):
            # Convert numpy to PIL
            if image.dtype == np.uint8:
                image = Image.fromarray(image)
            else:
                # Assume float in [0, 1]
                image = Image.fromarray((image * 255).astype(np.uint8))
        
        return self.transform(image)
    
    def preprocess_batch(
        self,
        images: Union[list, torch.Tensor]
    ) -> torch.Tensor:
        """
        Preprocess a batch of images
        
        Args:
            images: List of images or tensor of shape (batch_size, ...)
        
        Returns:
            Tensor of shape (batch_size, 3, H, W)
        """
        if isinstance(images, torch.Tensor):
            processed = [self.preprocess(img) for img in images]
        else:
            processed = [self.preprocess(img) for img in images]
        
        return torch.stack(processed)


class AudioPreprocessor:
    """
    Preprocessing for audio inputs containing voice commands and operational sounds
    
    Audio Content:
    - Voice commands: "Move up", "Move down", "Along arm", gripper commands
    - Operational audio: Motor sounds, actuator noise, task-specific acoustics
    - Duration: 5 seconds (synchronized with one camera frame)
    - Sample rate: 16 kHz
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        duration: Optional[float] = 5.0,
        normalize: bool = True
    ):
        """
        Initialize Audio Preprocessor
        
        Args:
            sample_rate: Target sample rate in Hz (default 16kHz)
            duration: Duration in seconds (default 5.0 to match one camera frame)
                     5 seconds @ 16kHz = 80,000 samples for voice commands + response audio
            normalize: Whether to normalize audio to [-1, 1]
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.normalize = normalize
    
    def preprocess(
        self,
        audio: Union[np.ndarray, torch.Tensor, str],
        sr: Optional[int] = None
    ) -> torch.Tensor:
        """
        Preprocess audio
        
        Args:
            audio: Audio array, tensor, or path to audio file
            sr: Original sample rate (if resampling needed)
        
        Returns:
            Tensor of shape (num_samples,) with values in [-1, 1]
        """
        # Load audio if path provided
        if isinstance(audio, str):
            audio, sr = librosa.load(audio, sr=self.sample_rate)
        
        # Convert to tensor
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        # Resample if needed
        if sr is not None and sr != self.sample_rate:
            num_samples = int(audio.size(0) * self.sample_rate / sr)
            audio = torch.nn.functional.interpolate(
                audio.unsqueeze(0).unsqueeze(0),
                size=num_samples,
                mode='linear',
                align_corners=False
            ).squeeze()
        
        # Normalize
        if self.normalize:
            # Normalize to [-1, 1]
            max_val = audio.abs().max()
            if max_val > 0:
                audio = audio / max_val
        
        # Pad or truncate to duration
        if self.duration is not None:
            target_samples = int(self.duration * self.sample_rate)
            if audio.size(0) < target_samples:
                # Pad with zeros
                audio = F.pad(audio, (0, target_samples - audio.size(0)))
            else:
                # Truncate
                audio = audio[:target_samples]
        
        return audio
    
    def preprocess_batch(
        self,
        audio_list: list,
        sr: Optional[int] = None
    ) -> torch.Tensor:
        """
        Preprocess batch of audio samples
        
        Args:
            audio_list: List of audio arrays, tensors, or file paths
            sr: Original sample rate
        
        Returns:
            Tensor of shape (batch_size, num_samples)
        """
        processed = [self.preprocess(audio, sr) for audio in audio_list]
        
        # Pad to max length
        max_len = max(len(audio) for audio in processed)
        padded = []
        for audio in processed:
            if len(audio) < max_len:
                padded.append(F.pad(audio, (0, max_len - len(audio))))
            else:
                padded.append(audio)
        
        return torch.stack(padded)


class SensorPreprocessor:
    """Preprocessing for sensor data (pressure, EMG)"""
    
    def __init__(
        self,
        normalize: bool = True,
        standardize: bool = True
    ):
        """
        Initialize Sensor Preprocessor
        
        Args:
            normalize: Min-max normalization to [0, 1]
            standardize: Z-score standardization
        """
        self.normalize = normalize
        self.standardize = standardize
    
    def preprocess(
        self,
        sensor_data: Union[np.ndarray, torch.Tensor],
        axis: int = 0
    ) -> torch.Tensor:
        """
        Preprocess sensor data
        
        Args:
            sensor_data: Sensor readings as array or tensor
            axis: Axis along which to compute statistics
        
        Returns:
            Preprocessed tensor
        """
        if isinstance(sensor_data, np.ndarray):
            sensor_data = torch.from_numpy(sensor_data).float()
        
        data = sensor_data.clone()
        
        # Normalize to [0, 1]
        if self.normalize:
            min_val = data.min()
            max_val = data.max()
            if max_val > min_val:
                data = (data - min_val) / (max_val - min_val)
        
        # Standardize (z-score)
        if self.standardize:
            mean = data.mean()
            std = data.std()
            if std > 0:
                data = (data - mean) / std
        
        return data
    
    def preprocess_batch(
        self,
        sensor_batch: Union[list, torch.Tensor]
    ) -> torch.Tensor:
        """
        Preprocess batch of sensor data
        
        Args:
            sensor_batch: List or tensor of sensor readings
        
        Returns:
            Preprocessed tensor of shape (batch_size, ...)
        """
        if isinstance(sensor_batch, list):
            processed = [self.preprocess(data) for data in sensor_batch]
            return torch.stack(processed)
        else:
            return self.preprocess(sensor_batch)
    
    def extract_features_temporal(
        self,
        signal: torch.Tensor,
        window_size: int = 100
    ) -> torch.Tensor:
        """
        Extract temporal features from sensor signal
        
        Args:
            signal: Sensor signal of shape (..., sequence_length)
            window_size: Window size for feature extraction
        
        Returns:
            Statistical features (mean, std, min, max, energy)
        """
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)
        
        features = []
        seq_len = signal.size(-1)
        
        for start in range(0, seq_len, window_size):
            end = min(start + window_size, seq_len)
            window = signal[..., start:end]
            
            # Compute statistics
            feat = torch.stack([
                window.mean(dim=-1),
                window.std(dim=-1),
                window.min(dim=-1)[0],
                window.max(dim=-1)[0],
                (window ** 2).mean(dim=-1)  # Energy
            ], dim=-1)
            features.append(feat)
        
        if features:
            return torch.cat(features, dim=-1)
        else:
            return torch.zeros_like(signal)
