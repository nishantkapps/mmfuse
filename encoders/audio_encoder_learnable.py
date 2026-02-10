"""
Learnable Audio Encoder using CNN
Pure PyTorch implementation without external processors
Works directly with raw audio tensors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AudioEncoder(nn.Module):
    """
    Learnable Audio Encoder using 1D Convolutional Neural Network
    
    Architecture:
    - Multiple 1D conv layers to extract temporal features
    - Adaptive pooling to get fixed-size embeddings
    - Works directly with raw audio tensors
    
    Input: (batch_size, num_samples) - raw audio waveform
    Output: (batch_size, output_dim) - audio embedding
    """
    
    def __init__(
        self,
        output_dim: int = 768,
        num_filters: int = 256,
        num_layers: int = 4,
        kernel_size: int = 15,
        stride: int = 2,
        dropout: float = 0.2,
        frozen: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize Learnable Audio Encoder
        
        Args:
            output_dim: Dimension of output embedding (default 768)
            num_filters: Number of filters in conv layers
            num_layers: Number of conv layers
            kernel_size: Size of conv kernels
            stride: Stride for conv layers
            dropout: Dropout rate
            frozen: Whether to freeze weights (not typically used for learnable encoders)
            device: Device to load model on
        
        Real-World Audio Input Specification:
        - Input content: Voice commands + operational audio
        - Voice commands: "Move up", "Move down", "Along arm", gripper commands, etc.
        - Operational audio: Motor sounds, actuator noise, task-specific acoustics
        - Input duration: 5 seconds (captures one command sequence)
        - Sample rate: 16 kHz
        - Input shape: (batch_size, 80000) [5 seconds × 16000 Hz]
        - Output: 768-dimensional embedding
        
        Command Examples:
        - Movement: "up", "down", "left", "right", "forward", "backward"
        - Rotation: "rotate", "spin", "turn"
        - Gripper: "open", "close", "grip", "release"
        - Navigation: "home", "reset", "return"
        - Task control: "execute", "start", "stop", "pause"
        
        Temporal Synchronization:
        - One 5-second audio window per camera frame
        - Audio window captures voice command + resulting robot response
        - Synchronized timestamps enable joint vision-audio understanding
        - Fused embedding represents combined visual state + voice command intent
        """
        super().__init__()
        self.device = device
        self.output_dim = output_dim
        self.num_filters = num_filters
        
        # Initial projection layer to expand channels
        conv_layers = []
        
        # First conv layer: (batch, 1, num_samples) -> (batch, num_filters, seq_len)
        conv_layers.append(nn.Conv1d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2
        ))
        conv_layers.append(nn.BatchNorm1d(num_filters))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Dropout(dropout))
        
        # Additional conv layers
        for _ in range(num_layers - 1):
            conv_layers.append(nn.Conv1d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2
            ))
            conv_layers.append(nn.BatchNorm1d(num_filters))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Dropout(dropout))
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Adaptive pooling to get fixed-size output
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final projection to desired output dimension
        self.projection = nn.Sequential(
            nn.Linear(num_filters, num_filters * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_filters * 2, output_dim)
        )
        
        self.to(device)
    
    @property
    def output_dim_value(self) -> int:
        """Get embedding dimension"""
        return self.output_dim
    
    def forward(
        self,
        audio: torch.Tensor,
        sampling_rate: Optional[int] = None
    ) -> torch.Tensor:
        """
        Encode audio to embeddings
        
        Args:
            audio: Tensor of shape (batch_size, num_samples) or (num_samples,)
                   Raw audio waveform with values typically in [-1, 1]
            sampling_rate: Sampling rate (informational, not used for processing)
        
        Returns:
            Embeddings of shape (batch_size, output_dim)
        """
        # Handle 1D input
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        batch_size = audio.shape[0]
        
        # Normalize audio if needed
        max_val = audio.abs().max(dim=1, keepdim=True)[0]
        audio = audio / (max_val + 1e-8)
        
        # Add channel dimension: (batch_size, num_samples) -> (batch_size, 1, num_samples)
        audio = audio.unsqueeze(1)

        # Ensure model parameters live on the same device as the input to avoid
        # "Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor)"
        # runtime errors when caller moves tensors to CUDA but the module remained on CPU.
        target_device = audio.device
        try:
            param_device = next(self.parameters()).device
        except StopIteration:
            param_device = None

        if param_device != target_device:
            self.to(target_device)

        audio = audio.to(target_device)
        
        # Conv layers: (batch, 1, num_samples) -> (batch, num_filters, seq_len)
        features = self.conv_layers(audio)
        
        # Adaptive pooling: (batch, num_filters, seq_len) -> (batch, num_filters, 1)
        pooled = self.adaptive_pool(features)
        
        # Flatten: (batch, num_filters, 1) -> (batch, num_filters)
        pooled = pooled.squeeze(-1)
        
        # Project to output dimension: (batch, num_filters) -> (batch, output_dim)
        embeddings = self.projection(pooled)
        
        return embeddings


class AudioEncoderBase:
    """
    Alternative: Simple learnable audio feature extractor
    Can be used as a baseline or lightweight alternative
    """
    
    @staticmethod
    def extract_temporal_features(
        audio: torch.Tensor,
        window_size: int = 1024
    ) -> torch.Tensor:
        """
        Extract temporal features from raw audio
        
        Args:
            audio: Tensor of shape (batch_size, num_samples)
            window_size: Window size for feature extraction
        
        Returns:
            Features of shape (batch_size, num_windows * 5)
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        batch_size = audio.size(0)
        num_windows = max(1, audio.size(1) // window_size)
        
        features = []
        
        for i in range(num_windows):
            start_idx = i * window_size
            end_idx = min((i + 1) * window_size, audio.size(1))
            window = audio[:, start_idx:end_idx]
            
            # Compute statistical features
            mean = window.mean(dim=1, keepdim=True)
            std = window.std(dim=1, keepdim=True)
            min_val = window.min(dim=1, keepdim=True)[0]
            max_val = window.max(dim=1, keepdim=True)[0]
            energy = (window ** 2).mean(dim=1, keepdim=True)
            
            window_features = torch.cat([mean, std, min_val, max_val, energy], dim=1)
            features.append(window_features)
        
        # Concatenate all window features
        if features:
            all_features = torch.cat(features, dim=1)
        else:
            all_features = torch.zeros(batch_size, 5, device=audio.device)
        
        return all_features
