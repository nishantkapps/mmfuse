"""
Sensor Encoder for pressure and EMG signals
Uses neural network with learnable layers on top of pre-processed sensor data
"""

import torch
import torch.nn as nn
from typing import Optional


class SensorEncoder(nn.Module):
    """
    Encoder for tabular sensor data (pressure, EMG)
    
    Uses a small neural network to encode 1D sensor signals.
    Pre-processes using statistical features (mean, std, energy, etc.)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.2
    ):
        """
        Initialize Sensor Encoder
        
        Args:
            input_dim: Dimension of input signal (e.g., number of channels)
            output_dim: Dimension of output embedding
            hidden_dim: Dimension of hidden layers
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Multi-layer network for encoding
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    @property
    def output_dim_value(self) -> int:
        """Get embedding dimension"""
        return self.output_dim
    
    def forward(self, sensor_data: torch.Tensor) -> torch.Tensor:
        """
        Encode sensor data to embeddings
        
        Args:
            sensor_data: Tensor of shape (batch_size, input_dim)
                        Can be pre-computed features or raw sensor readings
        
        Returns:
            Embeddings of shape (batch_size, output_dim)
        """
        return self.encoder(sensor_data)
    
    def extract_temporal_features(
        self,
        signal: torch.Tensor,
        window_size: int = 100
    ) -> torch.Tensor:
        """
        Extract statistical features from temporal sensor signals
        
        Args:
            signal: Tensor of shape (batch_size, sequence_length) or (sequence_length,)
            window_size: Size of window for feature extraction
        
        Returns:
            Features of shape (batch_size, num_features) with statistical summaries
        """
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)
        
        batch_size = signal.size(0)
        num_windows = max(1, signal.size(1) // window_size)
        
        features = []
        
        for i in range(num_windows):
            start_idx = i * window_size
            end_idx = min((i + 1) * window_size, signal.size(1))
            window = signal[:, start_idx:end_idx]
            
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
            all_features = torch.zeros(batch_size, 5, device=signal.device)
        
        return all_features


class PressureSensorEncoder(SensorEncoder):
    """Specialized encoder for pressure sensor data"""
    
    def __init__(self, output_dim: int = 256, num_channels: int = 1, input_features: int = 100):
        """
        Initialize Pressure Sensor Encoder
        
        Args:
            output_dim: Dimension of output embedding
            num_channels: Number of pressure sensor channels
            input_features: Number of input features (from temporal extraction)
        """
        # Input dim is num_channels * input_features
        # Default: 1 channel * 100 features = 100 dims
        super().__init__(
            input_dim=input_features,
            output_dim=output_dim,
            hidden_dim=128
        )
        self.num_channels = num_channels


class EMGSensorEncoder(SensorEncoder):
    """Specialized encoder for EMG (Electromyography) sensor data"""
    
    def __init__(self, output_dim: int = 256, num_channels: int = 8, input_features: int = 100):
        """
        Initialize EMG Sensor Encoder
        
        Args:
            output_dim: Dimension of output embedding
            num_channels: Number of EMG sensor channels (typical: 1-16)
            input_features: Number of input features (from temporal extraction)
        """
        super().__init__(
            input_dim=input_features,
            output_dim=output_dim,
            hidden_dim=128
        )
        self.num_channels = num_channels
