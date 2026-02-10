"""
Audio Encoder using Wav2Vec 2.0 for audio input
Uses pre-trained Wav2Vec 2.0 from HuggingFace for robust audio encoding
"""

import torch
import torch.nn as nn
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from typing import Union, Optional


class AudioEncoder(nn.Module):
    """
    Audio encoder using Wav2Vec 2.0 (Facebook/Meta)
    
    Wav2Vec 2.0 is trained on 960 hours of unlabeled speech data
    and provides excellent representations for audio signals.
    Suitable for robotic audio feedback including voice and ambient sounds.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        frozen: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        sampling_rate: int = 16000
    ):
        """
        Initialize Audio Encoder
        
        Args:
            model_name: HuggingFace model identifier
            frozen: Whether to freeze pre-trained weights
            device: Device to load model on
            sampling_rate: Expected audio sampling rate in Hz
        """
        super().__init__()
        self.device = device
        self.sampling_rate = sampling_rate
        
        # Load pre-trained processor and model
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model.to(device)
        
        # Get embedding dimension from model config
        self.embedding_dim = self.model.config.hidden_size
        
        # Freeze parameters if requested
        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False
    
    @property
    def output_dim(self) -> int:
        """Get embedding dimension"""
        return self.embedding_dim
    
    def forward(
        self,
        audio: torch.Tensor,
        sampling_rate: Optional[int] = None
    ) -> torch.Tensor:
        """
        Encode audio to embeddings
        
        Args:
            audio: Tensor of shape (batch_size, num_samples) with values in [-1, 1]
            sampling_rate: Sampling rate of audio (if different from default)
        
        Returns:
            Embeddings of shape (batch_size, sequence_length, embedding_dim)
            Use mean pooling across sequence dimension for fixed-size embedding
        """
        sr = sampling_rate or self.sampling_rate
        
        with torch.no_grad():
            # Process audio inputs
            inputs = self.processor(
                audio,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Get audio features
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state
        
        return last_hidden_state
    
    def encode(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        sampling_rate: Optional[int] = None,
        pooling: str = "mean"
    ) -> torch.Tensor:
        """
        Encode audio with optional pooling
        
        Args:
            audio: Audio tensor or numpy array
            sampling_rate: Sampling rate of audio
            pooling: Pooling method - 'mean', 'max', or None (return sequence)
        
        Returns:
            Embeddings of shape (batch_size, embedding_dim) if pooling is specified,
            else (batch_size, sequence_length, embedding_dim)
        """
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        # Get sequence embeddings
        embeddings = self.forward(audio, sampling_rate)
        
        if pooling == "mean":
            # Mean pooling across sequence dimension
            embeddings = embeddings.mean(dim=1)
        elif pooling == "max":
            # Max pooling across sequence dimension
            embeddings = embeddings.max(dim=1)[0]
        elif pooling is not None:
            raise ValueError(f"Unknown pooling method: {pooling}")
        
        return embeddings
    
    def load_audio_file(
        self,
        file_path: str,
        sr: Optional[int] = None
    ) -> torch.Tensor:
        """
        Load audio file and convert to appropriate format
        
        Args:
            file_path: Path to audio file
            sr: Target sampling rate (defaults to self.sampling_rate)
        
        Returns:
            Audio tensor of shape (num_samples,)
        """
        target_sr = sr or self.sampling_rate
        audio, _ = librosa.load(file_path, sr=target_sr)
        audio = torch.from_numpy(audio).float()
        
        return audio
