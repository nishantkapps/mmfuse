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

        # Normalize input shape to (batch, sequence_length)
        # Accepts: 1D (num_samples,), 2D (batch, num_samples), or higher dims
        if isinstance(audio, torch.Tensor):
            # Collapse all leading dims into batch dimension except the last (time) dim
            if audio.dim() == 1:
                audio_list = [audio.cpu().numpy()]
            else:
                if audio.dim() > 2:
                    batch = int(torch.prod(torch.tensor(audio.shape[:-1])).item())
                    audio = audio.reshape(batch, audio.shape[-1])
                # convert to numpy list per example
                audio_list = [a.cpu().numpy() for a in audio]
        elif isinstance(audio, np.ndarray):
            if audio.ndim == 1:
                audio_list = [audio]
            else:
                if audio.ndim > 2:
                    batch = int(np.prod(audio.shape[:-1]))
                    audio = audio.reshape(batch, audio.shape[-1])
                audio_list = [a for a in audio]
        elif isinstance(audio, list):
            audio_list = audio
        else:
            raise TypeError(f"Unsupported audio type: {type(audio)}")

        with torch.no_grad():
            # Use processor on list of 1D arrays (or Python floats)
            inputs = self.processor(
                audio_list,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True
            )
            # Move tensors to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

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


class Wav2VecPooledEncoder(AudioEncoder):
    """
    Wav2Vec encoder that returns (batch, embedding_dim) via mean pooling.
    Use this for fusion pipelines that expect fixed-size embeddings per modality.
    """
    def forward(
        self,
        audio: torch.Tensor,
        sampling_rate: Optional[int] = None
    ) -> torch.Tensor:
        out = super().forward(audio, sampling_rate)
        return out.mean(dim=1)
