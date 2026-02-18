"""
Audio Encoder using Whisper (OpenAI) - frozen pretrained encoder.
Returns (batch, 768) embeddings via mean pooling over encoder outputs.
"""

import torch
import torch.nn as nn
from transformers import WhisperProcessor, WhisperModel
from typing import Optional


class WhisperAudioEncoder(nn.Module):
    """
    Whisper encoder (frozen) for audio embedding.
    Uses encoder-only path; returns mean-pooled (batch, 768) for fusion.
    """
    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        frozen: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        sampling_rate: int = 16000
    ):
        super().__init__()
        self.device = device
        self.sampling_rate = sampling_rate
        self.embedding_dim = 768  # whisper-small d_model

        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperModel.from_pretrained(model_name)
        self.model.to(device)

        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False

    @property
    def output_dim(self) -> int:
        return self.embedding_dim

    def forward(
        self,
        audio: torch.Tensor,
        sampling_rate: Optional[int] = None
    ) -> torch.Tensor:
        """
        Args:
            audio: (batch, num_samples) raw waveform in [-1, 1]
        Returns:
            (batch, 768) mean-pooled encoder embeddings
        """
        sr = sampling_rate or self.sampling_rate
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio_list = [a.cpu().numpy() for a in audio]

        with torch.no_grad():
            inputs = self.processor(
                audio_list,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model.encoder(**inputs)
            last_hidden = outputs.last_hidden_state  # (B, seq, 768)
        return last_hidden.mean(dim=1)
