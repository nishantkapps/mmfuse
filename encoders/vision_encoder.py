"""
Vision Encoder using CLIP for camera inputs
Uses pre-trained CLIP model to encode camera frames
"""

import torch
import torch.nn as nn
from torchvision import transforms
import open_clip
from typing import Optional, Tuple


class VisionEncoder(nn.Module):
    """
    Vision encoder using CLIP (Contrastive Language-Image Pre-training)
    
    CLIP is trained on 400M image-text pairs and provides robust
    visual embeddings that work well for robotic applications.
    
    Real-World Input Handling:
    - Raw camera input: 1024p (1920×1024) or HD (1280×720) webcam frames
    - CLIP processing: Automatically resized to 224×224 RGB
    - Output: L2-normalized 512-dimensional embedding
    
    The encoder's transform pipeline automatically handles resizing from
    high-resolution webcam frames to CLIP's expected 224×224 input size.
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        frozen: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize Vision Encoder
        
        Args:
            model_name: CLIP model variant ('ViT-B-32', 'ViT-B-16', 'ViT-L-14')
            pretrained: Pretrained weights ('openai', 'laion400m_e32', etc.)
            frozen: Whether to freeze pre-trained weights
            device: Device to load model on
        """
        super().__init__()
        self.device = device
        self.model_name = model_name
        
        # Load pre-trained CLIP model using open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device
        )
        
        # Get embedding dimension
        self.embedding_dim = self.model.text_projection.shape[0] if hasattr(self.model, 'text_projection') else 512
        
        # Freeze parameters if requested
        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Standard normalization for image inputs
        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    
    @property
    def output_dim(self) -> int:
        """Get embedding dimension"""
        return self.embedding_dim
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to embeddings
        
        Args:
            images: Tensor of shape (batch_size, 3, height, width)
                   Values should be in range [0, 1]
        
        Returns:
            Embeddings of shape (batch_size, embedding_dim)
        """
        with torch.no_grad():
            # Get visual features from CLIP
            image_features = self.model.encode_image(images)
            # Normalize embeddings
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features
    
    def preprocess_image(self, image):
        """
        Preprocess image for CLIP model
        
        Args:
            image: PIL Image or numpy array
        
        Returns:
            Preprocessed tensor ready for encoding
        """
        return self.preprocess(image)
