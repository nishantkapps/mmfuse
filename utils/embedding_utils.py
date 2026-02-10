"""
Utility functions for the multimodal fusion system
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import json


class EmbeddingAnalysis:
    """Utilities for analyzing embeddings"""
    
    @staticmethod
    def cosine_similarity(
        embedding1: torch.Tensor,
        embedding2: torch.Tensor
    ) -> float:
        """
        Compute cosine similarity between embeddings
        
        Args:
            embedding1: First embedding tensor
            embedding2: Second embedding tensor
        
        Returns:
            Cosine similarity score (0 to 1)
        """
        # Normalize
        emb1_norm = F.normalize(embedding1, p=2, dim=-1)
        emb2_norm = F.normalize(embedding2, p=2, dim=-1)
        
        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            emb1_norm.unsqueeze(0),
            emb2_norm.unsqueeze(0)
        ).item()
        
        return similarity
    
    @staticmethod
    def euclidean_distance(
        embedding1: torch.Tensor,
        embedding2: torch.Tensor
    ) -> float:
        """Compute Euclidean distance between embeddings"""
        distance = torch.norm(embedding1 - embedding2).item()
        return distance
    
    @staticmethod
    def modality_contribution(
        modality_embeddings: Dict[str, torch.Tensor],
        fused_embedding: torch.Tensor
    ) -> Dict[str, float]:
        """
        Estimate each modality's contribution to fused embedding
        
        Args:
            modality_embeddings: Dictionary of individual modality embeddings
            fused_embedding: The fused embedding
        
        Returns:
            Dictionary with normalized contribution scores
        """
        contributions = {}
        
        for modality, emb in modality_embeddings.items():
            # Use cosine similarity as contribution metric
            sim = F.cosine_similarity(
                emb.unsqueeze(0),
                fused_embedding.unsqueeze(0)
            ).item()
            contributions[modality] = max(0, sim)  # Normalize to [0, 1]
        
        # Normalize to sum to 1
        total = sum(contributions.values())
        if total > 0:
            contributions = {k: v/total for k, v in contributions.items()}
        
        return contributions
    
    @staticmethod
    def embedding_statistics(
        embedding: torch.Tensor,
        modality_name: str = "embedding"
    ) -> Dict:
        """
        Compute statistics of an embedding
        
        Args:
            embedding: Embedding tensor
            modality_name: Name of the modality
        
        Returns:
            Dictionary with statistics
        """
        return {
            'modality': modality_name,
            'dimension': int(embedding.size(-1)),
            'norm': float(embedding.norm()),
            'mean': float(embedding.mean()),
            'std': float(embedding.std()),
            'min': float(embedding.min()),
            'max': float(embedding.max()),
            'sparsity': float((embedding.abs() < 0.01).float().mean())
        }


class EmbeddingStorage:
    """Utilities for storing and retrieving embeddings"""
    
    @staticmethod
    def save_embedding(
        embedding: torch.Tensor,
        file_path: str,
        metadata: Optional[Dict] = None
    ):
        """
        Save embedding to disk
        
        Args:
            embedding: Embedding tensor
            file_path: Path to save to
            metadata: Optional metadata dictionary
        """
        save_dict = {
            'embedding': embedding.cpu(),
            'metadata': metadata or {}
        }
        torch.save(save_dict, file_path)
    
    @staticmethod
    def load_embedding(file_path: str) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Load embedding from disk
        
        Args:
            file_path: Path to load from
        
        Returns:
            (embedding, metadata) tuple
        """
        data = torch.load(file_path)
        return data['embedding'], data.get('metadata')
    
    @staticmethod
    def save_batch_embeddings(
        embeddings: Dict[str, torch.Tensor],
        labels: List[str],
        file_path: str
    ):
        """Save a batch of embeddings with labels"""
        save_dict = {
            'embeddings': {k: v.cpu() for k, v in embeddings.items()},
            'labels': labels
        }
        torch.save(save_dict, file_path)
    
    @staticmethod
    def load_batch_embeddings(file_path: str) -> Tuple[Dict, List[str]]:
        """Load a batch of embeddings with labels"""
        data = torch.load(file_path)
        return data['embeddings'], data['labels']


class EmbeddingRetrieval:
    """Simple retrieval system using embeddings"""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize retrieval system
        
        Args:
            device: Computation device
        """
        self.device = device
        self.database_embeddings = []
        self.database_labels = []
    
    def add_to_database(
        self,
        embedding: torch.Tensor,
        label: str
    ):
        """Add embedding to database"""
        self.database_embeddings.append(embedding.to(self.device))
        self.database_labels.append(label)
    
    def retrieve_similar(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Retrieve most similar embeddings from database
        
        Args:
            query_embedding: Query embedding
            top_k: Number of results to return
        
        Returns:
            List of (label, similarity) tuples, sorted by similarity
        """
        query_embedding = query_embedding.to(self.device)
        
        similarities = []
        for db_emb, label in zip(self.database_embeddings, self.database_labels):
            sim = F.cosine_similarity(
                query_embedding.unsqueeze(0),
                db_emb.unsqueeze(0)
            ).item()
            similarities.append((label, sim))
        
        # Sort by similarity, descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def clear_database(self):
        """Clear the database"""
        self.database_embeddings = []
        self.database_labels = []


class EmbeddingVisualization:
    """Utilities for visualizing embeddings"""
    
    @staticmethod
    def reduce_to_2d(
        embeddings: torch.Tensor,
        method: str = "pca"
    ) -> np.ndarray:
        """
        Reduce embeddings to 2D for visualization
        
        Args:
            embeddings: Tensor of shape (num_samples, embedding_dim)
            method: Reduction method ('pca', 'tsne', 'umap')
        
        Returns:
            2D coordinates of shape (num_samples, 2)
        """
        emb_np = embeddings.cpu().numpy()
        
        if method == "pca":
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            return pca.fit_transform(emb_np)
        
        elif method == "tsne":
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2)
            return tsne.fit_transform(emb_np)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def create_similarity_matrix(
        embeddings_dict: Dict[str, torch.Tensor]
    ) -> np.ndarray:
        """
        Create similarity matrix between modalities
        
        Args:
            embeddings_dict: Dictionary of modality embeddings
        
        Returns:
            Similarity matrix of shape (num_modalities, num_modalities)
        """
        modalities = list(embeddings_dict.keys())
        n = len(modalities)
        
        similarity_matrix = np.zeros((n, n))
        
        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities):
                emb1 = embeddings_dict[mod1]
                emb2 = embeddings_dict[mod2]
                
                sim = F.cosine_similarity(
                    emb1.unsqueeze(0),
                    emb2.unsqueeze(0)
                ).item()
                
                similarity_matrix[i, j] = sim
        
        return similarity_matrix


def create_embeddings_report(
    fused_embedding: torch.Tensor,
    modality_embeddings: Dict[str, torch.Tensor],
    output_path: Optional[str] = None
) -> Dict:
    """
    Create a comprehensive report of embeddings
    
    Args:
        fused_embedding: The fused embedding
        modality_embeddings: Dictionary of modality embeddings
        output_path: Optional path to save JSON report
    
    Returns:
        Dictionary with report data
    """
    report = {
        'fused': EmbeddingAnalysis.embedding_statistics(fused_embedding, 'fused'),
        'modalities': {},
        'contribution': EmbeddingAnalysis.modality_contribution(
            modality_embeddings,
            fused_embedding
        )
    }
    
    for modality, emb in modality_embeddings.items():
        report['modalities'][modality] = EmbeddingAnalysis.embedding_statistics(
            emb,
            modality
        )
    
    if output_path:
        with open(output_path, 'w') as f:
            # Convert tensors to floats for JSON serialization
            json_report = {
                'fused': report['fused'],
                'modalities': report['modalities'],
                'contribution': report['contribution']
            }
            json.dump(json_report, f, indent=2)
    
    return report
