"""
Base embedding interface.

Defines abstract interface for embedding models to ensure consistency.
"""

from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np


class EmbeddingModel(ABC):
    """
    Abstract base class for embedding models.
    
    All embedding implementations should subclass this and implement
    the required methods.
    """
    
    @abstractmethod
    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text.
        
        Args:
            text: Single string or list of strings to embed
            
        Returns:
            Numpy array of embeddings. Shape:
                - (embedding_dim,) for single string
                - (num_texts, embedding_dim) for list of strings
        """
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of strings to embed
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of shape (num_texts, embedding_dim)
        """
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """
        Get the dimensionality of embeddings produced by this model.
        
        Returns:
            Embedding dimension
        """
        pass
    
    def normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings to unit length.
        
        Args:
            embeddings: Embeddings to normalize
            
        Returns:
            Normalized embeddings
        """
        if embeddings.ndim == 1:
            # Single embedding
            norm = np.linalg.norm(embeddings)
            if norm > 0:
                return embeddings / norm
            return embeddings
        else:
            # Batch of embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)  # Avoid division by zero
            return embeddings / norms
    
    def embed_and_normalize(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate and normalize embeddings.
        
        Args:
            text: Text(s) to embed
            
        Returns:
            Normalized embeddings
        """
        embeddings = self.embed(text)
        return self.normalize(embeddings)
