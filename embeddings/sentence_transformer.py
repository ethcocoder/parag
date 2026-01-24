"""
SentenceTransformer embeddings implementation.

Uses the sentence-transformers library for high-quality text embeddings.
"""

from typing import List, Union, Optional
import numpy as np
from tqdm import tqdm

from parag.embeddings.base import EmbeddingModel


class SentenceTransformerEmbeddings(EmbeddingModel):
    """
    SentenceTransformer-based embedding model.
    
    Uses pre-trained models from the sentence-transformers library.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        """
        Initialize SentenceTransformer embeddings.
        
        Args:
            model_name: Name of the sentence-transformers model
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            normalize: Whether to normalize embeddings
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.normalize_embeddings = normalize
        
        # Load model
        self.model = SentenceTransformer(model_name, device=device)
        self._embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text."""
        if isinstance(text, str):
            text = [text]
            single_input = True
        else:
            single_input = False
        
        # Generate embeddings
        embeddings = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        
        if single_input:
            return embeddings[0]
        return embeddings
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Generate embeddings for a batch of texts."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=show_progress,
        )
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        return self._embedding_dim
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SentenceTransformerEmbeddings("
            f"model='{self.model_name}', "
            f"dim={self._embedding_dim}, "
            f"normalize={self.normalize_embeddings})"
        )
