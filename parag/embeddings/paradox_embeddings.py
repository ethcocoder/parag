"""
ParadoxEmbeddings: Custom embedding model using modules.framework.

Replaces sentence-transformers with a lightweight embedding model
built on modules.framework.Tensor and Paradma operations.
"""

import sys
import os
from typing import List, Union, Optional
import numpy as np

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import modules.framework
try:
    from modules.framework.tensor import Tensor
    from modules.framework.module import Module
    from modules.framework.tokenizer import SimpleTokenizer
    FRAMEWORK_AVAILABLE = True
except ImportError:
    FRAMEWORK_AVAILABLE = False
    Tensor = None
    Module = None
    SimpleTokenizer = None

# Import Paradma
try:
    from paradma import learning, Axiom
    PARADMA_AVAILABLE = True
except ImportError:
    PARADMA_AVAILABLE = False
    learning = None

from parag.embeddings.base import EmbeddingModel
from parag.utils.paradox_utils import ensure_numpy_type


class SimpleEmbeddingLayer(Module if FRAMEWORK_AVAILABLE else object):
    """
    Simple embedding layer using modules.framework.
    
    Maps token indices to dense vectors.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Initialize embedding layer.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
        """
        if not FRAMEWORK_AVAILABLE:
            raise ImportError("modules.framework is required for ParadoxEmbeddings")
        
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Initialize embedding matrix with Xavier initialization
        scale = np.sqrt(2.0 / (vocab_size + embedding_dim))
        self.embeddings = Tensor.normal(
            vocab_size, embedding_dim,
            mean=0.0, std=scale,
            requires_grad=True
        )
    
    def forward(self, indices):
        """
        Look up embeddings for token indices.
        
        Args:
            indices: Tensor of token indices
            
        Returns:
            Tensor of embeddings
        """
        # Simple lookup
        if isinstance(indices, (list, np.ndarray)):
            indices = Tensor(indices)
        
        # Gather embeddings (simplified - assumes indices are valid)
        # In production, would use proper indexing
        result = []
        indices_data = indices.data if hasattr(indices, 'data') else indices
        
        for idx in indices_data.flatten():
            idx = int(idx)
            if 0 <= idx < self.vocab_size:
                result.append(self.embeddings.data[idx])
        
        return Tensor(np.array(result))


class ParadoxEmbeddings(EmbeddingModel):
    """
    Custom embedding model using modules.framework and Paradma.
    
    This is a lightweight embedding model that:
    1. Uses modules.framework.Tensor for all operations
    2. Leverages Paradma for mathematical operations (self-learning)
    3. Provides simple but effective embeddings
    
    For production, can be extended with:
    - modules.transformer for transformer-based embeddings
    - Pre-trained weights loaded from file
    - Fine-tuning capabilities
    """
    
    def __init__(
        self,
        embedding_dim: int = 384,
        vocab_size: int = 30000,
        model_name: str = "paradox-simple",
        normalize: bool = True,
        use_paradma: bool = True,
    ):
        """
        Initialize ParadoxEmbeddings.
        
        Args:
            embedding_dim: Dimension of output embeddings
            vocab_size: Size of vocabulary
            model_name: Name identifier
            normalize: Whether to normalize embeddings
            use_paradma: Whether to use Paradma for operations
        """
        if not FRAMEWORK_AVAILABLE:
            raise ImportError(
                "modules.framework is required. "
                "Ensure modules directory is in sys.path."
            )
        
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.model_name = model_name
        self.normalize_embeddings = normalize
        self.use_paradma = use_paradma and PARADMA_AVAILABLE
        
        # Initialize tokenizer
        self.tokenizer = SimpleTokenizer(vocab_size=vocab_size)
        
        # Initialize embedding layer
        self.embedding_layer = SimpleEmbeddingLayer(vocab_size, embedding_dim)
        
        print(f"[ParadoxEmbeddings] Initialized: dim={embedding_dim}, "
              f"paradma={'ON' if self.use_paradma else 'OFF'}")
    
    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text."""
        if isinstance(text, str):
            text = [text]
            single_input = True
        else:
            single_input = False
        
        embeddings = []
        
        for txt in text:
            # Tokenize
            tokens = self.tokenizer.tokenize(txt)
            token_ids = self.tokenizer.encode(tokens)
            
            # Get embeddings
            token_embeddings = self.embedding_layer.forward(token_ids)
            
            # Pool embeddings (mean pooling)
            if self.use_paradma:
                # Use Paradma for mean calculation (self-learning!)
                pooled = self._mean_pool_paradma(token_embeddings)
            else:
                # Use NumPy
                pooled = self._mean_pool_numpy(token_embeddings)
            
            # Normalize if requested
            if self.normalize_embeddings:
                if self.use_paradma:
                    pooled = self._normalize_paradma(pooled)
                else:
                    pooled = self._normalize_numpy(pooled)
            
            embeddings.append(pooled)
        
        # Stack into array
        result = np.array(embeddings)
        
        if single_input:
            return result[0]
        return result
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Generate embeddings for a batch of texts."""
        all_embeddings = []
        
        # Process in batches
        num_batches = (len(texts) + batch_size - 1) // batch_size
        
        iterator = range(num_batches)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Embedding batches")
            except ImportError:
                pass
        
        for i in iterator:
            batch_texts = texts[i * batch_size:(i + 1) * batch_size]
            batch_embeddings = self.embed(batch_texts)
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        return self.embedding_dim
    
    def _mean_pool_paradma(self, token_embeddings: Tensor) -> np.ndarray:
        """Mean pooling using Paradma (self-learning)."""
        # Convert to Axiom for Paradma operations
        data = token_embeddings.data if hasattr(token_embeddings, 'data') else token_embeddings
        
        # Use Paradma's learning.mean() - this learns over time!
        axiom_data = Axiom(data.tolist(), manifold=learning)
        mean_result = learning.mean(axiom_data)
        
        # Convert back to NumPy
        result = mean_result.value if hasattr(mean_result, 'value') else mean_result
        return np.array(result)
    
    def _mean_pool_numpy(self, token_embeddings: Tensor) -> np.ndarray:
        """Mean pooling using NumPy."""
        data = token_embeddings.data if hasattr(token_embeddings, 'data') else token_embeddings
        return np.mean(data, axis=0)
    
    def _normalize_paradma(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector using Paradma operations."""
        # Convert to Axiom
        axiom = Axiom(vector.tolist(), manifold=learning)
        
        # Calculate norm using Paradma (sqrt of dot product)
        dot_result = learning.dot(axiom, axiom)
        norm_squared = dot_result.value if hasattr(dot_result, 'value') else dot_result
        
        # Use Paradma's sqrt (self-learning!)
        norm_axiom = Axiom(norm_squared, manifold=learning)
        norm_result = learning.sqrt(norm_axiom)
        norm = norm_result.value if hasattr(norm_result, 'value') else norm_result
        
        # Avoid division by zero
        if norm > 0:
            return vector / norm
        return vector
    
    def _normalize_numpy(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector using NumPy."""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    def save(self, path: str):
        """
        Save embedding model to file.
        
        Args:
            path: Path to save to
        """
        import pickle
        from pathlib import Path
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        state = {
            'embedding_dim': self.embedding_dim,
            'vocab_size': self.vocab_size,
            'model_name': self.model_name,
            'normalize_embeddings': self.normalize_embeddings,
            'embeddings': self.embedding_layer.embeddings.data,
            'tokenizer_vocab': self.tokenizer.vocab if hasattr(self.tokenizer, 'vocab') else None,
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"[ParadoxEmbeddings] Saved to {path}")
    
    def load(self, path: str):
        """
        Load embedding model from file.
        
        Args:
            path: Path to load from
        """
        import pickle
        
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.embedding_dim = state['embedding_dim']
        self.vocab_size = state['vocab_size']
        self.model_name = state['model_name']
        self.normalize_embeddings = state['normalize_embeddings']
        
        # Restore embeddings
        self.embedding_layer.embeddings = Tensor(
            state['embeddings'],
            requires_grad=True
        )
        
        if state['tokenizer_vocab']:
            self.tokenizer.vocab = state['tokenizer_vocab']
        
        print(f"[ParadoxEmbeddings] Loaded from {path}")
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ParadoxEmbeddings("
            f"model='{self.model_name}', "
            f"dim={self.embedding_dim}, "
            f"paradma={'ON' if self.use_paradma else 'OFF'})"
        )
