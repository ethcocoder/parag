"""
KnowledgeUnit: The fundamental data structure for retrieved information.

All knowledge in the RAG system is represented as KnowledgeUnits,
providing a standard interface for content, embeddings, and metadata.
"""

from typing import Any, Dict, Optional, Union
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
import sys
import os

# Add parent directory to path for Paradox imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import Paradma components
try:
    from paradma import Axiom, TensorAxiom, learning
    PARADMA_AVAILABLE = True
except ImportError:
    PARADMA_AVAILABLE = False
    Axiom = None
    TensorAxiom = None
    learning = None

# Import conversion utilities
try:
    from parag.utils.paradox_utils import (
        ensure_paradma_type,
        ensure_numpy_type,
        numpy_to_axiom,
        numpy_to_tensor_axiom,
    )
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False


@dataclass
class KnowledgeUnit:
    """
    A single unit of knowledge in the RAG system.
    
    Attributes:
        content: The actual content (text, image, tensor, etc.)
        embedding: Vector representation of the content
        metadata: Additional information (source, timestamp, tags, etc.)
        confidence: Optional confidence score for this knowledge unit
        unit_id: Unique identifier for this unit
    """
    
    content: Union[str, bytes, Any]
    embedding: Optional[Union[np.ndarray, 'Axiom', 'TensorAxiom']] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: Optional[float] = None
    unit_id: Optional[str] = None
    _use_paradma: bool = True  # Prefer Paradma when available
    
    def __post_init__(self):
        """Validate and process fields after initialization."""
        # Auto-generate ID if not provided
        if self.unit_id is None:
            self.unit_id = self._generate_id()
        
        # Add creation timestamp if not in metadata
        if "created_at" not in self.metadata:
            self.metadata["created_at"] = datetime.now().isoformat()
        
        # Validate confidence score
        if self.confidence is not None:
            if not (0.0 <= self.confidence <= 1.0):
                raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
    
    def _generate_id(self) -> str:
        """Generate a unique ID for this knowledge unit."""
        import hashlib
        
        # Create ID from content and timestamp
        content_str = str(self.content)[:1000]  # Use first 1000 chars
        timestamp = self.metadata.get("created_at", datetime.now().isoformat())
        
        id_string = f"{content_str}{timestamp}"
        return hashlib.sha256(id_string.encode()).hexdigest()[:16]
    
    def has_embedding(self) -> bool:
        """Check if this unit has an embedding."""
        if self.embedding is None:
            return False
        
        # Handle Paradma types
        if PARADMA_AVAILABLE and isinstance(self.embedding, (Axiom, TensorAxiom)):
            value = self.embedding.value if hasattr(self.embedding, 'value') else self.embedding
            if hasattr(value, '__len__'):
                return len(value) > 0
            return True
        
        # Handle NumPy
        return len(self.embedding) > 0
    
    def get_source(self) -> Optional[str]:
        """Get the source of this knowledge unit from metadata."""
        return self.metadata.get("source")
    
    def get_timestamp(self) -> Optional[str]:
        """Get the creation timestamp."""
        return self.metadata.get("created_at")
    
    def get_tags(self) -> list:
        """Get tags associated with this knowledge unit."""
        return self.metadata.get("tags", [])
    
    def add_tag(self, tag: str):
        """Add a tag to this knowledge unit."""
        if "tags" not in self.metadata:
            self.metadata["tags"] = []
        if tag not in self.metadata["tags"]:
            self.metadata["tags"].append(tag)
    
    def set_confidence(self, confidence: float):
        """Set the confidence score."""
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0 and 1, got {confidence}")
        self.confidence = confidence
    
    def get_embedding_as_numpy(self) -> Optional[np.ndarray]:
        """
        Get embedding as NumPy array regardless of internal type.
        
        Returns:
            NumPy array or None if no embedding
        """
        if not self.has_embedding():
            return None
        
        # Already NumPy
        if isinstance(self.embedding, np.ndarray):
            return self.embedding
        
        # Convert from Paradma if available
        if PARADMA_AVAILABLE and UTILS_AVAILABLE:
            if isinstance(self.embedding, (Axiom, TensorAxiom)):
                return ensure_numpy_type(self.embedding)
        
        # Fallback: try to convert
        return np.array(self.embedding)
    
    def get_embedding_as_paradma(self) -> Optional[Union['Axiom', 'TensorAxiom']]:
        """
        Get embedding as Paradma Axiom/TensorAxiom.
        
        Returns:
            Axiom/TensorAxiom or None if no embedding or Paradma unavailable
        """
        if not self.has_embedding():
            return None
        
        if not PARADMA_AVAILABLE:
            return None
        
        # Already Paradma
        if isinstance(self.embedding, (Axiom, TensorAxiom)):
            return self.embedding
        
        # Convert from NumPy
        if UTILS_AVAILABLE:
            if isinstance(self.embedding, np.ndarray):
                # Use TensorAxiom for multi-dimensional, Axiom for 1D
                if self.embedding.ndim > 1:
                    return numpy_to_tensor_axiom(self.embedding)
                else:
                    return numpy_to_axiom(self.embedding)
        
        return None
    
    def set_embedding_from_numpy(self, embedding: np.ndarray, use_paradma: bool = None):
        """
        Set embedding from NumPy array, optionally converting to Paradma.
        
        Args:
            embedding: NumPy array
            use_paradma: Whether to convert to Paradma (uses _use_paradma if None)
        """
        if use_paradma is None:
            use_paradma = self._use_paradma and PARADMA_AVAILABLE and UTILS_AVAILABLE
        
        if use_paradma:
            # Convert to Paradma
            if embedding.ndim > 1:
                self.embedding = numpy_to_tensor_axiom(embedding)
            else:
                self.embedding = numpy_to_axiom(embedding)
        else:
            # Keep as NumPy
            self.embedding = embedding
    
    def is_using_paradma(self) -> bool:
        """Check if this unit is using Paradma for embeddings."""
        return PARADMA_AVAILABLE and isinstance(self.embedding, (Axiom, TensorAxiom))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "unit_id": self.unit_id,
            "content": self.content if isinstance(self.content, str) else "<binary>",
            "has_embedding": self.has_embedding(),
            "embedding_shape": self.embedding.shape if self.has_embedding() else None,
            "metadata": self.metadata,
            "confidence": self.confidence,
        }
    
    def __repr__(self) -> str:
        """String representation."""
        content_preview = str(self.content)[:50] + "..." if len(str(self.content)) > 50 else str(self.content)
        return f"KnowledgeUnit(id={self.unit_id[:8]}, content='{content_preview}', confidence={self.confidence})"
