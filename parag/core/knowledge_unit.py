"""
KnowledgeUnit: The fundamental data structure for retrieved information.

All knowledge in the RAG system is represented as KnowledgeUnits,
providing a standard interface for content, embeddings, and metadata.
"""

from typing import Any, Dict, Optional, Union
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime


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
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: Optional[float] = None
    unit_id: Optional[str] = None
    
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
        return self.embedding is not None and len(self.embedding) > 0
    
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
