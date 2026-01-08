"""
RetrievalResult: Structured container for retrieval outputs.

Provides a clean abstraction for similarity search results,
including knowledge units, scores, and ranking information.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import Paradma
try:
    from paradma import learning, Axiom
    PARADMA_AVAILABLE = True
except ImportError:
    PARADMA_AVAILABLE = False
    learning = None

from parag.core.knowledge_unit import KnowledgeUnit


@dataclass
class RetrievalResult:
    """
    Container for retrieval results.
    
    Attributes:
        units: List of retrieved KnowledgeUnits
        scores: Similarity/relevance scores for each unit
        query: The original query string
        query_embedding: Embedding of the query
        metadata: Additional retrieval metadata
    """
    
    units: List[KnowledgeUnit] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    query: Optional[str] = None
    query_embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate retrieval result."""
        if len(self.units) != len(self.scores):
            raise ValueError(
                f"Number of units ({len(self.units)}) must match "
                f"number of scores ({len(self.scores)})"
            )
    
    def __len__(self) -> int:
        """Return number of results."""
        return len(self.units)
    
    def is_empty(self) -> bool:
        """Check if result is empty."""
        return len(self.units) == 0
    
    def get_top_k(self, k: int) -> 'RetrievalResult':
        """Get top k results."""
        if k >= len(self.units):
            return self
        
        return RetrievalResult(
            units=self.units[:k],
            scores=self.scores[:k],
            query=self.query,
            query_embedding=self.query_embedding,
            metadata={**self.metadata, "top_k": k}
        )
    
    def filter_by_threshold(self, threshold: float) -> 'RetrievalResult':
        """Filter results by minimum score threshold."""
        filtered_units = []
        filtered_scores = []
        
        for unit, score in zip(self.units, self.scores):
            if score >= threshold:
                filtered_units.append(unit)
                filtered_scores.append(score)
        
        return RetrievalResult(
            units=filtered_units,
            scores=filtered_scores,
            query=self.query,
            query_embedding=self.query_embedding,
            metadata={**self.metadata, "threshold": threshold}
        )
    
    def get_sources(self) -> List[str]:
        """Get all unique sources from retrieved units."""
        sources = set()
        for unit in self.units:
            source = unit.get_source()
            if source:
                sources.add(source)
        return sorted(list(sources))
    
    def get_average_score(self) -> float:
        """Get average similarity score."""
        if not self.scores:
            return 0.0
        
        # Use Paradma if available (self-learning!)
        if PARADMA_AVAILABLE and learning:
            try:
                scores_axiom = Axiom(self.scores, manifold=learning)
                mean_result = learning.mean(scores_axiom)
                return float(mean_result.value if hasattr(mean_result, 'value') else mean_result)
            except:
                pass  # Fall back to NumPy
        
        return float(np.mean(self.scores))
    
    def get_max_score(self) -> float:
        """Get maximum retrieval score."""
        if not self.scores:
            return 0.0
        return float(max(self.scores))
    
    def get_min_score(self) -> float:
        """Get minimum retrieval score."""
        if not self.scores:
            return 0.0
        return float(min(self.scores))
    
    def has_high_confidence(self, threshold: float = 0.7) -> bool:
        """Check if at least one result has high confidence."""
        return any(score >= threshold for score in self.scores)
    
    def get_content_list(self) -> List[str]:
        """Get list of content strings from all units."""
        return [str(unit.content) for unit in self.units]
    
    def merge(self, other: 'RetrievalResult') -> 'RetrievalResult':
        """Merge with another retrieval result."""
        if self.query != other.query:
            raise ValueError("Cannot merge results from different queries")
        
        # Combine and sort by score
        all_units = self.units + other.units
        all_scores = self.scores + other.scores
        
        # Sort by score (descending)
        sorted_pairs = sorted(
            zip(all_units, all_scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        sorted_units, sorted_scores = zip(*sorted_pairs) if sorted_pairs else ([], [])
        
        return RetrievalResult(
            units=list(sorted_units),
            scores=list(sorted_scores),
            query=self.query,
            query_embedding=self.query_embedding,
            metadata={**self.metadata, "merged": True}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "query": self.query,
            "num_results": len(self),
            "units": [unit.to_dict() for unit in self.units],
            "scores": self.scores,
            "average_score": self.get_average_score(),
            "sources": self.get_sources(),
            "metadata": self.metadata,
        }
    
    def __repr__(self) -> str:
        """String representation."""
        query_preview = self.query[:30] + "..." if self.query and len(self.query) > 30 else self.query
        return (
            f"RetrievalResult(query='{query_preview}', "
            f"num_results={len(self)}, "
            f"avg_score={self.get_average_score():.3f})"
        )
