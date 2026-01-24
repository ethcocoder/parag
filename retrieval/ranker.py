"""
Result ranking and filtering.

Provides advanced ranking strategies for retrieval results.
"""

from typing import List, Callable, Optional
from datetime import datetime
import numpy as np

from parag.core.retrieval_result import RetrievalResult
from parag.core.knowledge_unit import KnowledgeUnit


class ResultRanker:
    """
    Rank and filter retrieval results.
    
    Supports multiple ranking strategies including:
    - Score-based ranking
    - Source diversity
    - Recency weighting
    - Custom ranking functions
    """
    
    def __init__(
        self,
        diversity_weight: float = 0.0,
        recency_weight: float = 0.0,
    ):
        """
        Initialize ResultRanker.
        
        Args:
            diversity_weight: Weight for source diversity (0-1)
            recency_weight: Weight for recency (0-1)
        """
        self.diversity_weight = diversity_weight
        self.recency_weight = recency_weight
    
    def rank(self, result: RetrievalResult) -> RetrievalResult:
        """
        Re-rank a retrieval result.
        
        Args:
            result: RetrievalResult to re-rank
            
        Returns:
            Re-ranked RetrievalResult
        """
        if result.is_empty():
            return result
        
        # Calculate combined scores
        new_scores = []
        
        for i, (unit, score) in enumerate(zip(result.units, result.scores)):
            # Base similarity score
            combined_score = score
            
            # Add diversity bonus
            if self.diversity_weight > 0:
                diversity_score = self._calculate_diversity_score(unit, result.units[:i])
                combined_score += self.diversity_weight * diversity_score
            
            # Add recency bonus
            if self.recency_weight > 0:
                recency_score = self._calculate_recency_score(unit)
                combined_score += self.recency_weight * recency_score
            
            new_scores.append(combined_score)
        
        # Sort by new scores
        sorted_pairs = sorted(
            zip(result.units, new_scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        sorted_units, sorted_scores = zip(*sorted_pairs) if sorted_pairs else ([], [])
        
        return RetrievalResult(
            units=list(sorted_units),
            scores=list(sorted_scores),
            query=result.query,
            query_embedding=result.query_embedding,
            metadata={**result.metadata, "reranked": True}
        )
    
    def _calculate_diversity_score(
        self,
        unit: KnowledgeUnit,
        previous_units: List[KnowledgeUnit]
    ) -> float:
        """
        Calculate diversity score for a unit.
        
        Awards higher scores for units from different sources.
        """
        if not previous_units:
            return 1.0
        
        unit_source = unit.get_source()
        previous_sources = set(u.get_source() for u in previous_units)
        
        # Higher score if source is different from previous results
        if unit_source not in previous_sources:
            return 1.0
        else:
            # Penalize repeated sources
            return 0.5
    
    def _calculate_recency_score(self, unit: KnowledgeUnit) -> float:
        """
        Calculate recency score for a unit.
        
        More recent units get higher scores.
        """
        try:
            created_at = unit.metadata.get("created_at")
            if not created_at:
                return 0.5  # Neutral score for unknown dates
            
            # Parse timestamp
            if isinstance(created_at, str):
                created_dt = datetime.fromisoformat(created_at)
            else:
                created_dt = created_at
            
            # Calculate age in days
            age_days = (datetime.now() - created_dt).total_seconds() / (24 * 3600)
            
            # Decay function: score = e^(-age_days / 30)
            # Units from today: ~1.0, units from 30 days ago: ~0.37
            recency_score = np.exp(-age_days / 30.0)
            return float(recency_score)
        
        except Exception:
            return 0.5  # Neutral score on error
    
    def filter_by_confidence(
        self,
        result: RetrievalResult,
        min_confidence: float = 0.5,
    ) -> RetrievalResult:
        """Filter results by minimum confidence score."""
        return result.filter_by_threshold(min_confidence)
    
    def promote_source_diversity(
        self,
        result: RetrievalResult,
        max_per_source: int = 2,
    ) -> RetrievalResult:
        """
        Limit results per source to promote diversity.
        
        Args:
            result: RetrievalResult to filter
            max_per_source: Maximum results per unique source
            
        Returns:
            Filtered RetrievalResult
        """
        if result.is_empty():
            return result
        
        source_counts = {}
        filtered_units = []
        filtered_scores = []
        
        for unit, score in zip(result.units, result.scores):
            source = unit.get_source()
            
            if source not in source_counts:
                source_counts[source] = 0
            
            if source_counts[source] < max_per_source:
                filtered_units.append(unit)
                filtered_scores.append(score)
                source_counts[source] += 1
        
        return RetrievalResult(
            units=filtered_units,
            scores=filtered_scores,
            query=result.query,
            query_embedding=result.query_embedding,
            metadata={**result.metadata, "diversity_filtered": True}
        )
    
    def custom_rank(
        self,
        result: RetrievalResult,
        scoring_fn: Callable[[KnowledgeUnit, float], float],
    ) -> RetrievalResult:
        """
        Apply custom ranking function.
        
        Args:
            result: RetrievalResult to rank
            scoring_fn: Function that takes (unit, original_score) and returns new score
            
        Returns:
            Re-ranked RetrievalResult
        """
        new_scores = [
            scoring_fn(unit, score)
            for unit, score in zip(result.units, result.scores)
        ]
        
        # Sort by new scores
        sorted_pairs = sorted(
            zip(result.units, new_scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        sorted_units, sorted_scores = zip(*sorted_pairs) if sorted_pairs else ([], [])
        
        return RetrievalResult(
            units=list(sorted_units),
            scores=list(sorted_scores),
            query=result.query,
            query_embedding=result.query_embedding,
            metadata={**result.metadata, "custom_ranked": True}
        )
