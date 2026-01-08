"""
Uncertainty quantification for RAG systems.

Measures confidence and completeness of retrieved information.
"""

from typing import List, Dict, Any
import numpy as np

from parag.core.rag_state import RAGState
from parag.core.retrieval_result import RetrievalResult
from parag.core.knowledge_unit import KnowledgeUnit


class UncertaintyCalculator:
    """
    Calculate uncertainty in retrieved knowledge.
    
    Provides multiple uncertainty metrics:
    - Retrieval confidence
    - Information completeness
    - Conflict-based uncertainty
    """
    
    def __init__(
        self,
        min_units_threshold: int = 3,
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize UncertaintyCalculator.
        
        Args:
            min_units_threshold: Minimum knowledge units for "complete" information
            confidence_threshold: Threshold for "high confidence"
        """
        self.min_units_threshold = min_units_threshold
        self.confidence_threshold = confidence_threshold
    
    def calculate_retrieval_uncertainty(
        self,
        result: RetrievalResult
    ) -> Dict[str, float]:
        """
        Calculate uncertainty from retrieval result.
        
        Args:
            result: RetrievalResult to analyze
            
        Returns:
            Dictionary with uncertainty metrics
        """
        if result.is_empty():
            return {
                "overall_uncertainty": 1.0,
                "score_uncertainty": 1.0,
                "coverage_uncertainty": 1.0,
                "confidence": 0.0,
            }
        
        # Score-based uncertainty (inverse of average score)
        avg_score = result.get_average_score()
        score_uncertainty = 1.0 - avg_score
        
        # Coverage uncertainty (based on number of results)
        num_results = len(result)
        coverage_uncertainty = max(0.0, 1.0 - num_results / self.min_units_threshold)
        
        # Overall uncertainty (weighted combination)
        overall_uncertainty = 0.6 * score_uncertainty + 0.4 * coverage_uncertainty
        
        # Confidence (inverse of uncertainty)
        confidence = 1.0 - overall_uncertainty
        
        return {
            "overall_uncertainty": float(overall_uncertainty),
            "score_uncertainty": float(score_uncertainty),
            "coverage_uncertainty": float(coverage_uncertainty),
            "confidence": float(confidence),
        }
    
    def calculate_state_uncertainty(self, state: RAGState) -> Dict[str, float]:
        """
        Calculate uncertainty from RAGState.
        
        Args:
            state: RAGState to analyze
            
        Returns:
            Dictionary with uncertainty metrics
        """
        # Fact confidence uncertainty
        if state.facts:
            confidences = [fact.confidence for fact in state.facts.values()]
            avg_confidence = np.mean(confidences)
            confidence_std = np.std(confidences)
            
            confidence_uncertainty = 1.0 - avg_confidence
        else:
            confidence_uncertainty = 1.0
            confidence_std = 0.0
        
        # Conflict-based uncertainty
        num_facts = len(state.facts)
        num_conflicts = len(state.conflicts)
        
        if num_facts > 0:
            conflict_rate = num_conflicts / num_facts
            conflict_uncertainty = min(1.0, conflict_rate * 2)  # Scale to [0, 1]
        else:
            conflict_uncertainty = 0.5  # Neutral if no facts
        
        # Coverage uncertainty
        coverage_uncertainty = max(
            0.0,
            1.0 - len(state.knowledge_units) / self.min_units_threshold
        )
        
        # Overall uncertainty
        overall_uncertainty = (
            0.4 * confidence_uncertainty +
            0.3 * conflict_uncertainty +
            0.3 * coverage_uncertainty
        )
        
        return {
            "overall_uncertainty": float(overall_uncertainty),
            "confidence_uncertainty": float(confidence_uncertainty),
            "conflict_uncertainty": float(conflict_uncertainty),
            "coverage_uncertainty": float(coverage_uncertainty),
            "confidence_std": float(confidence_std),
            "num_conflicts": num_conflicts,
        }
    
    def is_information_sufficient(
        self,
        state: RAGState,
        max_uncertainty: float = 0.5,
    ) -> bool:
        """
        Check if state has sufficient information.
        
        Args:
            state: RAGState to check
            max_uncertainty: Maximum acceptable uncertainty
            
        Returns:
            True if information is sufficient
        """
        uncertainty = self.calculate_state_uncertainty(state)
        return uncertainty["overall_uncertainty"] <= max_uncertainty
    
    def detect_missing_information(self, state: RAGState) -> List[str]:
        """
        Detect potential gaps in information.
        
        Returns:
            List of detected issues/gaps
        """
        issues = []
        
        # Check for insufficient knowledge units
        if len(state.knowledge_units) < self.min_units_threshold:
            issues.append(
                f"Insufficient knowledge units "
                f"(have {len(state.knowledge_units)}, need {self.min_units_threshold})"
            )
        
        # Check for low confidence facts
        low_confidence_facts = [
            fact for fact in state.facts.values()
            if fact.confidence < self.confidence_threshold
        ]
        
        if low_confidence_facts:
            issues.append(
                f"{len(low_confidence_facts)} facts have low confidence "
                f"(< {self.confidence_threshold})"
            )
        
        # Check for conflicts
        if state.conflicts:
            issues.append(f"{len(state.conflicts)} conflicts detected")
        
        # Check for high uncertainty
        uncertainty = self.calculate_state_uncertainty(state)
        if uncertainty["overall_uncertainty"] > 0.7:
            issues.append(
                f"High overall uncertainty ({uncertainty['overall_uncertainty']:.2f})"
            )
        
        return issues
    
    def get_confidence_breakdown(
        self,
        state: RAGState
    ) -> Dict[str, Any]:
        """
        Get detailed confidence breakdown for a state.
        
        Returns:
            Dictionary with confidence information
        """
        if not state.facts:
            return {
                "num_facts": 0,
                "confidence_distribution": {},
                "high_confidence_count": 0,
                "low_confidence_count": 0,
            }
        
        confidences = [fact.confidence for fact in state.facts.values()]
        
        # Categorize by confidence level
        high_confidence = sum(1 for c in confidences if c >= 0.7)
        medium_confidence = sum(1 for c in confidences if 0.4 <= c < 0.7)
        low_confidence = sum(1 for c in confidences if c < 0.4)
        
        return {
            "num_facts": len(state.facts),
            "average_confidence": float(np.mean(confidences)),
            "min_confidence": float(min(confidences)),
            "max_confidence": float(max(confidences)),
            "confidence_std": float(np.std(confidences)),
            "confidence_distribution": {
                "high (≥0.7)": high_confidence,
                "medium (0.4-0.7)": medium_confidence,
                "low (<0.4)": low_confidence,
            },
            "high_confidence_count": high_confidence,
            "low_confidence_count": low_confidence,
        }
