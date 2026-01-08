"""
RAGState: Internal state representation for reasoning.

Maintains structured state from retrieved knowledge,
enabling conflict detection, uncertainty measurement,
and rule-based reasoning without LLM prompts.
"""

from typing import List, Dict, Any, Set, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

from parag.core.knowledge_unit import KnowledgeUnit
from parag.core.retrieval_result import RetrievalResult


@dataclass
class Fact:
    """A single fact extracted from knowledge units."""
    content: str
    source_units: List[str]  # IDs of supporting knowledge units
    confidence: float
    supporting_count: int = 1
    contradicting_facts: Set[str] = field(default_factory=set)


@dataclass
class RAGState:
    """
    Internal state built from retrieved knowledge.
    
    This state enables reasoning without directly prompting an LLM.
    It aggregates facts, detects conflicts, and measures uncertainty.
    
    Attributes:
        facts: Aggregated facts from knowledge units
        knowledge_units: All contributing knowledge units
        conflicts: Detected contradictions
        uncertainty: Uncertainty measurement
        metadata: Additional state information
    """
    
    facts: Dict[str, Fact] = field(default_factory=dict)
    knowledge_units: List[KnowledgeUnit] = field(default_factory=list)
    conflicts: List[Tuple[str, str]] = field(default_factory=list)  # Pairs of conflicting fact IDs
    uncertainty: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize state metadata."""
        if "created_at" not in self.metadata:
            self.metadata["created_at"] = datetime.now().isoformat()
    
    @classmethod
    def from_retrieval_result(cls, result: RetrievalResult) -> 'RAGState':
        """
        Build RAGState from a RetrievalResult.
        
        Args:
            result: RetrievalResult containing knowledge units
            
        Returns:
            RAGState with aggregated facts
        """
        state = cls()
        state.knowledge_units = result.units
        
        # Extract facts from knowledge units
        for unit in result.units:
            fact_id = unit.unit_id
            content = str(unit.content)
            
            # Create fact from knowledge unit
            fact = Fact(
                content=content,
                source_units=[unit.unit_id],
                confidence=unit.confidence if unit.confidence else 0.5
            )
            
            state.facts[fact_id] = fact
        
        # Calculate initial uncertainty
        state._calculate_uncertainty()
        
        state.metadata["source_query"] = result.query
        state.metadata["num_units"] = len(result.units)
        
        return state
    
    def add_knowledge_unit(self, unit: KnowledgeUnit):
        """Add a knowledge unit to the state."""
        self.knowledge_units.append(unit)
        
        # Create or update fact
        fact_id = unit.unit_id
        content = str(unit.content)
        
        if fact_id in self.facts:
            # Update existing fact
            fact = self.facts[fact_id]
            if unit.unit_id not in fact.source_units:
                fact.source_units.append(unit.unit_id)
                fact.supporting_count += 1
        else:
            # Create new fact
            fact = Fact(
                content=content,
                source_units=[unit.unit_id],
                confidence=unit.confidence if unit.confidence else 0.5
            )
            self.facts[fact_id] = fact
        
        # Recalculate uncertainty
        self._calculate_uncertainty()
    
    def detect_conflicts(self, similarity_threshold: float = 0.8) -> List[Tuple[str, str]]:
        """
        Detect contradictions between facts.
        
        Simple heuristic: facts are potentially conflicting if they have
        similar topics but different content patterns.
        
        Args:
            similarity_threshold: Threshold for considering facts related
            
        Returns:
            List of tuples containing IDs of conflicting facts
        """
        conflicts = []
        fact_items = list(self.facts.items())
        
        for i, (id1, fact1) in enumerate(fact_items):
            for id2, fact2 in fact_items[i+1:]:
                # Simple conflict detection: check for negation words
                # or contradictory patterns
                if self._are_conflicting(fact1.content, fact2.content):
                    conflicts.append((id1, id2))
                    fact1.contradicting_facts.add(id2)
                    fact2.contradicting_facts.add(id1)
        
        self.conflicts = conflicts
        return conflicts
    
    def _are_conflicting(self, content1: str, content2: str) -> bool:
        """
        Simple heuristic to detect if two fact contents conflict.
        
        This is a basic implementation. In production, this would use
        more sophisticated NLP techniques or even LLM-based verification.
        """
        # Normalize
        c1 = content1.lower()
        c2 = content2.lower()
        
        # Check for negation patterns
        negation_words = ["not", "never", "no", "don't", "doesn't", "didn't", "cannot"]
        
        # If one has negation and other doesn't, but they share keywords
        c1_has_negation = any(word in c1 for word in negation_words)
        c2_has_negation = any(word in c2 for word in negation_words)
        
        if c1_has_negation != c2_has_negation:
            # Check if they share significant keywords
            words1 = set(c1.split())
            words2 = set(c2.split())
            common = words1 & words2
            
            # If they share more than 3 words, consider them potentially conflicting
            if len(common) > 3:
                return True
        
        return False
    
    def _calculate_uncertainty(self):
        """
        Calculate uncertainty based on state properties.
        
        Uncertainty increases with:
        - Low confidence scores
        - Conflicting facts
        - Few knowledge units
        """
        if not self.facts:
            self.uncertainty = 1.0
            return
        
        # Calculate average confidence
        confidences = [fact.confidence for fact in self.facts.values()]
        avg_confidence = np.mean(confidences)
        
        # Penalize for conflicts
        conflict_penalty = len(self.conflicts) * 0.1
        
        # Penalize for insufficient knowledge
        knowledge_penalty = 0.3 if len(self.knowledge_units) < 3 else 0.0
        
        # Combine factors
        uncertainty = (1.0 - avg_confidence) + conflict_penalty + knowledge_penalty
        self.uncertainty = min(1.0, max(0.0, uncertainty))
    
    def get_high_confidence_facts(self, threshold: float = 0.7) -> List[Fact]:
        """Get facts with confidence above threshold."""
        return [
            fact for fact in self.facts.values()
            if fact.confidence >= threshold
        ]
    
    def get_conflicting_facts(self) -> List[Fact]:
        """Get facts that have conflicts."""
        return [
            fact for fact in self.facts.values()
            if len(fact.contradicting_facts) > 0
        ]
    
    def has_sufficient_information(self, min_units: int = 2, max_uncertainty: float = 0.5) -> bool:
        """
        Check if state has sufficient information to answer confidently.
        
        Args:
            min_units: Minimum number of knowledge units required
            max_uncertainty: Maximum acceptable uncertainty
            
        Returns:
            True if state has sufficient information
        """
        return (
            len(self.knowledge_units) >= min_units and
            self.uncertainty <= max_uncertainty
        )
    
    def merge(self, other: 'RAGState') -> 'RAGState':
        """Merge with another RAGState."""
        merged = RAGState()
        
        # Merge knowledge units
        merged.knowledge_units = self.knowledge_units + other.knowledge_units
        
        # Merge facts
        merged.facts = {**self.facts, **other.facts}
        
        # Recalculate conflicts and uncertainty
        merged.detect_conflicts()
        merged._calculate_uncertainty()
        
        merged.metadata["merged"] = True
        
        return merged
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "num_facts": len(self.facts),
            "num_knowledge_units": len(self.knowledge_units),
            "num_conflicts": len(self.conflicts),
            "uncertainty": self.uncertainty,
            "has_sufficient_info": self.has_sufficient_information(),
            "high_confidence_facts": len(self.get_high_confidence_facts()),
            "metadata": self.metadata,
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RAGState("
            f"facts={len(self.facts)}, "
            f"units={len(self.knowledge_units)}, "
            f"conflicts={len(self.conflicts)}, "
            f"uncertainty={self.uncertainty:.2f})"
        )
