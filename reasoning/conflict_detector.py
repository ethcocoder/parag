"""
Conflict detection in retrieved knowledge.

Detects contradictions and inconsistencies in knowledge units.
"""

from typing import List, Tuple, Set, Optional
from parag.core.rag_state import RAGState, Fact
from parag.core.knowledge_unit import KnowledgeUnit


class ConflictDetector:
    """
    Detect conflicts in retrieved knowledge.
    
    Uses heuristics and pattern matching to identify contradictions.
    """
    
    def __init__(
        self,
        negation_words: Optional[List[str]] = None,
        similarity_threshold: float = 0.8,
    ):
        """
        Initialize ConflictDetector.
        
        Args:
            negation_words: Words indicating negation/contradiction
            similarity_threshold: Threshold for considering facts related
        """
        self.negation_words = negation_words or [
            "not", "never", "no", "don't", "doesn't", "didn't",
            "cannot", "can't", "won't", "shouldn't", "isn't", "aren't"
        ]
        self.similarity_threshold = similarity_threshold
    
    def detect_conflicts_in_state(self, state: RAGState) -> List[Tuple[str, str, str]]:
        """
        Detect conflicts in a RAGState.
        
        Args:
            state: RAGState to check
            
        Returns:
            List of tuples (fact_id_1, fact_id_2, reason)
        """
        conflicts = []
        facts_list = list(state.facts.items())
        
        for i, (id1, fact1) in enumerate(facts_list):
            for id2, fact2 in facts_list[i+1:]:
                conflict_reason = self._check_conflict(fact1, fact2)
                if conflict_reason:
                    conflicts.append((id1, id2, conflict_reason))
        
        return conflicts
    
    def detect_conflicts_in_units(
        self,
        units: List[KnowledgeUnit]
    ) -> List[Tuple[int, int, str]]:
        """
        Detect conflicts between knowledge units.
        
        Args:
            units: List of KnowledgeUnits
            
        Returns:
            List of tuples (index_1, index_2, reason)
        """
        conflicts = []
        
        for i, unit1 in enumerate(units):
            for j, unit2 in enumerate(units[i+1:], i+1):
                # Create temporary facts
                fact1 = Fact(
                    content=str(unit1.content),
                    source_units=[unit1.unit_id],
                    confidence=unit1.confidence or 0.5
                )
                fact2 = Fact(
                    content=str(unit2.content),
                    source_units=[unit2.unit_id],
                    confidence=unit2.confidence or 0.5
                )
                
                conflict_reason = self._check_conflict(fact1, fact2)
                if conflict_reason:
                    conflicts.append((i, j, conflict_reason))
        
        return conflicts
    
    def _check_conflict(self, fact1: Fact, fact2: Fact) -> Optional[str]:
        """
        Check if two facts conflict.
        
        Returns conflict reason if they conflict, None otherwise.
        """
        content1 = fact1.content.lower()
        content2 = fact2.content.lower()
        
        # Check for direct negation pattern
        if self._has_negation_conflict(content1, content2):
            return "negation_pattern"
        
        # Check for contradictory numeric values
        if self._has_numeric_conflict(content1, content2):
            return "numeric_contradiction"
        
        # Check for opposite statements about same entity
        if self._has_entity_conflict(content1, content2):
            return "entity_contradiction"
        
        return None
    
    def _has_negation_conflict(self, content1: str, content2: str) -> bool:
        """
        Check if one content negates the other.
        
        Example: "X is Y" vs "X is not Y"
        """
        c1_has_negation = any(word in content1 for word in self.negation_words)
        c2_has_negation = any(word in content2 for word in self.negation_words)
        
        # If one has negation and other doesn't, check for shared keywords
        if c1_has_negation != c2_has_negation:
            words1 = set(content1.split())
            words2 = set(content2.split())
            
            # Remove negation words for comparison
            words1 = {w for w in words1 if w not in self.negation_words}
            words2 = {w for w in words2 if w not in self.negation_words}
            
            common = words1 & words2
            
            # If they share many words, likely a negation conflict
            if len(common) >= 3:
                return True
        
        return False
    
    def _has_numeric_conflict(self, content1: str, content2: str) -> bool:
        """
        Check for contradictory numeric values.
        
        Example: "population is 100" vs "population is 200"
        """
        import re
        
        # Extract numbers
        numbers1 = set(re.findall(r'\d+(?:\.\d+)?', content1))
        numbers2 = set(re.findall(r'\d+(?:\.\d+)?', content2))
        
        # If they have different numbers but similar text structure
        if numbers1 and numbers2 and numbers1 != numbers2:
            # Check if the text without numbers is similar
            text1_no_nums = re.sub(r'\d+(?:\.\d+)?', 'NUM', content1)
            text2_no_nums = re.sub(r'\d+(?:\.\d+)?', 'NUM', content2)
            
            # Simple similarity check
            words1 = set(text1_no_nums.split())
            words2 = set(text2_no_nums.split())
            
            if len(words1 & words2) >= 3:
                return True
        
        return False
    
    def _has_entity_conflict(self, content1: str, content2: str) -> bool:
        """
        Check for opposite statements about same entity.
        
        This is a placeholder for more sophisticated NLP-based detection.
        """
        # Simple heuristic: look for opposite adjectives
        opposites = [
            ("true", "false"),
            ("correct", "incorrect"),
            ("valid", "invalid"),
            ("possible", "impossible"),
            ("legal", "illegal"),
            ("safe", "unsafe"),
            ("hot", "cold"),
            ("large", "small"),
        ]
        
        for word1, word2 in opposites:
            if word1 in content1 and word2 in content2:
                # Check if they're talking about same thing
                words1 = set(content1.split())
                words2 = set(content2.split())
                
                if len(words1 & words2) >= 2:
                    return True
        
        return False
    
    def resolve_conflict(
        self,
        fact1: Fact,
        fact2: Fact,
        strategy: str = "highest_confidence"
    ) -> Fact:
        """
        Resolve a conflict between two facts.
        
        Args:
            fact1: First conflicting fact
            fact2: Second conflicting fact
            strategy: Resolution strategy ('highest_confidence', 'most_sources', 'newest')
            
        Returns:
            The "winning" fact based on strategy
        """
        if strategy == "highest_confidence":
            return fact1 if fact1.confidence >= fact2.confidence else fact2
        
        elif strategy == "most_sources":
            return fact1 if fact1.supporting_count >= fact2.supporting_count else fact2
        
        elif strategy == "newest":
            # Placeholder - would need timestamp comparison
            return fact1
        
        else:
            raise ValueError(f"Unknown resolution strategy: {strategy}")
