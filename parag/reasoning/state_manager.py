"""
RAG state management.

Manages the construction and evolution of RAGState from retrieval results.
"""

from typing import List, Optional
from parag.core.rag_state import RAGState
from parag.core.retrieval_result import RetrievalResult
from parag.core.knowledge_unit import KnowledgeUnit


class StateManager:
    """
    Manage RAGState construction and updates.
    
    Provides high-level interface for building and evolving state.
    """
    
    def __init__(self):
        """Initialize StateManager."""
        self.current_state: Optional[RAGState] = None
    
    def build_from_retrieval(self, result: RetrievalResult) -> RAGState:
        """
        Build RAGState from retrieval result.
        
        Args:
            result: RetrievalResult to build state from
            
        Returns:
            RAGState containing aggregated facts
        """
        state = RAGState.from_retrieval_result(result)
        self.current_state = state
        return state
    
    def update_state(
        self,
        units: List[KnowledgeUnit],
        detect_conflicts: bool = True,
    ) -> RAGState:
        """
        Update current state with new knowledge units.
        
        Args:
            units: New knowledge units to add
            detect_conflicts: Whether to run conflict detection after update
            
        Returns:
            Updated RAGState
        """
        if self.current_state is None:
            self.current_state = RAGState()
        
        # Add each unit
        for unit in units:
            self.current_state.add_knowledge_unit(unit)
        
        # Detect conflicts if requested
        if detect_conflicts:
            self.current_state.detect_conflicts()
        
        return self.current_state
    
    def merge_states(self, *states: RAGState) -> RAGState:
        """
        Merge multiple RAGStates.
        
        Args:
            states: RAGStates to merge
            
        Returns:
            Merged RAGState
        """
        if not states:
            return RAGState()
        
        merged = states[0]
        for state in states[1:]:
            merged = merged.merge(state)
        
        self.current_state = merged
        return merged
    
    def get_current_state(self) -> Optional[RAGState]:
        """Get the current state."""
        return self.current_state
    
    def reset_state(self):
        """Reset to empty state."""
        self.current_state = None
    
    def state_summary(self) -> dict:
        """Get summary of current state."""
        if self.current_state is None:
            return {"status": "no_state"}
        
        return self.current_state.to_dict()
