"""
CognitiveLoop: Autonomous feedback loop for the Paradox RAG system.

Enables self-directed information seeking based on internal uncertainty and emotions.
"""

import logging
from typing import List, Optional, Dict, Any
from parag.core import RAGState, RetrievalResult
from parag.retrieval import Retriever
from parag.reasoning import StateManager, UncertaintyCalculator

logger = logging.getLogger(__name__)

class CognitiveLoop:
    """
    Orchestrates the autonomous sentience loop.
    
    If retrieval results are uncertain or conflicting, this loop 
    automatically triggers reflexive queries until cognitive equilibrium is reached.
    """
    
    def __init__(
        self,
        retriever: Retriever,
        state_manager: Optional[StateManager] = None,
        max_turns: int = 2,
        uncertainty_threshold: float = 0.45
    ):
        self.retriever = retriever
        self.state_manager = state_manager or StateManager()
        self.max_turns = max_turns
        self.uncertainty_threshold = uncertainty_threshold
        self.history = []

    def run(self, query: str) -> RAGState:
        """
        Executes the autonomous loop for a given query.
        
        Args:
            query: User's initial query
            
        Returns:
            Final RAGState after autonomous refinement
        """
        logger.info(f"🧠 Cognitive Loop started for query: '{query}'")
        
        # Reset state manager for new query
        self.state_manager.reset_state()
        self.history = []
        
        # Turn 0: Initial Retrieval
        result = self.retriever.retrieve(query)
        state = self.state_manager.build_from_retrieval(result)
        state.detect_conflicts()
        
        self.history.append({
            "turn": 0,
            "query": query,
            "uncertainty": float(state.uncertainty),
            "num_units": len(state.knowledge_units)
        })

        # Reflexive Loop
        for turn in range(1, self.max_turns + 1):
            if self._is_equilibrium_reached(state):
                logger.info(f"✨ Equilibrium reached at turn {turn-1}")
                break
                
            logger.info(f"🔄 Low confidence (U={state.uncertainty:.2f}). Triggering Reflexion turn {turn}...")
            
            # 1. Generate reflexive query
            reflexive_query = self._generate_reflexive_query(query, state)
            logger.info(f"💭 Curious logic: '{reflexive_query}'")
            
            # 2. Perform secondary retrieval
            new_result = self.retriever.retrieve(reflexive_query)
            
            # 3. Merge knowledge into existing state
            state = self.state_manager.update_state(new_result.units)
            state.detect_conflicts()
            
            self.history.append({
                "turn": turn,
                "query": reflexive_query,
                "uncertainty": float(state.uncertainty),
                "num_units": len(state.knowledge_units)
            })

        # Save history to state metadata
        state.metadata["cognitive_loop_history"] = self.history
        return state

    def _is_equilibrium_reached(self, state: RAGState) -> bool:
        """Determines if the system is satisfied with current knowledge."""
        # Condition 1: Low uncertainty
        if state.uncertainty < self.uncertainty_threshold:
            # Condition 2: Emotional stability (Equilibria > 0.6)
            if state.emotions:
                eq = state.emotions.get_state().get("Equilibria", 0.0)
                if eq > 0.6:
                    return True
            else:
                return True
        return False

    def _generate_reflexive_query(self, original_query: str, state: RAGState) -> str:
        """Formulates a new search query based on missing data or conflicts."""
        calc = UncertaintyCalculator()
        issues = calc.detect_missing_information(state)
        
        if state.conflicts:
            # Conflict-driven curiosity
            id1, id2 = state.conflicts[0]
            content1 = state.facts[id1].content if id1 in state.facts else id1
            content2 = state.facts[id2].content if id2 in state.facts else id2
            
            # Use shorter snippets for the query
            q1 = content1[:60] + "..." if len(content1) > 60 else content1
            q2 = content2[:60] + "..." if len(content2) > 60 else content2
            
            return f"{original_query} evidence for '{q1}' vs '{q2}'"
        
        if issues:
            # Missing-info curiosity
            issue = issues[0]
            # Strip boilerplate from issues list if possible
            clean_issue = issue.split("(")[0].strip() if "(" in issue else issue
            return f"Information about {original_query} specifically regarding {clean_issue}"
            
        # Fallback expansion
        return f"Expanded context for {original_query}"
