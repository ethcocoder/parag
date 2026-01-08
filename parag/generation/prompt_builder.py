"""
Prompt building for LLM generation.

Constructs prompts from RAGState and retrieval results.
"""

from typing import List, Optional, Dict, Any
from parag.core.rag_state import RAGState
from parag.core.retrieval_result import RetrievalResult
from parag.core.knowledge_unit import KnowledgeUnit


class PromptBuilder:
    """
    Build prompts for LLM generation.
    
    Converts structured state and retrieval results into 
    formatted prompts for language models.
    """
    
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        include_sources: bool = True,
        include_confidence: bool = False,
    ):
        """
        Initialize PromptBuilder.
        
        Args:
            system_prompt: Optional system prompt to prepend
            include_sources: Whether to include source information
            include_confidence: Whether to include confidence scores
        """
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.include_sources = include_sources
        self.include_confidence = include_confidence
    
    def _default_system_prompt(self) -> str:
        """Default system prompt."""
        return (
            "You are a helpful AI assistant. "
            "Answer questions based on the provided context. "
            "If the information is insufficient, say so clearly."
        )
    
    def build_from_result(
        self,
        query: str,
        result: RetrievalResult,
    ) -> str:
        """
        Build prompt from retrieval result.
        
        Args:
            query: User query
            result: RetrievalResult with relevant knowledge
            
        Returns:
            Formatted prompt string
        """
        context_parts = []
        
        for i, (unit, score) in enumerate(zip(result.units, result.scores)):
            context_part = f"[{i+1}] {unit.content}"
            
            if self.include_sources and unit.get_source():
                context_part += f"\n   Source: {unit.get_source()}"
            
            if self.include_confidence:
                context_part += f"\n   Confidence: {score:.2f}"
            
            context_parts.append(context_part)
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""{self.system_prompt}

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    def build_from_state(
        self,
        query: str,
        state: RAGState,
    ) -> str:
        """
        Build prompt from RAGState.
        
        Args:
            query: User query
            state: RAGState with aggregated knowledge
            
        Returns:
            Formatted prompt string
        """
        # Build context from facts
        context_parts = []
        
        for i, (fact_id, fact) in enumerate(state.facts.items()):
            context_part = f"[{i+1}] {fact.content}"
            
            if self.include_confidence:
                context_part += f"\n   Confidence: {fact.confidence:.2f}"
                context_part += f"\n   Supporting sources: {fact.supporting_count}"
            
            # Indicate conflicts
            if fact.contradicting_facts:
                context_part += f"\n   ⚠️ Conflicts with {len(fact.contradicting_facts)} other facts"
            
            context_parts.append(context_part)
        
        context = "\n\n".join(context_parts)
        
        # Add uncertainty information
        uncertainty_note = ""
        if state.uncertainty > 0.5:
            uncertainty_note = f"\n\nNote: Information uncertainty is high ({state.uncertainty:.2f})"
        
        if state.conflicts:
            uncertainty_note += f"\nDetected {len(state.conflicts)} conflicts in the information."
        
        prompt = f"""{self.system_prompt}

Context:
{context}{uncertainty_note}

Question: {query}

Answer:"""
        
        return prompt
    
    def build_deterministic_response(
        self,
        query: str,
        state: RAGState,
    ) -> str:
        """
        Build deterministic response without LLM.
        
        Args:
            query: User query
            state: RAGState
            
        Returns:
            Formatted response string
        """
        if not state.has_sufficient_information():
            return self._build_insufficient_data_response(query, state)
        
        # Get high confidence facts
        high_conf_facts = state.get_high_confidence_facts()
        
        if not high_conf_facts:
            return self._build_low_confidence_response(query, state)
        
        # Build response from facts
        response_parts = [f"Based on the available information regarding '{query}':"]
        
        for fact in high_conf_facts:
            response_parts.append(f"- {fact.content}")
        
        # Add confidence summary
        avg_confidence = sum(f.confidence for f in high_conf_facts) / len(high_conf_facts)
        response_parts.append(f"\nAverage confidence: {avg_confidence:.2f}")
        
        # Add conflict warning if needed
        if state.conflicts:
            response_parts.append(
                f"\n⚠️ Warning: {len(state.conflicts)} conflicting pieces of information detected."
            )
        
        return "\n".join(response_parts)
    
    def _build_insufficient_data_response(
        self,
        query: str,
        state: RAGState,
    ) -> str:
        """Build response for insufficient data."""
        return (
            f"I don't have sufficient information to answer '{query}' confidently.\n\n"
            f"Current state:\n"
            f"- Knowledge units: {len(state.knowledge_units)}\n"
            f"- Uncertainty: {state.uncertainty:.2f}\n"
            f"- Conflicts: {len(state.conflicts)}\n\n"
            f"More information is needed to provide a reliable answer."
        )
    
    def _build_low_confidence_response(
        self,
        query: str,
        state: RAGState,
    ) -> str:
        """Build response for low confidence information."""
        return (
            f"The available information about '{query}' has low confidence.\n\n"
            f"Found {len(state.facts)} facts, but none meet the confidence threshold.\n"
            f"Uncertainty: {state.uncertainty:.2f}\n\n"
            f"I recommend gathering more reliable sources before drawing conclusions."
        )
