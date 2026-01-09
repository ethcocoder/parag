"""
LLM adapter interface.

Provides abstraction for integrating with various LLM backends.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from parag.core.rag_state import RAGState


class LLMAdapter(ABC):
    """
    Abstract interface for LLM integration.
    
    Allows swapping between different LLM backends.
    """
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        pass


class DeterministicGenerator:
    """
    Generate responses without an LLM.
    
    Uses structured state to produce deterministic,
    explainable responses.
    """
    
    def __init__(self):
        """Initialize DeterministicGenerator."""
        pass
    
    def generate_from_state(
        self,
        query: str,
        state: RAGState,
        include_reasoning: bool = True,
    ) -> str:
        """
        Generate response from RAGState without LLM.
        
        Args:
            query: User query
            state: RAGState containing knowledge
            include_reasoning: Whether to include reasoning explanation
            
        Returns:
            Generated response
        """
        # Check if we have sufficient information
        if not state.has_sufficient_information():
            return self._generate_insufficient_response(query, state)
        
        # Get high confidence facts
        high_conf_facts = state.get_high_confidence_facts(threshold=0.7)
        
        if not high_conf_facts:
            return self._generate_low_confidence_response(query, state)
        
        # Build response
        response_parts = []
        
        # Main content
        response_parts.append(f"Regarding '{query}':\n")
        
        for i, fact in enumerate(high_conf_facts, 1):
            response_parts.append(f"{i}. {fact.content}")
        
        # Add reasoning section if requested
        if include_reasoning:
            reasoning_parts = []
            
            # Confidence information
            avg_conf = sum(f.confidence for f in high_conf_facts) / len(high_conf_facts)
            reasoning_parts.append(f"Average confidence: {avg_conf:.2f}")
            reasoning_parts.append(f"Based on {len(state.knowledge_units)} knowledge unit(s)")
            
            # Conflicts
            if state.conflicts:
                reasoning_parts.append(
                    f"⚠️ Note: {len(state.conflicts)} conflict(s) detected in the information"
                )
            
            # Uncertainty
            if state.uncertainty > 0.3:
                reasoning_parts.append(
                    f"Uncertainty level: {state.uncertainty:.2f}"
                )
                
            # Paradox Cognitive Signals
            if "emotions" in state.metadata:
                emotions = state.metadata["emotions"]
                # Report strongest feelings
                feelings = [f"{k}: {v:.2f}" for k, v in emotions.items() if v > 0.6 or v < 0.4]
                if feelings:
                    reasoning_parts.append(f"AI Emotions: {', '.join(feelings)}")
                    
            if "manifold_curvature" in state.metadata:
                curvature = state.metadata["manifold_curvature"]
                reasoning_parts.append(f"Manifold Curvature: {curvature:.4f}")
                if "intuition_signal" in state.metadata:
                    reasoning_parts.append(f"Intuition: {state.metadata['intuition_signal']}")
            
            response_parts.append("\n---")
            response_parts.append("Reasoning:")
            response_parts.extend(f"- {part}" for part in reasoning_parts)
        
        return "\n".join(response_parts)
    
    def _generate_insufficient_response(self, query: str, state: RAGState) -> str:
        """Generate response for insufficient data."""
        from parag.reasoning.uncertainty import UncertaintyCalculator
        
        calc = UncertaintyCalculator()
        issues = calc.detect_missing_information(state)
        
        response = [
            f"Unable to provide a confident answer to '{query}'.",
            "",
            "Reasons:",
        ]
        
        for issue in issues:
            response.append(f"- {issue}")
        
        response.append("")
        response.append("Recommendation: Gather more information before proceeding.")
        
        return "\n".join(response)
    
    def _generate_low_confidence_response(self, query: str, state: RAGState) -> str:
        """Generate response for low confidence."""
        response = [
            f"Information about '{query}' found, but confidence is low.",
            "",
            f"Available facts: {len(state.facts)}",
            f"Overall uncertainty: {state.uncertainty:.2f}",
        ]
        
        if "emotions" in state.metadata:
            emotions = state.metadata["emotions"]
            reflexion = emotions.get("Reflexion", 0.0)
            if reflexion > 0.6:
                response.append(f"Reflexive State: High (Active conflict resolution required)")
        
        response.extend([
            "",
            "The available information does not meet reliability thresholds.",
            "Recommend seeking additional sources.",
        ])
        
        return "\n".join(response)
    
    def should_use_llm(self, state: RAGState) -> bool:
        """
        Decide whether to use LLM or deterministic generation.
        
        Args:
            state: RAGState to evaluate
            
        Returns:
            True if LLM should be used, False for deterministic generation
        """
        # Use deterministic if:
        # - High uncertainty
        # - Conflicts present
        # - Few knowledge units
        
        if state.uncertainty > 0.6:
            return False  # Too uncertain, use deterministic
        
        if len(state.conflicts) > 2:
            return False  # Too many conflicts
        
        if len(state.knowledge_units) < 2:
            return False  # Too little information
        
        return True  # Safe to use LLM
