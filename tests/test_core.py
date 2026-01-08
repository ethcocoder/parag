"""Basic tests for core functionality."""

import pytest
import numpy as np
from parag.core import KnowledgeUnit, RetrievalResult, RAGState


class TestKnowledgeUnit:
    """Test KnowledgeUnit class."""
    
    def test_create_knowledge_unit(self):
        """Test creating a knowledge unit."""
        unit = KnowledgeUnit(
            content="Test content",
            metadata={"source": "test.txt"}
        )
        
        assert unit.content == "Test content"
        assert unit.get_source() == "test.txt"
        assert unit.unit_id is not None
        assert unit.confidence is None
    
    def test_knowledge_unit_with_embedding(self):
        """Test knowledge unit with embedding."""
        embedding = np.random.rand(384).astype(np.float32)
        
        unit = KnowledgeUnit(
            content="Test content",
            embedding=embedding,
            confidence=0.8
        )
        
        assert unit.has_embedding()
        assert unit.confidence == 0.8
        assert unit.embedding.shape == (384,)
    
    def test_add_tags(self):
        """Test adding tags to knowledge unit."""
        unit = KnowledgeUnit(content="Test")
        
        unit.add_tag("important")
        unit.add_tag("verified")
        
        assert "important" in unit.get_tags()
        assert "verified" in unit.get_tags()


class TestRetrievalResult:
    """Test RetrievalResult class."""
    
    def test_create_empty_result(self):
        """Test creating empty result."""
        result = RetrievalResult(query="test query")
        
        assert result.is_empty()
        assert len(result) == 0
        assert result.query == "test query"
    
    def test_create_result_with_units(self):
        """Test creating result with units."""
        units = [
            KnowledgeUnit(content=f"Content {i}")
            for i in range(3)
        ]
        scores = [0.9, 0.7, 0.5]
        
        result = RetrievalResult(
            units=units,
            scores=scores,
            query="test query"
        )
        
        assert len(result) == 3
        assert result.get_average_score() == pytest.approx(0.7)
        assert result.get_max_score() == 0.9
    
    def test_filter_by_threshold(self):
        """Test filtering by threshold."""
        units = [
            KnowledgeUnit(content=f"Content {i}")
            for i in range(3)
        ]
        scores = [0.9, 0.7, 0.3]
        
        result = RetrievalResult(units=units, scores=scores)
        
        filtered = result.filter_by_threshold(0.6)
        
        assert len(filtered) == 2
        assert all(score >= 0.6 for score in filtered.scores)


class TestRAGState:
    """Test RAGState class."""
    
    def test_create_empty_state(self):
        """Test creating empty state."""
        state = RAGState()
        
        assert len(state.facts) == 0
        assert len(state.knowledge_units) == 0
        assert state.uncertainty >= 0.0
    
    def test_build_from_retrieval_result(self):
        """Test building state from retrieval result."""
        units = [
            KnowledgeUnit(content=f"Fact {i}", confidence=0.8)
            for i in range(3)
        ]
        scores = [0.9, 0.8, 0.7]
        
        result = RetrievalResult(
            units=units,
            scores=scores,
            query="test"
        )
        
        state = RAGState.from_retrieval_result(result)
        
        assert len(state.facts) == 3
        assert len(state.knowledge_units) == 3
        assert state.uncertainty < 1.0
    
    def test_add_knowledge_unit(self):
        """Test adding knowledge unit to state."""
        state = RAGState()
        
        unit = KnowledgeUnit(content="Test fact", confidence=0.9)
        state.add_knowledge_unit(unit)
        
        assert len(state.facts) == 1
        assert len(state.knowledge_units) == 1
    
    def test_has_sufficient_information(self):
        """Test checking for sufficient information."""
        state = RAGState()
        
        # Insufficient initially
        assert not state.has_sufficient_information()
        
        # Add units
        for i in range(3):
            unit = KnowledgeUnit(content=f"Fact {i}", confidence=0.9)
            state.add_knowledge_unit(unit)
        
        # Should be sufficient now
        assert state.has_sufficient_information()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
