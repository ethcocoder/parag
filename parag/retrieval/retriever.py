"""
Core retrieval engine.

Performs similarity search and returns structured RetrievalResults.
"""

from typing import List, Optional
import numpy as np

from parag.core.knowledge_unit import KnowledgeUnit
from parag.core.retrieval_result import RetrievalResult
from parag.embeddings.base import EmbeddingModel
from parag.vectorstore.faiss_store import FAISSVectorStore

# Import ParadoxVectorStore
try:
    from parag.vectorstore.paradox_store import ParadoxVectorStore
    PARADOX_AVAILABLE = True
except ImportError:
    PARADOX_AVAILABLE = False
    ParadoxVectorStore = None


class Retriever:
    """
    Main retrieval engine for the RAG system.
    
    Combines embedding generation with vector search to retrieve
    relevant knowledge units.
    """
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store,  # FAISSVectorStore or ParadoxVectorStore
        score_threshold: Optional[float] = None,
    ):
        """
        Initialize Retriever.
        
        Args:
            embedding_model: Model for generating query embeddings
            vector_store: Vector store (FAISS or ParadoxVectorStore)
            score_threshold: Minimum similarity score to include results
        """
        # Detect store type
        self.is_paradox_store = (
            PARADOX_AVAILABLE and 
            ParadoxVectorStore is not None and 
            isinstance(vector_store, ParadoxVectorStore)
        )
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.score_threshold = score_threshold
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant knowledge units for a query.
        
        Args:
            query: Query string
            top_k: Number of results to return
            score_threshold: Optional minimum score threshold (overrides default)
            
        Returns:
            RetrievalResult containing relevant knowledge units
        """
        # Generate query embedding
        query_embedding = self.embedding_model.embed(query)
        
        # Search vector store
        distances, indices, metadata_list = self.vector_store.search(
            query_embedding,
            k=top_k,
            return_metadata=True,
        )
        
        # Convert to similarity scores (inverse of L2 distance)
        # For normalized vectors, we can use: similarity = 1 / (1 + distance)
        scores = 1.0 / (1.0 + distances)
        
        # Build knowledge units from results
        units = []
        filtered_scores = []
        
        threshold = score_threshold if score_threshold is not None else self.score_threshold
        
        for i, (score, idx, metadata) in enumerate(zip(scores, indices, metadata_list)):
            # Filter by threshold if specified
            if threshold is not None and score < threshold:
                continue
            
            # Skip invalid indices
            if idx == -1:
                continue
            
            # Reconstruct knowledge unit
            # Note: In a real implementation, we'd store the actual content
            # For now, we use metadata to recreate the unit
            content = metadata.get("content", "<content not stored>")
            
            unit = KnowledgeUnit(
                content=content,
                embedding=None,  # Not storing embeddings in results to save memory
                metadata=metadata,
                confidence=float(score),
                unit_id=metadata.get("unit_id"),
            )
            
            units.append(unit)
            filtered_scores.append(float(score))
        
        # Create retrieval result
        result = RetrievalResult(
            units=units,
            scores=filtered_scores,
            query=query,
            query_embedding=query_embedding,
            metadata={
                "top_k": top_k,
                "threshold": threshold,
                "total_found": len(units),
            }
        )
        
        return result
    
    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve for multiple queries in batch.
        
        Args:
            queries: List of query strings
            top_k: Number of results per query
            score_threshold: Optional minimum score threshold
            
        Returns:
            List of RetrievalResults
        """
        results = []
        
        for query in queries:
            result = self.retrieve(query, top_k, score_threshold)
            results.append(result)
        
        return results
    
    def add_knowledge_units(
        self,
        units: List[KnowledgeUnit],
        generate_embeddings: bool = True,
    ) -> List[int]:
        """
        Add knowledge units to the retrieval system.
        
        Args:
            units: List of KnowledgeUnits to add
            generate_embeddings: Whether to generate embeddings for units
            
        Returns:
            List of assigned IDs in the vector store
        """
        embeddings = []
        metadata_list = []
        
        for unit in units:
            # Generate embedding if needed
            if generate_embeddings or not unit.has_embedding():
                content = str(unit.content)
                embedding = self.embedding_model.embed(content)
                # Store back in unit (optionally as Axiom if enabled)
                if hasattr(unit, 'set_embedding_from_numpy'):
                    unit.set_embedding_from_numpy(embedding)
                else:
                    unit.embedding = embedding
            
            # Use NumPy version for vector store compatibility
            if hasattr(unit, 'get_embedding_as_numpy'):
                embeddings.append(unit.get_embedding_as_numpy())
            else:
                embeddings.append(unit.embedding)
            
            # Prepare metadata (including content for reconstruction)
            metadata = {**unit.metadata}
            metadata["content"] = str(unit.content)
            metadata["unit_id"] = unit.unit_id
            
            metadata_list.append(metadata)
        
        # Add to vector store
        embeddings_array = np.array(embeddings, dtype=np.float32)
        ids = self.vector_store.add(embeddings_array, metadata_list)
        
        return ids
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Retriever("
            f"embedding_model={self.embedding_model.__class__.__name__}, "
            f"vector_store_size={self.vector_store.size()}, "
            f"threshold={self.score_threshold})"
        )
