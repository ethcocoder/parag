"""
Parag - Next-Generation RAG System

A modular, extensible RAG (Retrieval-Augmented Generation) system
designed to evolve beyond classical retrieval.

Key Features:
- Structured knowledge representation
- State-based reasoning
- Conflict detection and uncertainty measurement
- Future compatibility with Paradox cognitive engines
"""

__version__ = "0.1.0"
__author__ = "ethcocoder"

from .core.knowledge_unit import KnowledgeUnit
from .core.retrieval_result import RetrievalResult
from .core.rag_state import RAGState
from .knowledge_graph import KnowledgeGraph

__all__ = [
    "KnowledgeUnit",
    "RetrievalResult",
    "RAGState",
    "KnowledgeGraph",
    "__version__",
]
