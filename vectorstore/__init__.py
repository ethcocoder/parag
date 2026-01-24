"""Vector store module."""

from parag.vectorstore.faiss_store import FAISSVectorStore
from parag.vectorstore.paradox_store import ParadoxVectorStore
from parag.vectorstore.index_manager import IndexManager

__all__ = [
    "FAISSVectorStore",  # Legacy FAISS support
    "ParadoxVectorStore",  # New Paradox integration
    "IndexManager",
]
