"""Embeddings module."""

from parag.embeddings.base import EmbeddingModel
from parag.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from parag.embeddings.paradox_embeddings import ParadoxEmbeddings

__all__ = [
    "EmbeddingModel",
    "SentenceTransformerEmbeddings",  # Legacy support
    "ParadoxEmbeddings",  # New Paradox integration
]
