"""Embeddings module."""

from parag.embeddings.base import EmbeddingModel
from parag.embeddings.sentence_transformer import SentenceTransformerEmbeddings

__all__ = [
    "EmbeddingModel",
    "SentenceTransformerEmbeddings",
]
