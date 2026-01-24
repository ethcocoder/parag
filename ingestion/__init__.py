"""Document ingestion module."""

from parag.ingestion.loaders import (
    DocumentLoader,
    TextLoader,
    PDFLoader,
    MarkdownLoader,
)
from parag.ingestion.chunker import TextChunker, ChunkingStrategy
from parag.ingestion.metadata import MetadataExtractor

__all__ = [
    "DocumentLoader",
    "TextLoader",
    "PDFLoader",
    "MarkdownLoader",
    "TextChunker",
    "ChunkingStrategy",
    "MetadataExtractor",
]
