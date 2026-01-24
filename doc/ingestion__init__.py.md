# `ingestion/__init__.py` - Ingestion Module Entry

## Overview
Package initialization for the Parag ingestion module. This module provides a comprehensive pipeline for loading, chunking, and extracting metadata from various document formats.

## Purpose
- **Pipeline Components**: Exports loaders, chunkers, and metadata extractors.
- **Simplified Imports**: Provides a flat namespace for common ingestion tasks.

## Exported Components

### Loaders
- **`DocumentLoader`**: Abstract base class for all file loaders.
- **`TextLoader`**: For plain text files (.txt).
- **`PDFLoader`**: For PDF documents (.pdf).
- **`MarkdownLoader`**: For Markdown files (.md) with frontmatter support.

### Processing
- **`TextChunker`**: Splits documents into manageable pieces for embedding.
- **`ChunkingStrategy`**: Enum defining available splitting methods (Fixed size, Sentence, Paragraph).
- **`MetadataExtractor`**: Handles automatic extraction of file and content properties.

## Usage Example
```python
from parag.ingestion import TextLoader, TextChunker, MetadataExtractor

# Load a document
loader = TextLoader()
docs = loader.load("data/research_paper.txt")

# Chunk it for embedding
chunker = TextChunker(chunk_size=512, chunk_overlap=50)
chunks = chunker.chunk(docs[0]["content"])

# Extract metadata
extractor = MetadataExtractor()
metadata = extractor.extract(file_path="data/research_paper.txt", content=docs[0]["content"])
```

## Module Structure
- `loaders.py`: File format specific loading logic.
- `chunker.py`: Text splitting and segmenting algorithms.
- `metadata.py`: Attribute extraction and management.
