# `embeddings/__init__.py` - Embeddings Module Entry

## Overview
Package initialization for the Parag embeddings module. This module provides a unified interface for different embedding model implementations, allowing for both legacy support (SentenceTransformers) and native Paradox integration.

## Purpose
- **Model Registry**: Exports available embedding model implementations
- **Interface Exposure**: Provides the base `EmbeddingModel` abstract class
- **Implementation Selection**: Allows users to choose between standard and Paradox-enhanced embeddings

## Exported Components

### Base Interface
- **`EmbeddingModel`**: The abstract base class that defines the interface for all embedding implementations in Parag.

### Implementations
- **`SentenceTransformerEmbeddings`**: Legacy support using the `sentence-transformers` library (all-MiniLM-L6-v2 by default).
- **`ParadoxEmbeddings`**: Native Paradox implementation using `modules.framework` and Paradma for self-learning operations.

## Usage Example
```python
from parag.embeddings import ParadoxEmbeddings

# Initialize the native Paradox embedding model
model = ParadoxEmbeddings(embedding_dim=384)

# Generate an embedding
vector = model.embed("This is a test sentence")
```

## Design Philosophy
The embeddings module is designed to be **interchangeable**:
1. **Consistency**: All models share the same API via `EmbeddingModel`.
2. **Evolution**: Users can start with `SentenceTransformerEmbeddings` and transition to `ParadoxEmbeddings` as they integrate more of the Paradox ecosystem.
3. **Integration**: `ParadoxEmbeddings` directly uses the Paradox framework's `Tensor` and Paradma's learning capabilities.
