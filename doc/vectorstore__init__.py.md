# `vectorstore/__init__.py` - Vector Store Module Entry

## Overview
Package initialization for the Parag vector store module. This module provides storage solutions for high-dimensional embeddings, supporting both the industry-standard FAISS library and the native, self-learning Paradox memory engines.

## Purpose
- **Persistence Layer**: Exports classes for saving and searching vectors.
- **Backend Choice**: Allows users to choose between low-latency standard search (FAISS) and autonomous cognitive memory (ParadoxStore).

## Exported Components

### Stores
- **`FAISSVectorStore`**: A wrapper for Facebook's FAISS library. Highly optimized for traditional RAG.
- **`ParadoxVectorStore`**: The native Paradox implementation using `LatentMemoryEngine`. Optimized for self-learning and autonomous operations.

### Management
- **`IndexManager`**: A lifecycle tool for managing index versions, backups, and persistence paths.

## Usage Example
```python
from parag.vectorstore import FAISSVectorStore, ParadoxVectorStore

# Standard setup
store = FAISSVectorStore(dimension=384)
store.add(vectors, metadata)

# Native Paradox setup (Autonomous memory)
px_store = ParadoxVectorStore(dimension=384, backend="paradma")
px_store.add(vectors, metadata)
```

## Module Structure
- `faiss_store.py`: FAISS-based implementation.
- `paradox_store.py`: Paradma-powered LatentMemoryEngine implementation.
- `index_manager.py`: Versioning and file system management.
