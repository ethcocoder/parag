# `__init__.py` - Parag Main Module

## Overview
Main entry point for the Parag (Next-Generation RAG System) package. This module exports the core public API for the RAG system.

## Purpose
- **Package Initialization**: Defines the public API surface for the Parag library
- **Version Management**: Contains version and author information
- **Core Exports**: Exposes the fundamental data structures used throughout the system

## Key Features
- Structured knowledge representation via `KnowledgeUnit`
- State-based reasoning through `RAGState`
- Retrieval result encapsulation with `RetrievalResult`
- Future compatibility with Paradox cognitive engines

## Public API

### Exported Classes
- **`KnowledgeUnit`**: Fundamental data structure for retrieved information
- **`RetrievalResult`**: Container for retrieval outputs with scores and metadata
- **`RAGState`**: State management for RAG reasoning

### Metadata
- **Version**: 0.1.0
- **Author**: ethcocoder

## Module Structure
```python
parag/
├── __init__.py          # This file - main package entry
├── core/                # Core data models
├── embeddings/          # Embedding generation
├── generation/          # LLM integration
├── ingestion/           # Data loading and chunking
├── reasoning/           # Advanced reasoning
├── retrieval/           # Retrieval engines
├── utils/               # Utilities
└── vectorstore/         # Vector storage backends
```

## Usage Example
```python
from parag import KnowledgeUnit, RetrievalResult, RAGState

# Create a knowledge unit
unit = KnowledgeUnit(
    content="The Paradox framework uses self-learning mathematics",
    embedding=np.array([0.1, 0.2, 0.3]),
    metadata={"source": "paradma_docs"}
)

# Check version
import parag
print(f"Parag version: {parag.__version__}")
```

## Design Philosophy
Parag is designed to **evolve beyond classical retrieval**:
- **Structured Knowledge**: Not just text chunks, but rich semantic units
- **State-Based Reasoning**: Maintains context across retrieval sessions
- **Conflict Detection**: Identifies contradictions in retrieved knowledge
- **Uncertainty Measurement**: Quantifies confidence in retrieval results
- **Paradox Integration**: Ready for fusion with Paradox cognitive engines

## Future Roadmap
- Integration with ParadoxLF for cognitive reasoning
- Paradma backend for self-learning embeddings
- Quantum-inspired retrieval algorithms
- Multi-modal knowledge units (text, images, tensors)
