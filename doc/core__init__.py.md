# `core/__init__.py` - Core Data Models

## Overview
Package initialization for Parag's core data models. Exports the fundamental data structures used throughout the RAG system.

## Purpose
- **Central Exports**: Single import point for core classes
- **Clean API**: Provides `__all__` for controlled exports
- **Module Organization**: Defines the core model layer

## Exported Classes

### Data Structures
- **`KnowledgeUnit`**: Fundamental unit of knowledge (content + embedding + metadata)
- **`RetrievalResult`**: Container for retrieval outputs with scores
- **`RAGState`**: State management for RAG reasoning sessions
- **`Fact`**: Individual fact representation used in RAGState

### Cognitive Components
- **`CognitiveLoop`**: Advanced reasoning loop for iterative retrieval

## Module Structure
```
parag/core/
├── __init__.py          # This file - exports core API
├── knowledge_unit.py    # KnowledgeUnit class
├── retrieval_result.py  # RetrievalResult class
├── rag_state.py         # RAGState and Fact classes
└── cognitive_loop.py    # CognitiveLoop class
```

## Usage Example
```python
# Import all core classes at once
from parag.core import (
    KnowledgeUnit,
    RetrievalResult,
    RAGState,
    Fact,
    CognitiveLoop
)

# Or import from parag directly
from parag import KnowledgeUnit, RetrievalResult, RAGState
```

## Design Philosophy
The core module provides the **foundational abstractions** for the RAG system:
- **`KnowledgeUnit`**: Represents "what we know"
- **`RetrievalResult`**: Represents "what we found"
- **`RAGState`**: Represents "what we're thinking about"
- **`Fact`**: Represents "a single piece of truth"
- **`CognitiveLoop`**: Represents "how we reason"

These abstractions enable building complex RAG systems with clean, composable components.
