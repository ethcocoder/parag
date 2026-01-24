# `retrieval/__init__.py` - Retrieval Module Entry

## Overview
Package initialization for the Parag retrieval module. This module is the engine room of the RAG system, responsible for finding and ranking relevant information based on semantic similarity.

## Purpose
- **Core Engine**: Exports the main `Retriever` and `ResultRanker` classes.
- **Search Logic**: Centralizes the logic for searching vector stores and refining the resulting hits.

## Exported Components
- **`Retriever`**: The primary search engine. It translates text queries into vector space and queries the underlying vector store.
- **`ResultRanker`**: Provides advanced sorting and filtering logic (diversity, recency) after the initial similarity search.

## Usage Example
```python
from parag.retrieval import Retriever, ResultRanker

# Initialize with an embedding model and vector store
retriever = Retriever(embedding_model=my_model, vector_store=my_store)

# perform a search
result = retriever.retrieve("What is the Paradox core?")

# Optionally re-rank for better diversity
ranker = ResultRanker(diversity_weight=0.5)
refined_result = ranker.rank(result)
```

## Module Structure
- `retriever.py`: Main search orchestration.
- `ranker.py`: Result refinement and filtering logic.
