# `retrieval/retriever.py` - Core Retrieval Engine

## Overview
The `Retriever` class is the main coordinator for the Parag RAG pipeline. it bridges the gap between text queries, embedding models, and vector stores. It also integrates directly with the broader Paradox ecosystem for "cognitive" searching.

## Key Features
- **Unified Search API**: Simple `retrieve(query)` call that handles embedding generation and vector lookup.
- **Batch Support**: `batch_retrieve` for processing multi-query workloads efficiently.
- **Paradox Integration**: Specifically designed to work with `ParadoxVectorStore` and `AlienIntuition`.
- **Cognitive Loop Implementation**: `cognitive_retrieve` runs a "reflexive" Search-Think-Search loop.

## Core Methods

### `retrieve(query, top_k, score_threshold)`
The primary method. 
1. Generates an embedding for the query.
2. Queries the vector store.
3. Wraps the raw results into a `RetrievalResult`.

### `add_knowledge_units(units, generate_embeddings)`
Interface for indexing new data. If `generate_embeddings` is True, it automatically uses the internal model to create vectors for all units before saving them to the store.

### `cognitive_retrieve(query, max_turns)`
A more "autonomous" version of standard retrieval:
- It performs an initial search.
- It analyzes the `RAGState` for uncertainty or conflicts.
- If the result is unsatisfactory, it uses `AlienIntuition` or internal reasoning to refine the search.

## Paradox Integration
- **Alien Intuition**: If `modules.reasoning.alien_intuition` is available, the retriever uses it to guide reflexive searches.
- **Vector Backends**: Seamlessly switches between local FAISS and the custom `ParadoxVectorStore`.

## Usage Example
```python
from parag.retrieval import Retriever
from parag.embeddings import ParadoxEmbeddings
from parag.vectorstore import FAISSVectorStore

retriever = Retriever(
    embedding_model=ParadoxEmbeddings(),
    vector_store=FAISSVectorStore()
)

# Standard search
res = retriever.retrieve("How does Paradma learn?")

# Advanced cognitive search (multiple turns)
state = retriever.cognitive_retrieve("Complex contradictory topic")
```

## Internal Workflow
1. **Query Encoding**: Converts text to `np.ndarray`.
2. **Similarity Search**: Executes the distance metric query in the vector store.
3. **Filtering**: Applies the `score_threshold`.
4. **Wrapping**: Encapsulates data in a structured `RetrievalResult`.
