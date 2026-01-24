# `embeddings/base.py` - Embedding Interface

## Overview
Defines the `EmbeddingModel` abstract base class, which establishes the standard interface for all text embedding implementations in the Parag system.

## Purpose
- **Standardization**: Ensures all embedding models have the same method signatures.
- **Utility Features**: Provides common functionality like vector normalization that can be reused by all implementations.
- **Abstraction**: Allows the rest of the RAG system to work with any embedding model without knowing its internal implementation details.

## Key Components

### `EmbeddingModel` (Abstract Class)
The core interface that every embedding implementation must inherit from.

#### Abstract Methods (Must be implemented)
- **`embed(text)`**: Generate embeddings for a single string or a list of strings.
- **`embed_batch(texts, batch_size)`**: Process a large list of texts in batches.
- **`get_embedding_dim()`**: Return the size of the vectors produced by the model.

#### Concrete Methods (Provided)
- **`normalize(embeddings)`**: Scales vectors to unit length (L2 normalization). Handles both single vectors (1D) and batches (2D).
- **`embed_and_normalize(text)`**: A convenience wrapper that generates embeddings and immediately normalizes them.

## Method Details

### `normalize(embeddings: np.ndarray) -> np.ndarray`
Standardizes the length of vectors to 1. This is crucial for cosine similarity, which is the default metric for most vector stores.
- **Input**: A NumPy array of shape `(dim,)` or `(N, dim)`.
- **Handling**: Safely handles zero-length vectors by avoiding division by zero.

### `embed_and_normalize(text: Union[str, List[str]]) -> np.ndarray`
Combines embedding generation and normalization in one call. This is the most commonly used method when preparing data for a vector store.

## Implementation Requirements
When creating a new embedding model for Parag:
1. Subclass `EmbeddingModel`.
2. Implement `get_embedding_dim()`.
3. Implement `embed()` for both single strings and lists.
4. Implement `embed_batch()` for efficient processing of large datasets.
