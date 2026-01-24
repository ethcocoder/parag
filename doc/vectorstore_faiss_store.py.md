# `vectorstore/faiss_store.py` - FAISS Backend

## Overview
`FAISSVectorStore` provides a high-performance similarity search engine by wrapping Facebook's FAISS (Facebook AI Similarity Search) library. It is the recommended backend for large-scale, production RAG systems where latency and memory efficiency are paramount.

## Features
- **Multiple Index Types**: Supports `FlatL2`, `IVFFlat`, and `HNSW`.
- **Metric Support**: Supports L2 distance and Inner Product (IP) metrics.
- **Metadata Management**: Synchronizes an internal metadata store with the FAISS vector IDs.
- **Serialization**: Built-in `save` and `load` methods for disk persistence.

## Supported Index Types

### `Flat`
- **Logic**: Brute-force L2/IP search.
- **Best For**: Small to medium datasets (< 1M vectors) where accuracy is the only concern.

### `IVFFlat`
- **Logic**: Inverted File with Flat storage. Partitions the space into clusters.
- **Best For**: Fast search on large datasets via quantization.

### `HNSW`
- **Horizontal Small World**: Efficient graph-based search.
- **Best For**: Very high performance on large datasets with a small memory trade-off.

## Core Methods

### `add(vectors, metadata)`
Appends new embeddings to the index. Returns a list of generated integer IDs. 

### `search(query_vector, k)`
Performs a nearest-neighbor search. Returns:
- **Distances**: List of floats representing similarity.
- **Indices**: List of IDs corresponding to the hits.
- **Metadata**: List of dictionaries attached to the vectors.

### `save(path)` / `load(path)`
Saves/loads both the Faiss binary index and the JSON-encoded metadata to the specified directory.

## Usage Example
```python
from parag.vectorstore import FAISSVectorStore
import numpy as np

store = FAISSVectorStore(dimension=384, index_type="HNSW", metric="IP")

# Add some data
embeddings = np.random.rand(10, 384).astype('float32')
meta = [{"doc_id": i} for i in range(10)]
store.add(embeddings, meta)

# Search
dist, idx, metadata = store.search(embeddings[0], k=3)
print(f"Nearest hit: {metadata[0]['doc_id']} with dist {dist[0]}")
```

## Requirements
- **FAISS**: Requires `faiss-cpu` or `faiss-gpu` to be installed in the environment.
