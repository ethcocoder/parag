# `vectorstore/paradox_store.py` - Autonomous Memory

## Overview
`ParadoxVectorStore` is the next-generation memory engine for Parag. Instead of using a static index like FAISS, it leverages ParadoxLF's `LatentMemoryEngine` and the Paradma learning manifold. It treats retrieval as an autonomous, self-optimizing process.

## Key Features
- **Paradma Backend**: Vectors are stored and searched using Paradma-optimized operations.
- **Autonomous Optimization**: The memory engine can "re-organize" itself based on retrieval patterns.
- **Conceptual Search**: High-level semantic search that goes beyond simple keyword or distance matching.
- **Trajectory Prediction**: Can predict future queries based on historical retrieval trajectories.
- **Imaginative Blending**: Can "imagine" or blend two vectors to find information that exists in the latent space between them.

## Backend Modes

### `paradma`
- Uses the symbolic learning manifold for all operations.
- Slower than FAISS but allows the store to "learn" how to search better over time.
- Supports gradient synchronization.

### `turbo`
- Optimized for speed while still using the Paradox engine logic.

## Unique Cognitive Methods

### `conceptual_search(concept, k)`
Uses the internal encoder of the Paradox engine to perform a deeper semantic lookup that understands context better than raw embeddings.

### `imagine(vector_a, vector_b, ratio)`
A creative function that interpolates between two knowledge states. Useful for finding information that bridges two different topics.

### `predict_future(history_vectors, steps)`
Analyzes the movement of a user's interest in latent space to predict what information they might need next.

## Core Methods

### `add(vectors, metadata)`
Inserts data into the `LatentMemoryEngine`. Unlike FAISS, this step can trigger "learning" cycles where the store analyzes the new patterns.

### `search(query_vector, k)`
Performs a similarity search using Paradox-native distance metrics.

## Usage Example
```python
from parag.vectorstore import ParadoxVectorStore

store = ParadoxVectorStore(
    dimension=512, 
    backend="paradma",
    storage_dir="./brain"
)

# Imagine a point between "quantum computing" and "thermodynamics"
inter_vect = store.imagine(quantum_vec, thermo_vec, ratio=0.5)
hits = store.search(inter_vect, k=5)

# Predict next query
future = store.predict_future([prev_query_1, prev_query_2])
```

## Integration Details
- **LatentMemoryEngine**: The core of the ParadoxLF ecosystem.
- **Self-Optimization**: Periodically runs maintenance cycles to prune redundant or low-confidence "memories."
- **Persistent States**: Saves its entire memory "manifold" rather than just a flat index.
