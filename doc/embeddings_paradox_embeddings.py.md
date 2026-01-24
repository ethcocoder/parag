# `embeddings/paradox_embeddings.py` - Native Paradox Embeddings

## Overview
`ParadoxEmbeddings` is a custom, lightweight embedding model built entirely on the Paradox framework. It replaces external dependencies like `sentence-transformers` with native `modules.framework.Tensor` operations and Paradma-optimized math.

## Features
- **Native Tensor Ops**: Uses `modules.framework.Tensor` for all mathematical calculations.
- **Paradma Accelerated**: Leverages Paradma's self-learning manifold for mean pooling and normalization.
- **Lightweight**: Avoids heavy external ML libraries.
- **Self-Contained**: Includes its own `SimpleEmbeddingLayer` and serialization logic.

## Key Components

### `SimpleEmbeddingLayer`
A basic neural layer that maps token indices to dense vectors.
- **Initialization**: Creates a weights tensor with normal distribution.
- **Forward Pass**: Performs vectorized lookup of embeddings for a batch of tokens.

### `ParadoxEmbeddings` Class
The main embedding model implementation.

#### Attributes
- **`embedding_dim`**: The size of the output vectors (default: 384).
- **`vocab_size`**: The size of the expected vocabulary (default: 30,000).
- **`normalize`**: Boolean flag to enable/disable vector normalization.
- **`use_paradma`**: Boolean flag to use Paradma-optimized operations when available.

#### Core Methods
- **`embed(text)`**: Tokenizes input and passes it through the embedding layer.
- **`encode(text)`**: An alias for `embed()` to maintain compatibility with other Paradox engines.
- **`embed_batch(texts, batch_size)`**: Efficiently processes multiple sentences.
- **`save(path)` / `load(path)`**: Persists the model weights to disk using standard Paradox serialization.

## Paradox Integration

### Paradma Pooling
When `use_paradma` is enabled, the model uses Paradma's learning manifold to perform mean pooling across token embeddings. This allows Paradma to observe and optimize common pooling patterns.

### Paradma Normalization
Similarly, vector normalization can be executed via Paradma, enabling "self-learning" hardware acceleration or algorithmic optimization for scaling retrieved vectors.

## Usage Example
```python
from parag.embeddings import ParadoxEmbeddings

# Initialize model
model = ParadoxEmbeddings(embedding_dim=512)

# Generate normalized embedding
vector = model.embed("Integrating RAG with Paradox cognitive loops.")

# Save model for later use
model.save("models/my_paradox_embeddings.bin")
```

## Internal Workflow
1. **Tokenization**: (Currently uses a simple space-based split or character-level logic depending on the integrated tokenizer).
2. **Lookup**: Converts tokens to indices and retrieves vectors from `SimpleEmbeddingLayer`.
3. **Pooling**: Averages token vectors to create a single "sentence" vector.
4. **Normalization**: Scales the resulting vector to unit length for similarity matching.
