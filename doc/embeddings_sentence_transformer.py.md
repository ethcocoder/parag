# `embeddings/sentence_transformer.py` - SentenceTransformer Integration

## Overview
Provides a wrapper for the popular `sentence-transformers` library, allowing Parag to use thousands of high-quality pre-trained models from Hugging Face (such as `all-MiniLM-L6-v2` or `multi-qa-mpnet-base-dot-v1`).

## Purpose
- **Quick Start**: Allows users to get high-quality retrieval working immediately.
- **Interoperability**: Enables using models that are standard in the industry.
- **Hardware Support**: Automatically leverages CUDA if available for fast inference.

## Key Components

### `SentenceTransformerEmbeddings` Class
Implementation of the `EmbeddingModel` interface using `SentenceTransformer`.

#### Initialization
- **`model_name`**: The Hugging Face model identifier (default: `"all-MiniLM-L6-v2"`).
- **`device`**: Target device ('cuda', 'cpu', or None for auto-detection).
- **`normalize`**: Whether to return vectors already scaled to unit length.

#### Methods
- **`embed(text)`**: Generates embeddings for one or more strings.
- **`embed_batch(texts, batch_size, show_progress)`**: Generates embeddings for a large list with an optional progress bar.
- **`get_embedding_dim()`**: Returns the dimension of the selected model.

## Dependencies
This module requires the `sentence-transformers` package:
```bash
pip install sentence-transformers
```
If not installed, attempting to initialize this class will raise a descriptive `ImportError`.

## Usage Example
```python
from parag.embeddings import SentenceTransformerEmbeddings

# Use a high-quality pre-trained model
model = SentenceTransformerEmbeddings("all-MiniLM-L6-v2", device="cuda")

# Embed a series of documents
docs = ["Document 1 content", "Document 2 content"]
embeddings = model.embed_batch(docs, batch_size=16)

print(f"Embedding dimension: {model.get_embedding_dim()}")
```

## Implementation Details
- **Batch Processing**: Uses the optimized batch encoding provided by the underlying library.
- **Progress Tracking**: Includes `tqdm` support for long-running ingestion tasks.
- **Auto-Detection**: Dynamically queries the underlying model for its specific embedding dimension.
