# `ingestion/chunker.py` - Text Segmentation

## Overview
The `chunker` module handles the transformation of large text bodies into smaller, semantically meaningful "chunks". This is critical for vector-based retrieval, as it ensures that the embeddings represent cohesive units of information.

## Key Components

### `ChunkingStrategy` (Enum)
Defines how the text should be split:
- **`FIXED_SIZE`**: Splits text into a specific number of characters, regardless of content boundaries.
- **`SENTENCE`**: Splits by sentences, ensuring no sentence is cut in half.
- **`PARAGRAPH`**: Splits by double newlines or paragraph markers.

### `TextChunker` Class
The main engine for document segmentation.

#### Configuration
- **`chunk_size`**: Target character count for each chunk (default: 500).
- **`chunk_overlap`**: Number of characters to duplicate between adjacent chunks to maintain context (default: 50).
- **`strategy`**: The `ChunkingStrategy` to apply.

#### Core Methods
- **`chunk(text, metadata)`**: The primary method. Returns a list of `Chunk` objects.
- **`_chunk_by_sentence()`**: Iteratively adds sentences until the `chunk_size` is reached.
- **`_chunk_fixed_size()`**: A simpler, faster splitting method with overlap.

### Helper Functions
- **`chunk_documents(documents, chunker)`**: A utility function to process a list of raw document dictionaries in bulk.

## Usage Example
```python
from parag.ingestion import TextChunker, ChunkingStrategy

chunker = TextChunker(
    chunk_size=800,
    chunk_overlap=100,
    strategy=ChunkingStrategy.SENTENCE
)

text = "This is the first sentence. This is the second. ..."
chunks = chunker.chunk(text, metadata={"theme": "tutorial"})

for chunk in chunks:
    print(f"Chunk {chunk.chunk_id}: {chunk.content[:50]}...")
```

## Implementation Details
- **Sentence Splitting**: Uses a regex-based `_split_into_sentences` method that handles common abbreviations and punctuation while being lightweight.
- **Overlap Handling**: In `FIXED_SIZE` mode, the overlap is precisely calculated to ensure no data loss between chunks.
- **Metadata Propagation**: Each generated `Chunk` inherits the document's metadata and adds its own positional information (`start_char`, `end_char`, `chunk_id`).
