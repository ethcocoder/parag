# `knowledge_unit.py` - Fundamental RAG Data Structure

## Overview
Defines `KnowledgeUnit`, the fundamental data structure for all retrieved information in the Parag RAG system. Provides a unified interface for content, embeddings, and metadata with optional Paradma integration.

## Purpose
- **Unified Data Model**: Standard representation for all knowledge in the RAG system
- **Embedding Support**: Handles both NumPy arrays and Paradma Axioms/TensorAxioms
- **Metadata Management**: Rich metadata tracking (source, timestamp, tags, confidence)
- **Type Conversion**: Seamless conversion between NumPy and Paradma types

## Key Components

### `KnowledgeUnit` Class
Main dataclass representing a single unit of knowledge.

#### Attributes
- **`content`**: The actual content (text, bytes, or any object)
- **`embedding`**: Vector representation (NumPy array, Axiom, or TensorAxiom)
- **`metadata`**: Dictionary of additional information
- **`confidence`**: Optional confidence score (0.0 to 1.0)
- **`unit_id`**: Unique identifier (auto-generated if not provided)
- **`_use_paradma`**: Whether to prefer Paradma types (default: True)

### Core Methods

#### Initialization
- **`__post_init__()`**: Auto-generates ID, adds timestamp, validates confidence

#### ID Generation
- **`_generate_id()`**: Creates unique SHA-256 hash from content and timestamp

#### Embedding Checks
- **`has_embedding()`**: Returns True if unit has a valid embedding
- **`is_using_paradma()`**: Returns True if embedding uses Paradma types

#### Metadata Access
- **`get_source()`**: Returns the source from metadata
- **`get_timestamp()`**: Returns creation timestamp
- **`get_tags()`**: Returns list of tags
- **`add_tag(tag)`**: Adds a tag to metadata
- **`set_confidence(confidence)`**: Sets confidence score (with validation)

#### Embedding Conversion
- **`get_embedding_as_numpy()`**: Returns embedding as NumPy array
- **`get_embedding_as_paradma()`**: Returns embedding as Paradma Axiom/TensorAxiom
- **`set_embedding_from_numpy(embedding, use_paradma)`**: Sets embedding from NumPy, optionally converting to Paradma

#### Serialization
- **`to_dict()`**: Converts to dictionary representation
- **`__repr__()`**: Human-readable string representation

## Paradma Integration

### Automatic Type Detection
The module automatically detects if Paradma is available and handles both types seamlessly:
```python
# Works with NumPy
unit = KnowledgeUnit(
    content="Example",
    embedding=np.array([1, 2, 3])
)

# Works with Paradma Axiom
from paradma import Axiom
unit = KnowledgeUnit(
    content="Example",
    embedding=Axiom([1, 2, 3])
)
```

### Type Conversion Utilities
Uses `parag.utils.paradox_utils` for conversion:
- `ensure_paradma_type()`: Convert to Paradma
- `ensure_numpy_type()`: Convert to NumPy
- `numpy_to_axiom()`: Create Axiom from NumPy 1D array
- `numpy_to_tensor_axiom()`: Create TensorAxiom from NumPy multi-dimensional array

## Usage Examples

### Basic Creation
```python
from parag import KnowledgeUnit
import numpy as np

# Create a simple unit
unit = KnowledgeUnit(
    content="Paradma learns NumPy operations",
    embedding=np.array([0.1, 0.2, 0.3]),
    metadata={"source": "documentation.md"}
)

print(unit.unit_id)  # Auto-generated ID
print(unit.get_timestamp())  # Auto-added timestamp
```

### With Tags and Confidence
```python
unit = KnowledgeUnit(
    content="Important fact",
    confidence=0.95
)

unit.add_tag("verified")
unit.add_tag("core-concept")
print(unit.get_tags())  # ['verified', 'core-concept']
```

### Working with Embeddings
```python
# Set embedding from NumPy
embedding = np.random.randn(768)
unit.set_embedding_from_numpy(embedding, use_paradma=True)

# Get as NumPy (regardless of internal type)
numpy_emb = unit.get_embedding_as_numpy()

# Get as Paradma (if available)
paradma_emb = unit.get_embedding_as_paradma()

print(f"Using Paradma: {unit.is_using_paradma()}")
```

### Serialization
```python
# Convert to dictionary
data = unit.to_dict()
print(data.keys())
# ['unit_id', 'content', 'has_embedding', 'embedding_shape', 'metadata', 'confidence']
```

## Dependencies
- **NumPy**: For array operations
- **Paradma** (optional): For self-learning math backend
- **`parag.utils.paradox_utils`** (optional): For type conversions

## Design Patterns
- **Dataclass**: Uses Python dataclasses for clean API
- **Auto-initialization**: Sensible defaults with `__post_init__`
- **Graceful Degradation**: Works without Paradma if unavailable
- **Type Safety**: Validates confidence scores and embedding types

## Error Handling
- Raises `ValueError` if confidence is not in [0.0, 1.0] range
- Falls back to NumPy if Paradma conversion fails
- Handles missing utilities gracefully

## Future Enhancements
- Multi-modal content support (images, audio, video)
- Lazy embedding generation
- Compression for large embeddings
- Distributed storage support
