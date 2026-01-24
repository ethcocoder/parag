# `vectorstore/index_manager.py` - Version Control

## Overview
`IndexManager` is a utility class for managing the lifecycle of FAISS indices on disk. It handles versioning, allows switching between different index iterations, and manages the metadata associated with each saved version.

## Key Features
- **Semantic Versioning**: Automatically generates timestamps or allows custom names for index versions.
- **Atomic Updates**: Ensures metadata is updated alongside the index files.
- **Cleanup**: Provides methods to safely delete old versions and free up disk space.

## Core Methods

### `create_index(vector_store, version)`
Persists the provided `FAISSVectorStore` and records its stats (size, dimension, type) in a central `index_metadata.json` file.

### `load_index(version)`
Reconstructs a `FAISSVectorStore` instance using the parameters saved in the metadata for the requested version.

### `list_versions()`
Returns a list of all currently saved index version names.

### `set_current_version(version)`
Updates the global "current" pointer in the metadata file, used as the default by RAG components when no version is specified.

## Metadata Structure
The `index_metadata.json` stores:
- **`created_at`**: ISO timestamp.
- **`size`**: Number of vectors in the index.
- **`dimension`**: Vector size.
- **`index_type`**: FAISS configuration (e.g., "Flat").

## Usage Example
```python
from parag.vectorstore import IndexManager, FAISSVectorStore

manager = IndexManager(base_path="./storage/indices")

# Save a new version
store = FAISSVectorStore(dimension=384)
# ... add vectors ...
manager.create_index(store, version="q1_knowledge_base")

# Load it later
loaded_store = manager.load_index("q1_knowledge_base")

# See what's available
print(manager.list_versions())
```

## Implementation Details
- **Directory Structure**: Each version is stored in its own sub-folder within the `base_path`.
- **Defaulting**: If no version is specified during `load_index`, it automatically attempts to load the most recently created version.
