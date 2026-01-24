# `ingestion/metadata.py` - Metadata Management

## Overview
The `metadata` module handles the enrichment of document data. It extracts environmental information (file stats), content properties (word count, hashes), and manages the merging and filtering of metadata across the ingestion pipeline.

## Key Components

### `MetadataExtractor` Class
The central manager for document attributes.

#### Core Methods
- **`extract(file_path, content, custom_metadata)`**: The primary extraction engine. It combines file stats, content analysis, and user-provided metadata into a single dictionary.
- **`_extract_file_metadata()`**: Collects absolute path, file size, and creation/modification timestamps.
- **`_extract_content_metadata()`**: Computes content length, word counts, and line counts.
- **`_hash_content()`**: Generates a SHA-256 content hash (first 16 chars) to facilitate deduplication.
- **`validate(metadata, required_fields)`**: Ensures a metadata dictionary contains the minimum necessary keys for the system to function correctly.

## Helper Functions
- **`add_tags(metadata, tags)`**: Safely appends a list of strings to the `tags` field in the metadata dictionary.
- **`filter_metadata(metadata, keep_fields)`**: Returns a copy of the metadata containing only the requested keys, useful for reducing storage overhead in vector stores.

## Usage Example
```python
from parag.ingestion import MetadataExtractor, add_tags

extractor = MetadataExtractor(custom_fields=["author", "department"])

# Extract everything at once
meta = extractor.extract(
    file_path="reports/Q1_Paradox.txt",
    content="This is the report content...",
    custom_metadata={"department": "AI Research"}
)

# Add some tags later
meta = add_tags(meta, ["internal", "high-priority"])

print(f"Hash: {meta['content_hash']}")
print(f"Extracted At: {meta['extracted_at']}")
```

## Features
- **Deduplication Ready**: By generating content hashes, the system can easily skip re-ingesting the same information.
- **Temporal Tracking**: Automatically adds an `extracted_at` timestamp.
- **Strict Validation**: Allows defining "required" fields to ensure data integrity before saving to a vector store.
