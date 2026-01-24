# `ingestion/loaders.py` - Document Loading Interface

## Overview
`loaders` provides a standardized way to read content from various file formats. Every loader returns a unified list of dictionaries containing `content` and `metadata`, making the rest of the ingestion pipeline file-format agnostic.

## Base Architecture

### `DocumentLoader` (Abstract Class)
All loaders inherit from this base class.
- **`load(file_path)`**: Must be implemented by subclasses.
- **`_extract_basic_metadata(file_path)`**: Automatically extracts absolute path, filename, and extension for every loaded file.

## Provided Loaders

### `TextLoader`
- Simple loader for .txt and other plain text file formats.
- Supports configurable character encoding (default: `utf-8`).

### `PDFLoader`
- Uses `PyPDF2` to extract text from PDF files.
- **Page-Aware**: Returns a separate document entry for each page, allowing for page-level retrieval.
- Adds `page` and `total_pages` to the metadata of each extracted entry.

### `MarkdownLoader`
- Specialized loader for .md files.
- **Frontmatter Support**: Automatically parses YAML-like blocks at the start of files.
- **Header Extraction**: Identifies `#` headers and adds them to the metadata.
- Sets the document `title` to the first H1 header found.

## Autoloading
The module provides a convenience function:
- **`load_document(file_path)`**: Dynamically selects the correct loader based on the file extension and returns the content.

## Usage Example
```python
from parag.ingestion import load_document, MarkdownLoader

# Automatic detection
docs = load_document("guides/Paradox_Setup.md")

# Manual loader with specific settings
loader = MarkdownLoader(extract_frontmatter=True)
docs = loader.load("project_plan.md")

print(f"File Title: {docs[0]['metadata'].get('title')}")
print(f"Frontmatter: {docs[0]['metadata'].get('status')}")
```

## Requirements
- **PDF Support**: Requires the `PyPDF2` package.
- **Other Formats**: Native Python libraries are used for text and markdown.
