# parag/core/rag_state.py

## Overview
The data structure representing the "Mind State" of the RAG system at a specific moment.

## Purpose
Most RAG systems just pass a list of text chunks to an LLM. `parag` constructs a `RAGState` object that explicitly models:
- **Facts**: Atomic units of truth extracted from documents.
- **Conflicts**: Explicit contradictions between facts.
- **Uncertainty**: A numerical score (0.0 to 1.0) of how confident the system is.
- **Emotions**: (If enabled) The emotional reaction to the knowledge.

## Key Component

### `RAGState` (Class)

- **`detect_conflicts()`**: 
  - Uses heuristics (or potentially an LLM) to check if any two facts contradict each other.
- **`_calculate_uncertainty()`**: 
  - Increases if facts conflict or if confidence scores are low.
  - Used by the `CognitiveLoop` to decide if more research is needed.
- **`to_dict()`**: Serializes the mind state for debugging or UI visualization.

## Significance
This moves RAG from "Stateless Search" to "Stateful Reasoning." The system knows *that* it doesn't know.
