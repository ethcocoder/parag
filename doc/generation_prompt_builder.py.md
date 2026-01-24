# `generation/prompt_builder.py` - Prompt Engineering

## Overview
`PromptBuilder` is responsible for transforming raw data—either from a `RetrievalResult` or an aggregated `RAGState`—into high-quality prompts suitable for consumption by an LLM.

## Features
- **Context Formatting**: Automatically numbers and formats retrieved knowledge units.
- **Metataching**: Optionally includes source attribution and confidence scores in the prompt context.
- **Conflict Awareness**: Visually marks conflicting facts for the LLM's attention.
- **Uncertainty Context**: Appends overall state uncertainty values to guide the LLM's level of confidence.

## Core Methods

### `build_from_result(query, result)`
Constructs a prompt directly from the output of a retriever.
- Best for simple "Retrieval -> Generation" pipelines.
- Includes context blocks [1], [2], etc., with optional source and confidence lines.

### `build_from_state(query, state)`
Constructs a prompt from the intelligent `RAGState`.
- Best for "Reasoning -> Generation" pipelines.
- Includes aggregated facts rather than raw chunks.
- Adds warnings about detected conflicts or high uncertainty.

### `build_deterministic_response(query, state)`
Mirroring the `DeterministicGenerator` logic, this builds a human-readable response without calling an LLM. 

## Configuration
The `PromptBuilder` can be customized during initialization:
- **`system_prompt`**: Overrides the default "helpful AI assistant" prompt.
- **`include_sources`**: If `True`, adds "Source: X" to each context block.
- **`include_confidence`**: If `True`, adds the numeric confidence score to each entry.

## Usage Example
```python
from parag.generation import PromptBuilder

builder = PromptBuilder(
    include_sources=True,
    include_confidence=True
)

# Build prompt from aggregated reasoning state
prompt = builder.build_from_state("Who founded the Paradox project?", state)

# The prompt now includes a curated list of facts + conflict warnings
```

## Internal Logic Details
- **Default System Prompt**: "You are a helpful AI assistant. Answer questions based on the provided context. If the information is insufficient, say so clearly."
- **Uncertainty Visualization**: High uncertainty (> 0.5) triggers an explicit "Note: Information uncertainty is high" block at the bottom of the context.
