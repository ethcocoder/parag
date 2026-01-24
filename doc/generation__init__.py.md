# `generation/__init__.py` - Generation Module Entry

## Overview
Package initialization for the Parag generation module. This module provides the tools necessary to convert retrieved knowledge into natural language responses, supporting both LLM-based and deterministic generation.

## Purpose
- **API Surface**: Exports the core generation components.
- **Unified Logic**: Provides a single point of entry for building prompts and generating responses.

## Exported Components
- **`PromptBuilder`**: Handles the construction of prompts from `RetrievalResult` or `RAGState`.
- **`LLMAdapter`**: Abstract base class for integrating various LLM backends (OpenAI, Anthropic, local models).
- **`DeterministicGenerator`**: A "no-LLM" generator that produces structured, rule-based responses from `RAGState`.

## Usage Example
```python
from parag.generation import PromptBuilder, DeterministicGenerator

# Build a prompt from the current state
builder = PromptBuilder()
prompt = builder.build_from_state("What is the Paradox core?", state)

# Or generate a response without an expensive LLM call
generator = DeterministicGenerator()
response = generator.generate_from_state("What is the Paradox core?", state)
```

## Module Structure
- `prompt_builder.py`: Logic for formatting context and queries.
- `llm_adapter.py`: Interfaces for language model interaction.
