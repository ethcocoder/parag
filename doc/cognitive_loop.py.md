# parag/core/cognitive_loop.py

## Overview
The "Consciousness" of the RAG system. The `CognitiveLoop` allows the knowledge retrieval system to "Think about what it knows" and actively seek more information if it's confused, rather than just returning the first result.

## Purpose
Classic RAG (Retrieval-Augmented Generation) is linear: Query $\to$ Search $\to$ Answer.
Parag's Cognitive Loop is circular: Query $\to$ Search $\to$ **Reflect** (Is this enough? Is it conflicting?) $\to$ New Query $\to$ Search...

## Key Component

### `CognitiveLoop` (Class)

- **`run(query)`**: Orchestrates the multi-turn thinking process.
- **`_is_equilibrium_reached(state)`**: 
  - Checks if the current `RAGState` has low uncertainty and high emotional stability (Equilibria).
  - Unless equilibrium is reached, the loop continues.
- **`_generate_reflexive_query(original_query, state)`**: 
  - If the AI detects a conflict (e.g., one document says X, another says Y), it generates a specific query: *"evidence for X vs Y"*.
  - If it detects missing info: *"Information about [Topic] specifically regarding [Missing Detail]"*.

## Usage
```python
from parag.core.cognitive_loop import CognitiveLoop

loop = CognitiveLoop(retriever=my_retriever)
final_state = loop.run("What caused the fall of Rome?")
# The loop might auto-query about "economic factors" if the first search was vague.
```
