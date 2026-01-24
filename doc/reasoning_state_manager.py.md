# `reasoning/state_manager.py` - State Lifecycle

## Overview
`StateManager` is the high-level orchestrator for the `RAGState`. It provides a clean API for building, updating, and summarizing the "mental state" of the RAG system as it processes new information.

## Purpose
- **Session Persistence**: Keeps track of the `current_state` across multiple retrieval steps.
- **Abstraction**: Encapsulates the complexity of creating `Fact` objects from `KnowledgeUnit` objects.
- **Evolusion**: Handles the merging of knowledge from different stages of a cognitive loop.

## Core Methods

### `build_from_retrieval(result)`
Initializes a new `RAGState` directly from a `RetrievalResult`. This is typically the starting point of a reasoning session.

### `update_state(units, detect_conflicts)`
Adds new knowledge bits to the existing state.
- If `detect_conflicts` is True, it automatically runs the `ConflictDetector` on the updated state.

### `merge_states(*states)`
Combines multiple `RAGState` objects into one. This is useful in multi-agent or multi-threaded retrieval scenarios where different "thoughts" need to be fused.

### `state_summary()`
Returns a dictionary representation of the current knowledge, uncertainty, and active conflicts.

## Usage Example
```python
from parag.reasoning import StateManager

manager = StateManager()

# First retrieval
res1 = retriever.search("Paradox Framework")
manager.build_from_retrieval(res1)

# Second retrieval (refined)
res2 = retriever.search("Paradox Framework Cognitive Loop")
manager.update_state(res2.units)

# Check what we know now
print(manager.state_summary()["num_facts"])
```

## Implementation Details
- **Singleton-like Behavior**: While not a strict singleton, usually one `StateManager` instance corresponds to one user session or one query lifecycle.
- **Automatic Fact Aggregation**: Internally calls `RAGState.add_knowledge_unit`, which handles the heavy lifting of fact extraction.
