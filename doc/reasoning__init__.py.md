# `reasoning/__init__.py` - Reasoning Module Entry

## Overview
Package initialization for the Parag reasoning module. This module provides advanced logic for managing state, detecting conflicts, and calculating uncertainty across the RAG pipeline.

## Purpose
- **Reasoning Engine**: Exports the core components that provide "thought" and "validation" to the RAG system.
- **Unified Logic**: Simplifies access to the state manager, conflict detector, and uncertainty calculator.

## Exported Components
- **`StateManager`**: Orchestrates the evolution of a `RAGState` as new information is retrieved.
- **`ConflictDetector`**: Analyzes KnowledgeUnits to find contradictions (negations, numeric mismatches).
- **`UncertaintyCalculator`**: Quantifies how "sure" the system is about the current state.

## Usage Example
```python
from parag.reasoning import StateManager, ConflictDetector, UncertaintyCalculator

# Build a state from retrieval
manager = StateManager()
state = manager.build_from_retrieval(result)

# Detect if the retrieved info contradicts itself
detector = ConflictDetector()
conflicts = detector.detect_conflicts_in_state(state)

# Calculate how much we can trust this state
calculator = UncertaintyCalculator()
metrics = calculator.calculate_state_uncertainty(state)
```

## Module Structure
- `state_manager.py`: High-level state lifecycle management.
- `conflict_detector.py`: Heuristics for identifying internal contradictions.
- `uncertainty.py`: Confidence and completeness metrics.
