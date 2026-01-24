# `reasoning/conflict_detector.py` - Contradiction Detection

## Overview
`ConflictDetector` is a logic engine designed to identify inconsistencies within retrieved information. It prevents the RAG system from blindly accepting contradictory data by flagging units that disagree on facts.

## Features
- **Negation Detection**: Identifies "X is Y" vs "X is not Y" patterns.
- **Numeric Contradiction**: Flags disagreeing numbers for the same entity (e.g., "Price is $10" vs "Price is $20").
- **Entity Resolution**: Heuristic matching to ensure the conflict is truly about the same subject.
- **Automated Resolution**: Strategies for picking a "winning" fact.

## Core Methods

### `detect_conflicts_in_state(state)`
The primary method for analyzing a `RAGState`. It compares every fact in the state against every other fact.

### `detect_conflicts_in_units(units)`
Checks for conflicts in a raw list of `KnowledgeUnit` objects before they are aggregated into a state.

### `resolve_conflict(fact1, fact2, strategy)`
Arbitrates between two conflicting facts.
- **`highest_confidence`**: Picks the fact with the higher numeric confidence score.
- **`most_sources`**: Picks the fact supported by more original knowledge units.
- **`newest`**: Picks the most recently extracted info.

## Heuristic Engines

### `_has_negation_conflict`
Uses a predefined list of negation words (not, never, isn't, etc.). If one sentence contains a negation and they share significant keywords, it flags a conflict.

### `_has_numeric_conflict`
Uses regular expressions to extract numbers. If two highly similar sentences contain different numbers, it flags a numeric contradiction.

### `_has_entity_conflict`
A modular placeholder for opposite adjective detection (true/false, possible/impossible, safe/unsafe).

## Usage Example
```python
from parag.reasoning import ConflictDetector

detector = ConflictDetector(similarity_threshold=0.85)

# Detect conflicts in a state
conflicts = detector.detect_conflicts_in_state(state)

for id1, id2, reason in conflicts:
    print(f"Conflict found between {id1} and {id2}: {reason}")
    
    # Resolve it
    winner = detector.resolve_conflict(state.facts[id1], state.facts[id2])
    print(f"Winning info: {winner.content}")
```

## Paradox Context
Conflict detection is a precursor to `Reflexion` in the Paradox engine. When many conflicts are detected, the `RAGState` metadata typically updates to reflect a state of agitation or critical thinking, signaling the need for more diverse retrieval.
