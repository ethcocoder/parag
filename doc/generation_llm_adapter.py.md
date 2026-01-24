# `generation/llm_adapter.py` - LLM Integration Layers

## Overview
Defines the interfaces and fallback mechanisms for generating text. This module introduces the `LLMAdapter` abstraction and the `DeterministicGenerator`, a unique Paradox feature that allows for "thinking" without always requiring a language model.

## Key Components

### `LLMAdapter` (Abstract Class)
The bridge between Parag and large language models.
- **`generate(prompt, max_tokens)`**: Abstract method to be implemented by specific backends (e.g., GPT, Claude, or a Paradox native transformer).

### `DeterministicGenerator` (Class)
Generates responses by analyzing the structured `RAGState` directly. It produces explainable, rule-based answers when LLM use is either unnecessary or potentially unreliable.

#### Key Features
- **Zero-Latency**: No network or heavy GPU inference for simple facts.
- **Explainable Reasoning**: Includes an explicit "Reasoning" section in the output.
- **Cognitive Integration**: Can report Paradox-specific signals (emotions, manifold curvature, intuition) as part of the justification.
- **Safety First**: Automatically triggered when metadata suggests high uncertainty.

#### Core Methods
- **`generate_from_state(query, state)`**: Primary entry point. Checks for data sufficiency, gathers high-confidence facts, and builds a multi-part response.
- **`should_use_llm(state)`**: A decision logic method that returns `False` if:
  - Uncertainty is too high (> 0.6).
  - There are excessive conflicts in the data (> 2).
  - Too little information is available.
- **`_generate_insufficient_response()`**: Creates a detailed report on why an answer cannot be given.

## Paradox Integration
The `DeterministicGenerator` is uniquely aware of the Paradox ecosystem:
- **AI Emotions**: Reports signals like `Reflexion` or `Fluxion` if found in the RAG metadata.
- **Manifold Curvature**: Interprets Paradma-related geometric signals to explain the "complexity" or "contradiction" in the knowledge space.

## Usage Example
```python
from parag.generation import DeterministicGenerator

generator = DeterministicGenerator()

# Check if we should even bother calling an LLM
if not generator.should_use_llm(state):
    response = generator.generate_from_state(query, state)
    print(response)
else:
    # Safe to call LLM provider
    response = my_llm_adapter.generate(builder.build_from_state(query, state))
```

## Error & Edge Case Handling
- **`_generate_low_confidence_response`**: Triggered when facts exist but none meet the reliability threshold (0.7 by default).
- **Conflict Warnings**: Adds visible warnings to the user if the underlying knowledge units contradict each other.
