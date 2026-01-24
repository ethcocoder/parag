# `utils/paradox_utils.py` - Paradox Ecosystem Bridge

## Overview
`paradox_utils` is the "glue" that allows Parag to interact seamlessly with Paradma (the self-learning math engine) and `modules.framework` (the custom tensor/autograd library). It provides specialized conversion for every data type transition required in a cognitive RAG pipeline.

## Type Conversion Support
The module handles four primary data states:
1. **NumPy**: Standard scientific computing arrays.
2. **Paradma (Axiom/TensorAxiom)**: Symbolic, self-learning math units.
3. **Framework (Tensor)**: Neural network tensors with gradient tracking.
4. **Native Python**: Lists and scalars.

## Core Conversion Functions

### Paradma Conversions
- **`numpy_to_axiom(arr)`**: Converts 1D data to a Paradma `Axiom`.
- **`numpy_to_tensor_axiom(arr)`**: Converts multi-dimensional arrays to `TensorAxiom`.
- **`to_paradma(data)`**: A high-level helper that automatically detects the input type and converts it to the appropriate Paradma structure.

### Framework Conversions
- **`paradma_to_framework_tensor(axiom)`**: Moves data from the learning manifold into the neural framework.
- **`framework_tensor_to_paradma(tensor)`**: Moves neural weights/activations back into the self-learning math engine.

### NumPy Exports
- **`axiom_to_numpy(axiom)`**: Standardizes Paradma data for external processing.
- **`framework_tensor_to_numpy(tensor)`**: Standardizes framework tensors for external processing.
- **`to_numpy(data)`**: Universal utility to get a NumPy array from any Paradox type.

## Advanced Features

### Gradient Synchronization
- **`sync_gradients(framework_tensor, paradma_axiom)`**: A unique utility that transfers gradient information from the framework (backpropagation) into Paradma (pattern analysis). This allows Paradma to "learn from the gradients" of a RAG query or embedding task.

### Environment Checks
- **`ensure_paradma_available()`**: Raises a runtime error if the library is missing, preventing cryptic import failures.
- **`ensure_framework_available()`**: Similar check for the neural framework.

## Usage Example
```python
from parag.utils.paradox_utils import to_paradma, to_numpy, sync_gradients

# Convert a raw list or numpy array for Paradma learning
axiom = to_paradma([1.0, 2.0, 3.5])

# Sync neural gradients to the learning manifold
sync_gradients(my_neural_tensor, axiom)

# Export processed data back to numpy for visualization
data = to_numpy(axiom)
```

## Implementation Details
- **Type Aliases**: Defines `AnyArrayType` as a union of all supported Paradox and Python array types.
- **Manifold Management**: `get_learning_manifold()` provides safe access to the global Paradma learning state.
- **Redundancy-Safe**: Includes protection against double-wrapping or conflicting manifold assignments.
