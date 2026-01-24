"""
Paradox Utilities: Conversion between Paradma, NumPy, and Framework types.

Handles conversions between:
- Paradma Axiom/TensorAxiom
- NumPy arrays
- modules.framework.Tensor
"""

import sys
import os
from typing import Any, Union, Optional
import numpy as np

# Add parent directory to path for Paradox imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import Paradma components
try:
    from paradma import learning, Axiom, TensorAxiom
    from paradma.learning_manifold import LearningManifold
    PARADMA_AVAILABLE = True
except ImportError:
    PARADMA_AVAILABLE = False
    Axiom = None
    TensorAxiom = None
    learning = None
    LearningManifold = None

# Import modules.framework components
try:
    from modules.framework.tensor import Tensor as FrameworkTensor
    FRAMEWORK_AVAILABLE = True
except ImportError:
    FRAMEWORK_AVAILABLE = False
    FrameworkTensor = None


# Type aliases
ParadmaType = Union['Axiom', 'TensorAxiom']
FrameworkType = 'FrameworkTensor'
NumpyType = np.ndarray
AnyArrayType = Union[ParadmaType, FrameworkType, NumpyType, list, float]


def ensure_paradma_available():
    """Raise error if Paradma is not available."""
    if not PARADMA_AVAILABLE:
        raise ImportError(
            "Paradma is not available. Ensure paradma module is in sys.path.\n"
            f"Current path: {sys.path}"
        )


def ensure_framework_available():
    """Raise error if modules.framework is not available."""
    if not FRAMEWORK_AVAILABLE:
        raise ImportError(
            "modules.framework is not available. Ensure modules directory is in sys.path."
        )


def numpy_to_axiom(arr: Union[np.ndarray, list, float], manifold: Optional[LearningManifold] = None) -> 'Axiom':
    """
    Convert NumPy array or list to Paradma Axiom.
    
    Args:
        arr: NumPy array, list, or scalar
        manifold: Optional manifold (defaults to learning manifold)
        
    Returns:
        Axiom wrapping the data
    """
    ensure_paradma_available()
    
    if manifold is None:
        manifold = learning
    
    # Convert to list if NumPy array
    if isinstance(arr, np.ndarray):
        arr = arr.tolist() if arr.ndim > 0 else float(arr)
    
    return Axiom(arr, manifold=manifold)


def numpy_to_tensor_axiom(arr: np.ndarray, manifold: Optional[LearningManifold] = None) -> 'TensorAxiom':
    """
    Convert NumPy array to Paradma TensorAxiom.
    
    Args:
        arr: NumPy array
        manifold: Optional manifold (defaults to learning manifold)
        
    Returns:
        TensorAxiom wrapping the data
    """
    ensure_paradma_available()
    
    if manifold is None:
        manifold = learning
    
    return TensorAxiom(arr, manifold=manifold)


def axiom_to_numpy(axiom: 'Axiom') -> np.ndarray:
    """
    Convert Paradma Axiom to NumPy array.
    
    Args:
        axiom: Paradma Axiom
        
    Returns:
        NumPy array
    """
    # Get the value from Axiom
    value = axiom.value if hasattr(axiom, 'value') else axiom
    
    # Convert to NumPy
    if isinstance(value, np.ndarray):
        return value
    elif isinstance(value, (list, tuple)):
        return np.array(value)
    else:
        return np.array([value])


def tensor_axiom_to_numpy(tensor_axiom: 'TensorAxiom') -> np.ndarray:
    """
    Convert Paradma TensorAxiom to NumPy array.
    
    Args:
        tensor_axiom: Paradma TensorAxiom
        
    Returns:
        NumPy array
    """
    # TensorAxiom stores data as .data attribute
    if hasattr(tensor_axiom, 'data'):
        return np.array(tensor_axiom.data)
    elif hasattr(tensor_axiom, 'value'):
        return np.array(tensor_axiom.value)
    else:
        return np.array(tensor_axiom)


def framework_tensor_to_numpy(tensor: 'FrameworkTensor') -> np.ndarray:
    """
    Convert modules.framework.Tensor to NumPy array.
    
    Args:
        tensor: modules.framework.Tensor
        
    Returns:
        NumPy array
    """
    if hasattr(tensor, 'data'):
        data = tensor.data
        # Handle CuPy arrays (convert to NumPy)
        if hasattr(data, 'get'):
            return data.get()
        return np.array(data)
    else:
        return np.array(tensor)


def framework_tensor_to_paradma(tensor: 'FrameworkTensor') -> 'TensorAxiom':
    """
    Convert modules.framework.Tensor to Paradma TensorAxiom.
    
    Args:
        tensor: modules.framework.Tensor
        
    Returns:
        TensorAxiom
    """
    ensure_paradma_available()
    
    # Convert to NumPy first
    numpy_arr = framework_tensor_to_numpy(tensor)
    
    # Then to TensorAxiom
    return numpy_to_tensor_axiom(numpy_arr)


def paradma_to_framework_tensor(axiom: ParadmaType, requires_grad: bool = False) -> 'FrameworkTensor':
    """
    Convert Paradma Axiom/TensorAxiom to modules.framework.Tensor.
    
    Args:
        axiom: Paradma Axiom or TensorAxiom
        requires_grad: Whether the tensor should track gradients
        
    Returns:
        modules.framework.Tensor
    """
    ensure_framework_available()
    
    # Convert to NumPy first
    if isinstance(axiom, TensorAxiom):
        numpy_arr = tensor_axiom_to_numpy(axiom)
    else:
        numpy_arr = axiom_to_numpy(axiom)
    
    # Create framework tensor
    return FrameworkTensor(numpy_arr, requires_grad=requires_grad)


def ensure_paradma_type(data: AnyArrayType) -> ParadmaType:
    """
    Ensure data is a Paradma Axiom or TensorAxiom.
    
    Args:
        data: Any array-like data
        
    Returns:
        Axiom or TensorAxiom
    """
    ensure_paradma_available()
    
    # Already Paradma type
    if isinstance(data, (Axiom, TensorAxiom)):
        return data
    
    # Framework tensor
    if FRAMEWORK_AVAILABLE and isinstance(data, FrameworkTensor):
        return framework_tensor_to_paradma(data)
    
    # NumPy array or list
    if isinstance(data, np.ndarray):
        if data.ndim > 1:
            return numpy_to_tensor_axiom(data)
        else:
            return numpy_to_axiom(data)
    
    if isinstance(data, (list, tuple)):
        # Check if nested (multi-dimensional)
        if isinstance(data[0], (list, tuple)):
            return numpy_to_tensor_axiom(np.array(data))
        else:
            return numpy_to_axiom(data)
    
    # Scalar
    return numpy_to_axiom([data])


def ensure_numpy_type(data: AnyArrayType) -> np.ndarray:
    """
    Ensure data is a NumPy array.
    
    Args:
        data: Any array-like data
        
    Returns:
        NumPy array
    """
    # Already NumPy
    if isinstance(data, np.ndarray):
        return data
    
    # Paradma Axiom
    if PARADMA_AVAILABLE and isinstance(data, Axiom):
        return axiom_to_numpy(data)
    
    # Paradma TensorAxiom
    if PARADMA_AVAILABLE and isinstance(data, TensorAxiom):
        return tensor_axiom_to_numpy(data)
    
    # Framework tensor
    if FRAMEWORK_AVAILABLE and isinstance(data, FrameworkTensor):
        return framework_tensor_to_numpy(data)
    
    # List or scalar
    return np.array(data)


def sync_gradients(framework_tensor: 'FrameworkTensor', paradma_axiom: ParadmaType):
    """
    Synchronize gradients between framework tensor and Paradma axiom.
    
    This is useful when using both for different purposes
    (e.g., framework for backprop, Paradma for learning).
    
    Args:
        framework_tensor: modules.framework.Tensor with gradients
        paradma_axiom: Paradma Axiom/TensorAxiom to sync to
    """
    ensure_framework_available()
    ensure_paradma_available()
    
    if not framework_tensor.requires_grad or framework_tensor.grad is None:
        return
    
    # Get gradient as NumPy
    grad_numpy = framework_tensor_to_numpy(framework_tensor.grad)
    
    # Store in Paradma (Paradma doesn't have built-in autograd yet,
    # but we can store as metadata)
    if hasattr(paradma_axiom, 'metadata'):
        paradma_axiom.metadata['gradient'] = grad_numpy


def get_learning_manifold() -> Optional[LearningManifold]:
    """
    Get the global learning manifold.
    
    Returns:
        LearningManifold if available, None otherwise
    """
    if PARADMA_AVAILABLE:
        return learning
    return None


# Convenience aliases
to_paradma = ensure_paradma_type
to_numpy = ensure_numpy_type
