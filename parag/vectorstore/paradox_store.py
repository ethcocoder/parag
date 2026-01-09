"""
ParadoxVectorStore: Vector storage using ParadoxLF LatentMemoryEngine.

Replaces FAISS with autonomous, self-learning memory engine powered by Paradma.
"""

import sys
import os
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import ParadoxLF engine
try:
    from paradoxlf.paradox.engine import LatentMemoryEngine
    PARADOXLF_AVAILABLE = True
except ImportError:
    PARADOXLF_AVAILABLE = False
    LatentMemoryEngine = None

from parag.utils.paradox_utils import ensure_numpy_type, ensure_paradma_type


class ParadoxVectorStore:
    """
    Vector store using ParadoxLF LatentMemoryEngine with Paradma backend.
    
    Features:
    - Paradma-powered vector storage (self-learning)
    - Autonomous memory optimization
    - Conceptual search capabilities
    - Creative concept blending via imagine()
    - Temporal prediction support
    """
    
    def __init__(
        self,
        dimension: int,
        storage_dir: Optional[str] = None,
    ):
        """
        Initialize ParadoxVectorStore.
        
        Args:
            dimension: Dimensionality of vectors
            storage_dir: Optional directory for persistence
        """
        if not PARADOXLF_AVAILABLE:
            raise ImportError(
                "ParadoxLF is not available. Ensure paradoxlf module is in sys.path."
            )
        
        self.dimension = dimension
        self.storage_dir = storage_dir
        
        # Initialize ParadoxLF LatentMemoryEngine with Paradma backend
        self.engine = LatentMemoryEngine(
            dimension=dimension,
            backend="paradma",  # Use Paradma backend exclusively
            storage_dir=storage_dir
        )
        
        # Track metadata separately (engine handles vectors)
        self.metadata_map: Dict[int, Dict[str, Any]] = {}
        self._next_id = 0
    
    def add(
        self,
        vectors: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[int]:
        """
        Add vectors to the store.
        
        Args:
            vectors: Array of vectors to add (shape: [n, dimension])
            metadata: Optional list of metadata dicts for each vector
            
        Returns:
            List of assigned IDs
        """
        # Ensure 2D array
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        # Validate dimension
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} doesn't match "
                f"store dimension {self.dimension}"
            )
        
        # Add each vector to engine
        num_vectors = vectors.shape[0]
        ids = []
        
        for i in range(num_vectors):
            vector = vectors[i]
            
            # Prepare metadata for engine
            attrs = metadata[i].copy() if metadata and i < len(metadata) else {}
            attrs['_store_id'] = self._next_id
            
            # Add to ParadoxLF engine
            # Engine handles conversion to Axiom internally via Paradma backend
            obj_id = self.engine.add(vector.tolist(), attributes=attrs)
            
            # Store metadata in our map
            self.metadata_map[self._next_id] = attrs
            
            ids.append(self._next_id)
            self._next_id += 1
        
        return ids
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        return_metadata: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[List[Dict[str, Any]]]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector (shape: [dimension] or [1, dimension])
            k: Number of nearest neighbors to return
            return_metadata: Whether to return metadata
            
        Returns:
            Tuple of (distances, indices, metadata)
        """
        # Ensure 1D vector
        if query_vector.ndim > 1:
            query_vector = query_vector.flatten()
        
        # Validate dimension
        if query_vector.shape[0] != self.dimension:
            raise ValueError(
                f"Query dimension {query_vector.shape[0]} doesn't match "
                f"store dimension {self.dimension}"
            )
        
        # Query ParadoxLF engine (uses Paradma for distance calculations)
        results = self.engine.query(
            query_vector.tolist(),
            k=k,
            metric="cosine",  # Use cosine for semantic similarity
            include_vectors=False
        )
        
        # Extract results
        if not results:
            return np.array([]), np.array([]), [] if return_metadata else None
        
        distances = []
        indices = []
        metadata_list = []
        
        for obj_id, distance, attrs in results:
            # Get our internal ID
            store_id = attrs.get('_store_id', obj_id)
            
            distances.append(distance)
            indices.append(store_id)
            
            if return_metadata:
                metadata_list.append(self.metadata_map.get(store_id, attrs))
        
        return (
            np.array(distances),
            np.array(indices),
            metadata_list if return_metadata else None
        )
    
    def conceptual_search(
        self,
        concept: str,
        k: int = 5,
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        High-level conceptual search using engine's encoder (if set).
        
        Args:
            concept: Text concept to search for
            k: Number of results
            
        Returns:
            List of (id, distance, metadata) tuples
        """
        if self.engine.encoder is None:
            raise ValueError(
                "Conceptual search requires an encoder. "
                "Set engine.encoder first or use search() with vectors."
            )
        
        results = self.engine.conceptual_search(concept, k=k)
        
        # Convert to our format
        formatted_results = []
        for obj_id, distance, attrs in results:
            store_id = attrs.get('_store_id', obj_id)
            formatted_results.append((
                store_id,
                distance,
                self.metadata_map.get(store_id, attrs)
            ))
        
        return formatted_results
    
    def imagine(
        self,
        vector_a: np.ndarray,
        vector_b: np.ndarray,
        ratio: float = 0.5
    ) -> np.ndarray:
        """
        Creative concept blending using ParadoxLF's imagine().
        
        Args:
            vector_a: First vector
            vector_b: Second vector
            ratio: Blending ratio (0.0 = all A, 1.0 = all B)
            
        Returns:
            Blended vector (NumPy array)
        """
        # Use engine's creative blending
        blended = self.engine.imagine(
            vector_a,
            vector_b,
            ratio=ratio
        )
        
        return np.array(blended)
    
    def predict_future(
        self,
        history_vectors: List[np.ndarray],
        steps: int = 1
    ) -> List[np.ndarray]:
        """
        Predict future vectors based on historical trajectory.
        
        Args:
            history_vectors: List of historical vectors
            steps: Number of steps to predict
            
        Returns:
            List of predicted vectors
        """
        # Convert to list format for engine
        history = [v.tolist() for v in history_vectors]
        
        # Use engine's temporal prediction
        predictions = self.engine.predict_future(history, steps=steps)
        
        # Convert back to NumPy
        return [np.array(p) for p in predictions]
    
    def save(self, path: Optional[str] = None):
        """
        Save vector store to disk.
        
        Args:
            path: Optional path (uses storage_dir if not provided)
        """
        if path:
            # Temporarily override storage_dir
            original_dir = self.engine.storage_dir
            self.engine.storage_dir = path
            self.engine.persistence.storage_dir = path
        
        # Save engine state (includes Paradma vectors)
        self.engine.save()
        
        # Save metadata map
        import pickle
        from pathlib import Path
        
        storage_path = Path(path or self.storage_dir)
        storage_path.mkdir(parents=True, exist_ok=True)
        
        metadata_file = storage_path / "parag_metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'metadata_map': self.metadata_map,
                'next_id': self._next_id,
                'dimension': self.dimension,
            }, f)
        
        if path:
            # Restore original storage_dir
            self.engine.storage_dir = original_dir
            self.engine.persistence.storage_dir = original_dir
    
    def load(self, path: str):
        """
        Load vector store from disk.
        
        Args:
            path: Path to load from
        """
        import pickle
        from pathlib import Path
        
        storage_path = Path(path)
        
        # Load metadata map
        metadata_file = storage_path / "parag_metadata.pkl"
        if metadata_file.exists():
            with open(metadata_file, 'rb') as f:
                data = pickle.load(f)
                self.metadata_map = data['metadata_map']
                self._next_id = data['next_id']
                self.dimension = data['dimension']
        
        # Engine loads its own state automatically in __init__
        # If we need to reload, recreate engine
        self.engine = LatentMemoryEngine(
            dimension=self.dimension,
            backend="paradma",
            storage_dir=path
        )
    
    def size(self) -> int:
        """Get number of vectors in the store."""
        return self.engine.count
    
    def get_info(self) -> Dict[str, Any]:
        """Get store information."""
        return {
            "backend": "paradoxlf+paradma",
            "dimension": self.dimension,
            "size": self.size(),
            "storage_dir": self.storage_dir,
            "engine_info": self.engine.get_info(),
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ParadoxVectorStore("
            f"size={self.size()}, "
            f"dim={self.dimension}, "
            f"backend=ParadoxLF+Paradma)"
        )
