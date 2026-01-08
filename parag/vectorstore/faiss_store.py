"""
FAISS vector store implementation.

Provides efficient similarity search using Facebook's FAISS library.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from pathlib import Path


class FAISSVectorStore:
    """
    Vector store using FAISS for similarity search.
    
    Supports multiple index types and provides save/load functionality.
    """
    
    def __init__(
        self,
        dimension: int,
        index_type: str = "Flat",
        metric: str = "L2",
    ):
        """
        Initialize FAISS vector store.
        
        Args:
            dimension: Dimensionality of vectors
            index_type: Type of FAISS index ('Flat', 'IVFFlat', 'HNSW')
            metric: Distance metric ('L2' or 'IP' for inner product)
        """
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError(
                "faiss is required. Install with: "
                "pip install faiss-cpu (or faiss-gpu for GPU support)"
            )
        
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        
        # Create index
        self.index = self._create_index()
        
        # Metadata storage (maps vector ID to metadata)
        self.metadata_map: Dict[int, Dict[str, Any]] = {}
        self._next_id = 0
    
    def _create_index(self):
        """Create FAISS index based on configuration."""
        if self.metric == "L2":
            if self.index_type == "Flat":
                return self.faiss.IndexFlatL2(self.dimension)
            elif self.index_type == "IVFFlat":
                quantizer = self.faiss.IndexFlatL2(self.dimension)
                return self.faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            elif self.index_type == "HNSW":
                return self.faiss.IndexHNSWFlat(self.dimension, 32)
        elif self.metric == "IP":
            if self.index_type == "Flat":
                return self.faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "IVFFlat":
                quantizer = self.faiss.IndexFlatIP(self.dimension)
                return self.faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        
        raise ValueError(f"Unsupported index type or metric: {self.index_type}, {self.metric}")
    
    def add(
        self,
        vectors: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[int]:
        """
        Add vectors to the index.
        
        Args:
            vectors: Array of vectors to add (shape: [n, dimension])
            metadata: Optional list of metadata dicts for each vector
            
        Returns:
            List of assigned IDs
        """
        # Ensure vectors are float32
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        
        # Ensure 2D array
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        # Validate dimension
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} doesn't match "
                f"index dimension {self.dimension}"
            )
        
        # Add to index
        num_vectors = vectors.shape[0]
        ids = list(range(self._next_id, self._next_id + num_vectors))
        
        self.index.add(vectors)
        
        # Store metadata
        if metadata:
            if len(metadata) != num_vectors:
                raise ValueError("Number of metadata items must match number of vectors")
            
            for i, meta in enumerate(metadata):
                self.metadata_map[ids[i]] = meta
        else:
            for i in ids:
                self.metadata_map[i] = {}
        
        self._next_id += num_vectors
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
            - distances: Array of distances to nearest neighbors
            - indices: Array of indices of nearest neighbors
            - metadata: List of metadata dicts (if return_metadata=True)
        """
        # Ensure query is float32
        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)
        
        # Ensure 2D array
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Validate dimension
        if query_vector.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension {query_vector.shape[1]} doesn't match "
                f"index dimension {self.dimension}"
            )
        
        # Search
        distances, indices = self.index.search(query_vector, k)
        
        # Get metadata if requested
        metadata = None
        if return_metadata:
            metadata = []
            for idx in indices[0]:
                if idx != -1:  # -1 indicates no result
                    metadata.append(self.metadata_map.get(int(idx), {}))
                else:
                    metadata.append({})
        
        return distances[0], indices[0], metadata
    
    def batch_search(
        self,
        query_vectors: np.ndarray,
        k: int = 5,
        return_metadata: bool = True,
    ) -> List[Tuple[np.ndarray, np.ndarray, Optional[List[Dict[str, Any]]]]]:
        """
        Search for similar vectors in batch.
        
        Args:
            query_vectors: Array of query vectors (shape: [n, dimension])
            k: Number of nearest neighbors to return per query
            return_metadata: Whether to return metadata
            
        Returns:
            List of (distances, indices, metadata) tuples, one per query
        """
        # Ensure vectors are float32
        if query_vectors.dtype != np.float32:
            query_vectors = query_vectors.astype(np.float32)
        
        # Search
        distances, indices = self.index.search(query_vectors, k)
        
        results = []
        for i in range(len(query_vectors)):
            metadata = None
            if return_metadata:
                metadata = []
                for idx in indices[i]:
                    if idx != -1:
                        metadata.append(self.metadata_map.get(int(idx), {}))
                    else:
                        metadata.append({})
            
            results.append((distances[i], indices[i], metadata))
        
        return results
    
    def save(self, path: str):
        """
        Save index and metadata to disk.
        
        Args:
            path: Directory path to save to
        """
        import pickle
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = path / "index.faiss"
        self.faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        metadata_path = path / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'metadata_map': self.metadata_map,
                'next_id': self._next_id,
                'dimension': self.dimension,
                'index_type': self.index_type,
                'metric': self.metric,
            }, f)
    
    def load(self, path: str):
        """
        Load index and metadata from disk.
        
        Args:
            path: Directory path to load from
        """
        import pickle
        
        path = Path(path)
        
        # Load FAISS index
        index_path = path / "index.faiss"
        self.index = self.faiss.read_index(str(index_path))
        
        # Load metadata
        metadata_path = path / "metadata.pkl"
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.metadata_map = data['metadata_map']
            self._next_id = data['next_id']
            self.dimension = data['dimension']
            self.index_type = data['index_type']
            self.metric = data['metric']
    
    def size(self) -> int:
        """Get number of vectors in the index."""
        return self.index.ntotal
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FAISSVectorStore("
            f"size={self.size()}, "
            f"dim={self.dimension}, "
            f"type={self.index_type}, "
            f"metric={self.metric})"
        )
