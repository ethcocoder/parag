"""
Index management for vector stores.

Handles index lifecycle, versioning, and persistence.
"""

from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import json

from parag.vectorstore.faiss_store import FAISSVectorStore


class IndexManager:
    """
    Manage FAISS index lifecycle.
    
    Provides index versioning, rebuilding, and persistence management.
    """
    
    def __init__(self, base_path: str):
        """
        Initialize IndexManager.
        
        Args:
            base_path: Base directory for storing indices
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.base_path / "index_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load index metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            "versions": {},
            "current_version": None,
        }
    
    def _save_metadata(self):
        """Save index metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def create_index(
        self,
        vector_store: FAISSVectorStore,
        version: Optional[str] = None,
    ) -> str:
        """
        Create and save a new index version.
        
        Args:
            vector_store: FAISS vector store to save
            version: Optional version name (auto-generated if not provided)
            
        Returns:
            Version identifier
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create version directory
        version_path = self.base_path / version
        version_path.mkdir(parents=True, exist_ok=True)
        
        # Save vector store
        vector_store.save(str(version_path))
        
        # Update metadata
        self.metadata["versions"][version] = {
            "created_at": datetime.now().isoformat(),
            "size": vector_store.size(),
            "dimension": vector_store.dimension,
            "index_type": vector_store.index_type,
        }
        self.metadata["current_version"] = version
        self._save_metadata()
        
        return version
    
    def load_index(
        self,
        version: Optional[str] = None,
    ) -> FAISSVectorStore:
        """
        Load an index version.
        
        Args:
            version: Version to load (uses current if not specified)
            
        Returns:
            Loaded FAISS vector store
        """
        if version is None:
            version = self.metadata.get("current_version")
            if version is None:
                raise ValueError("No index versions available")
        
        if version not in self.metadata["versions"]:
            raise ValueError(f"Version {version} not found")
        
        version_path = self.base_path / version
        
        # Load version metadata to get configuration
        version_meta = self.metadata["versions"][version]
        
        # Create empty vector store with same configuration
        vector_store = FAISSVectorStore(
            dimension=version_meta["dimension"],
            index_type=version_meta["index_type"],
        )
        
        # Load from disk
        vector_store.load(str(version_path))
        
        return vector_store
    
    def list_versions(self) -> list:
        """List all available index versions."""
        return list(self.metadata["versions"].keys())
    
    def get_current_version(self) -> Optional[str]:
        """Get current index version."""
        return self.metadata.get("current_version")
    
    def set_current_version(self, version: str):
        """Set the current index version."""
        if version not in self.metadata["versions"]:
            raise ValueError(f"Version {version} not found")
        
        self.metadata["current_version"] = version
        self._save_metadata()
    
    def delete_version(self, version: str):
        """Delete an index version."""
        if version not in self.metadata["versions"]:
            raise ValueError(f"Version {version} not found")
        
        # Delete files
        version_path = self.base_path / version
        if version_path.exists():
            import shutil
            shutil.rmtree(version_path)
        
        # Update metadata
        del self.metadata["versions"][version]
        
        if self.metadata["current_version"] == version:
            # Set to most recent remaining version
            if self.metadata["versions"]:
                self.metadata["current_version"] = max(self.metadata["versions"].keys())
            else:
                self.metadata["current_version"] = None
        
        self._save_metadata()
    
    def get_version_info(self, version: str) -> Dict[str, Any]:
        """Get information about a specific version."""
        if version not in self.metadata["versions"]:
            raise ValueError(f"Version {version} not found")
        
        return self.metadata["versions"][version]
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"IndexManager("
            f"base_path='{self.base_path}', "
            f"versions={len(self.metadata['versions'])}, "
            f"current='{self.metadata.get('current_version')}')"
        )
