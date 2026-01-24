"""
Metadata extraction and management for documents.

Handles automatic and custom metadata extraction from documents.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import hashlib


class MetadataExtractor:
    """Extract and manage document metadata."""
    
    def __init__(self, custom_fields: Optional[List[str]] = None):
        """
        Initialize MetadataExtractor().
        
        Args:
            custom_fields: Optional list of custom metadata fields to track
        """
        self.custom_fields = custom_fields or []
    
    def extract(
        self,
        file_path: Optional[str] = None,
        content: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extract metadata from a document.
        
        Args:
            file_path: Optional path to the source file
            content: Optional content string for content-based metadata
            custom_metadata: Optional custom metadata to include
            
        Returns:
            Metadata dictionary
        """
        metadata = {}
        
        # Extract file-based metadata
        if file_path:
            metadata.update(self._extract_file_metadata(file_path))
        
        # Extract content-based metadata
        if content:
            metadata.update(self._extract_content_metadata(content))
        
        # Add custom metadata
        if custom_metadata:
            metadata.update(custom_metadata)
        
        # Add extraction timestamp
        metadata["extracted_at"] = datetime.now().isoformat()
        
        # Validate custom fields
        for field in self.custom_fields:
            if field not in metadata:
                metadata[field] = None
        
        return metadata
    
    def _extract_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from file path."""
        path = Path(file_path)
        
        metadata = {
            "source": str(path.absolute()),
            "filename": path.name,
            "file_type": path.suffix,
            "file_stem": path.stem,
        }
        
        # Get file stats if file exists
        if path.exists():
            stats = path.stat()
            metadata.update({
                "file_size": stats.st_size,
                "created_at": datetime.fromtimestamp(stats.st_ctime).isoformat(),
                "modified_at": datetime.fromtimestamp(stats.st_mtime).isoformat(),
            })
        
        return metadata
    
    def _extract_content_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from content."""
        metadata = {
            "content_length": len(content),
            "word_count": len(content.split()),
            "line_count": content.count('\n') + 1,
            "content_hash": self._hash_content(content),
        }
        
        return metadata
    
    def _hash_content(self, content: str) -> str:
        """Generate hash of content for deduplication."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def validate(self, metadata: Dict[str, Any], required_fields: Optional[List[str]] = None) -> bool:
        """
        Validate metadata contains required fields.
        
        Args:
            metadata: Metadata dictionary to validate
            required_fields: Optional list of required fields
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        if required_fields is None:
            required_fields = ["source"]
        
        missing_fields = [field for field in required_fields if field not in metadata]
        
        if missing_fields:
            raise ValueError(f"Missing required metadata fields: {missing_fields}")
        
        return True
    
    def merge_metadata(self, *metadata_dicts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple metadata dictionaries.
        
        Later dictionaries take precedence over earlier ones.
        """
        merged = {}
        for metadata in metadata_dicts:
            merged.update(metadata)
        return merged


def add_tags(metadata: Dict[str, Any], tags: List[str]) -> Dict[str, Any]:
    """
    Add tags to metadata.
    
    Args:
        metadata: Metadata dictionary
        tags: List of tags to add
        
    Returns:
        Updated metadata
    """
    if "tags" not in metadata:
        metadata["tags"] = []
    
    for tag in tags:
        if tag not in metadata["tags"]:
            metadata["tags"].append(tag)
    
    return metadata


def filter_metadata(metadata: Dict[str, Any], keep_fields: List[str]) -> Dict[str, Any]:
    """
    Filter metadata to keep only specified fields.
    
    Args:
        metadata: Metadata dictionary
        keep_fields: Fields to keep
        
    Returns:
        Filtered metadata
    """
    return {k: v for k, v in metadata.items() if k in keep_fields}
