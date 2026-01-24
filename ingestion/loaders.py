"""
Document loaders for various file formats.

Provides a unified interface for loading documents from different sources.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
import io


class DocumentLoader(ABC):
    """Base class for document loaders."""
    
    @abstractmethod
    def load(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load a document and return a list of document chunks.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of dictionaries containing:
                - content: The text content
                - metadata: Document metadata
        """
        pass
    
    def _extract_basic_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract basic metadata from file path."""
        path = Path(file_path)
        return {
            "source": str(path.absolute()),
            "filename": path.name,
            "file_type": path.suffix,
        }


class TextLoader(DocumentLoader):
    """Load plain text files."""
    
    def __init__(self, encoding: str = "utf-8"):
        """
        Initialize TextLoader.
        
        Args:
            encoding: Text file encoding
        """
        self.encoding = encoding
    
    def load(self, file_path: str) -> List[Dict[str, Any]]:
        """Load a text file."""
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                content = f.read()
            
            metadata = self._extract_basic_metadata(file_path)
            metadata["loader"] = "TextLoader"
            
            return [{
                "content": content,
                "metadata": metadata,
            }]
        except Exception as e:
            raise RuntimeError(f"Failed to load text file {file_path}: {e}")


class PDFLoader(DocumentLoader):
    """Load PDF files."""
    
    def __init__(self):
        """Initialize PDFLoader."""
        try:
            import PyPDF2
            self.PyPDF2 = PyPDF2
        except ImportError:
            raise ImportError(
                "PyPDF2 is required for PDFLoader. "
                "Install with: pip install PyPDF2"
            )
    
    def load(self, file_path: str) -> List[Dict[str, Any]]:
        """Load a PDF file."""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = self.PyPDF2.PdfReader(file)
                
                documents = []
                for page_num, page in enumerate(pdf_reader.pages):
                    content = page.extract_text()
                    
                    if content.strip():  # Only add non-empty pages
                        metadata = self._extract_basic_metadata(file_path)
                        metadata.update({
                            "loader": "PDFLoader",
                            "page": page_num + 1,
                            "total_pages": len(pdf_reader.pages),
                        })
                        
                        documents.append({
                            "content": content,
                            "metadata": metadata,
                        })
                
                return documents
        except Exception as e:
            raise RuntimeError(f"Failed to load PDF file {file_path}: {e}")


class MarkdownLoader(DocumentLoader):
    """Load Markdown files with metadata extraction."""
    
    def __init__(self, extract_frontmatter: bool = True):
        """
        Initialize MarkdownLoader.
        
        Args:
            extract_frontmatter: Whether to extract YAML frontmatter
        """
        self.extract_frontmatter = extract_frontmatter
    
    def load(self, file_path: str) -> List[Dict[str, Any]]:
        """Load a Markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            metadata = self._extract_basic_metadata(file_path)
            metadata["loader"] = "MarkdownLoader"
            
            # Extract frontmatter if present
            if self.extract_frontmatter and content.startswith('---'):
                frontmatter, content = self._extract_yaml_frontmatter(content)
                metadata.update(frontmatter)
            
            # Extract headers for additional metadata
            headers = self._extract_headers(content)
            if headers:
                metadata["headers"] = headers
                metadata["title"] = headers[0] if headers else None
            
            return [{
                "content": content,
                "metadata": metadata,
            }]
        except Exception as e:
            raise RuntimeError(f"Failed to load Markdown file {file_path}: {e}")
    
    def _extract_yaml_frontmatter(self, content: str) -> tuple:
        """Extract YAML frontmatter from markdown content."""
        lines = content.split('\n')
        
        if lines[0].strip() != '---':
            return {}, content
        
        # Find closing ---
        frontmatter_lines = []
        content_start = 1
        
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == '---':
                content_start = i + 1
                break
            frontmatter_lines.append(line)
        
        # Parse YAML (simple key: value parsing)
        frontmatter = {}
        for line in frontmatter_lines:
            if ':' in line:
                key, value = line.split(':', 1)
                frontmatter[key.strip()] = value.strip()
        
        remaining_content = '\n'.join(lines[content_start:])
        return frontmatter, remaining_content
    
    def _extract_headers(self, content: str) -> List[str]:
        """Extract markdown headers."""
        headers = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('#'):
                # Remove # symbols and whitespace
                header = line.lstrip('#').strip()
                if header:
                    headers.append(header)
        return headers


def load_document(file_path: str, loader: Optional[DocumentLoader] = None) -> List[Dict[str, Any]]:
    """
    Automatically load a document using the appropriate loader.
    
    Args:
        file_path: Path to the document
        loader: Optional specific loader to use
        
    Returns:
        List of loaded document chunks
    """
    if loader:
        return loader.load(file_path)
    
    # Auto-detect loader based on file extension
    path = Path(file_path)
    extension = path.suffix.lower()
    
    if extension == '.pdf':
        return PDFLoader().load(file_path)
    elif extension in ['.md', '.markdown']:
        return MarkdownLoader().load(file_path)
    elif extension in ['.txt', '.text']:
        return TextLoader().load(file_path)
    else:
        # Default to text loader
        return TextLoader().load(file_path)
