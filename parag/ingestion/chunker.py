"""
Text chunking strategies for document processing.

Chunks large documents into smaller, manageable pieces
for embedding and retrieval.
"""

from typing import List, Dict, Any, Optional
import re
import numpy as np
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import Paradma
try:
    from paradma import learning, Axiom
    PARADMA_AVAILABLE = True
except ImportError:
    PARADMA_AVAILABLE = False
    learning = None
from dataclasses import dataclass
from enum import Enum


class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"


@dataclass
class Chunk:
    """A text chunk with metadata."""
    content: str
    metadata: Dict[str, Any]
    chunk_id: int
    start_char: int
    end_char: int


class TextChunker:
    """
    Split text into chunks for processing.
    
    Supports multiple chunking strategies with configurable parameters.
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE,
    ):
        """
        Initialize TextChunker.
        
        Args:
            chunk_size: Target size of each chunk (in characters)
            chunk_overlap: Number of overlapping characters between chunks
            strategy: Chunking strategy to use
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        
        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        Chunk text according to the configured strategy.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of Chunk objects
        """
        if metadata is None:
            metadata = {}
        
        if self.strategy == ChunkingStrategy.FIXED_SIZE:
            return self._chunk_fixed_size(text, metadata)
        elif self.strategy == ChunkingStrategy.SENTENCE:
            return self._chunk_by_sentence(text, metadata)
        elif self.strategy == ChunkingStrategy.PARAGRAPH:
            return self._chunk_by_paragraph(text, metadata)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")
    
    def _chunk_fixed_size(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunk text into fixed-size pieces with overlap."""
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Don't create chunks that are too small
            if end >= len(text):
                end = len(text)
            else:
                # Try to break at a word boundary
                while end > start and not text[end].isspace():
                    end -= 1
                
                # If we couldn't find a space, just use the original end
                if end == start:
                    end = start + self.chunk_size
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:  # Only add non-empty chunks
                chunk = Chunk(
                    content=chunk_text,
                    metadata={**metadata, "chunk_id": chunk_id},
                    chunk_id=chunk_id,
                    start_char=start,
                    end_char=end,
                )
                chunks.append(chunk)
                chunk_id += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            
            # Prevent infinite loop
            if start >= len(text):
                break
        
        return chunks
    
    def _chunk_by_sentence(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunk text by sentences, combining until reaching chunk_size."""
        # Simple sentence splitting (can be improved with NLTK or spaCy)
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_id = 0
        start_char = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            if current_size + sentence_len > self.chunk_size and current_chunk:
                # Create chunk from current sentences
                chunk_text = ' '.join(current_chunk)
                end_char = start_char + len(chunk_text)
                
                chunk = Chunk(
                    content=chunk_text,
                    metadata={**metadata, "chunk_id": chunk_id},
                    chunk_id=chunk_id,
                    start_char=start_char,
                    end_char=end_char,
                )
                chunks.append(chunk)
                chunk_id += 1
                
                # Add overlap (last sentence)
                if self.chunk_overlap > 0 and current_chunk:
                    current_chunk = [current_chunk[-1]]
                    current_size = len(current_chunk[-1])
                    start_char = end_char - current_size
                else:
                    current_chunk = []
                    current_size = 0
                    start_char = end_char
            
            current_chunk.append(sentence)
            current_size += sentence_len
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk = Chunk(
                content=chunk_text,
                metadata={**metadata, "chunk_id": chunk_id},
                chunk_id=chunk_id,
                start_char=start_char,
                end_char=start_char + len(chunk_text),
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_paragraph(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunk text by paragraphs."""
        paragraphs = text.split('\n\n')
        
        chunks = []
        chunk_id = 0
        char_pos = 0
        
        for para in paragraphs:
            para = para.strip()
            if para:
                chunk = Chunk(
                    content=para,
                    metadata={**metadata, "chunk_id": chunk_id},
                    chunk_id=chunk_id,
                    start_char=char_pos,
                    end_char=char_pos + len(para),
                )
                chunks.append(chunk)
                chunk_id += 1
            
            char_pos += len(para) + 2  # Account for \n\n
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Simple sentence splitter.
        
        For production, consider using NLTK or spaCy for better accuracy.
        """
        sentences = []
        sentence_lengths = []
        current_sentence = []
        
        for char in text:
            current_sentence.append(char)
            
            # Check for sentence endings
            if char in '.!?':
                sentence = ''.join(current_sentence).strip()
                if sentence:
                    sentences.append(sentence)
                    sentence_lengths.append(len(sentence))
                current_sentence = []
        
        # Add any remaining text
        if current_sentence:
            sentence = ''.join(current_sentence).strip()
            if sentence:
                sentences.append(sentence)
                sentence_lengths.append(len(sentence))
        
        # Calculate mean and std of sentence lengths for adaptive chunking
        if sentence_lengths:
            # Use Paradma if available (self-learning!)
            if PARADMA_AVAILABLE and learning:
                try:
                    lengths_axiom = Axiom(sentence_lengths, manifold=learning)
                    mean_result = lengths_axiom.mean()
                    mean_length = mean_result.value if hasattr(mean_result, 'value') else mean_result
                    # std not yet in Paradma autolearner, use NumPy
                    std_length = np.std(sentence_lengths)
                except:
                    mean_length = np.mean(sentence_lengths)
                    std_length = np.std(sentence_lengths)
            else:
                mean_length = np.mean(sentence_lengths)
                std_length = np.std(sentence_lengths)
            
            # Store in instance for potential adaptive use
            self._last_stats = {"mean": mean_length, "std": std_length}
        
        return sentences


def chunk_documents(
    documents: List[Dict[str, Any]],
    chunker: Optional[TextChunker] = None,
) -> List[Dict[str, Any]]:
    """
    Chunk multiple documents.
    
    Args:
        documents: List of documents with 'content' and 'metadata'
        chunker: Optional TextChunker instance
        
    Returns:
        List of chunked documents
    """
    if chunker is None:
        chunker = TextChunker()
    
    chunked_docs = []
    
    for doc in documents:
        content = doc["content"]
        metadata = doc.get("metadata", {})
        
        chunks = chunker.chunk(content, metadata)
        
        for chunk in chunks:
            chunked_docs.append({
                "content": chunk.content,
                "metadata": chunk.metadata,
            })
    
    return chunked_docs
