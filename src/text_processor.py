"""
Text Processor Module
Handles text chunking and preprocessing for RAG.
"""

import re
from typing import List


def preprocess_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Raw text content
        
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)
    
    # Normalize line breaks
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to split
        chunk_size: Maximum characters per chunk
        chunk_overlap: Number of overlapping characters between chunks
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    # Preprocess first
    text = preprocess_text(text)
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < text_length:
            # Look for sentence endings near the chunk boundary
            search_start = max(start + chunk_size - 100, start)
            search_end = min(start + chunk_size + 100, text_length)
            search_text = text[search_start:search_end]
            
            # Find the best break point
            best_break = None
            for pattern in ['. ', '! ', '? ', '\n']:
                pos = search_text.rfind(pattern)
                if pos != -1:
                    actual_pos = search_start + pos + len(pattern)
                    if best_break is None or actual_pos > best_break:
                        best_break = actual_pos
            
            if best_break and best_break > start:
                end = best_break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - chunk_overlap
        if start >= text_length:
            break
    
    return chunks


def create_chunks_with_metadata(
    text: str,
    source: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[dict]:
    """
    Create chunks with metadata for vector store.
    
    Args:
        text: Text to split
        source: Source filename or identifier
        chunk_size: Maximum characters per chunk
        chunk_overlap: Number of overlapping characters
        
    Returns:
        List of dicts with 'text' and 'metadata' keys
    """
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    
    return [
        {
            'text': chunk,
            'metadata': {
                'source': source,
                'chunk_index': i,
                'total_chunks': len(chunks)
            }
        }
        for i, chunk in enumerate(chunks)
    ]
