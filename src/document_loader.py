"""
Document Loader Module
Handles loading and extracting text from PDF, EPUB, and MOBI files.
"""

import os
from typing import Optional
from pypdf import PdfReader
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup


def load_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(file_path)
    text_parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_parts.append(text)
    return "\n\n".join(text_parts)


def load_epub(file_path: str) -> str:
    """Extract text from an EPUB file."""
    book = epub.read_epub(file_path)
    text_parts = []
    
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            content = item.get_content().decode('utf-8', errors='ignore')
            soup = BeautifulSoup(content, 'html.parser')
            text = soup.get_text(separator='\n')
            if text.strip():
                text_parts.append(text.strip())
    
    return "\n\n".join(text_parts)


def load_mobi(file_path: str) -> str:
    """Extract text from a MOBI file."""
    try:
        import mobi
        tempdir, extracted_file = mobi.extract(file_path)
        
        # Find the HTML file in extracted content
        html_file = None
        for root, dirs, files in os.walk(tempdir):
            for f in files:
                if f.endswith('.html') or f.endswith('.htm'):
                    html_file = os.path.join(root, f)
                    break
            if html_file:
                break
        
        if html_file:
            with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                return soup.get_text(separator='\n')
        
        return ""
    except Exception as e:
        raise ValueError(f"Failed to load MOBI file: {e}")


def load_document(file_path: str) -> str:
    """
    Load a document based on its file extension.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Extracted text content
        
    Raises:
        ValueError: If file type is not supported
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.pdf':
        return load_pdf(file_path)
    elif ext == '.epub':
        return load_epub(file_path)
    elif ext == '.mobi':
        return load_mobi(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def load_from_bytes(file_bytes: bytes, filename: str, temp_dir: str = "/tmp") -> str:
    """
    Load a document from bytes (for Streamlit file uploads).
    
    Args:
        file_bytes: The file content as bytes
        filename: Original filename (used to determine type)
        temp_dir: Directory to temporarily save the file
        
    Returns:
        Extracted text content
    """
    temp_path = os.path.join(temp_dir, filename)
    
    try:
        with open(temp_path, 'wb') as f:
            f.write(file_bytes)
        return load_document(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
