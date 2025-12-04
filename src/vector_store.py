"""
Vector Store Module
ChromaDB wrapper for document storage and similarity search.
"""

import os
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings

from .embeddings import get_embedding_generator


class VectorStore:
    """ChromaDB-based vector store for book chunks."""
    
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory for persistent storage
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.embedding_generator = get_embedding_generator()
        self._collection = None
        self._current_book = None
    
    def get_or_create_collection(self, book_name: str) -> chromadb.Collection:
        """
        Get or create a collection for a book.
        
        Args:
            book_name: Name of the book (used as collection name)
            
        Returns:
            ChromaDB collection
        """
        # Sanitize collection name
        safe_name = "".join(c if c.isalnum() else "_" for c in book_name)
        safe_name = safe_name[:50]  # ChromaDB has name length limits
        
        self._current_book = safe_name
        self._collection = self.client.get_or_create_collection(
            name=safe_name,
            metadata={"hnsw:space": "cosine"}
        )
        return self._collection
    
    def add_documents(
        self,
        chunks: List[Dict],
        book_name: str
    ) -> int:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of chunk dicts with 'text' and 'metadata'
            book_name: Name of the book
            
        Returns:
            Number of chunks added
        """
        collection = self.get_or_create_collection(book_name)
        
        # Clear existing documents for this book
        existing = collection.count()
        if existing > 0:
            collection.delete(where={})
        
        # Generate embeddings
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_generator.embed_texts(texts)
        
        # Prepare data for ChromaDB
        ids = [f"{book_name}_{i}" for i in range(len(chunks))]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        # Add in batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            end = min(i + batch_size, len(chunks))
            collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=texts[i:end],
                metadatas=metadatas[i:end]
            )
        
        return len(chunks)
    
    def similarity_search(
        self,
        query: str,
        book_name: str,
        n_results: int = 5
    ) -> List[Dict]:
        """
        Find similar chunks for a query.
        
        Args:
            query: Search query
            book_name: Name of the book to search in
            n_results: Number of results to return
            
        Returns:
            List of matching chunks with scores
        """
        collection = self.get_or_create_collection(book_name)
        
        if collection.count() == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_generator.embed_text(query)
        
        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, collection.count())
        )
        
        # Format results
        formatted = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                formatted.append({
                    'text': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else 0
                })
        
        return formatted
    
    def list_books(self) -> List[str]:
        """List all books in the vector store."""
        collections = self.client.list_collections()
        return [c.name for c in collections]
    
    def delete_book(self, book_name: str) -> bool:
        """Delete a book from the vector store."""
        safe_name = "".join(c if c.isalnum() else "_" for c in book_name)
        safe_name = safe_name[:50]
        
        try:
            self.client.delete_collection(safe_name)
            return True
        except Exception:
            return False
