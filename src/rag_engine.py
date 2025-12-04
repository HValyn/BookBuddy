"""
RAG Engine Module
Retrieval-Augmented Generation logic for book discussions.
"""

from typing import List, Dict, Optional, Generator

from .vector_store import VectorStore
from .ollama_client import get_ollama_client


# System prompt designed to prevent spoilers and stay grounded in the text
SYSTEM_PROMPT = """You are a helpful book discussion assistant. Your role is to help users understand and discuss the book they are reading.

IMPORTANT RULES:
1. ONLY use information from the provided context passages below. Do not use your general knowledge about the book.
2. If the context doesn't contain enough information to answer the question, say so clearly.
3. NEVER reveal plot points, character fates, or events that are NOT in the provided context. This prevents spoilers.
4. When discussing characters, only mention what is known from the provided passages.
5. Be engaging and help the reader explore themes, characters, and ideas from the text.
6. If asked about something not in the context, politely explain that you can only discuss what's in the current reading.

CONTEXT FROM THE BOOK:
{context}

Remember: Stay grounded in the provided context. Do not spoil anything beyond what's shown above."""


class RAGEngine:
    """RAG engine for book-grounded conversations."""
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        ollama_url: str = "http://localhost:11434"
    ):
        """
        Initialize the RAG engine.
        
        Args:
            vector_store: VectorStore instance (creates new if None)
            ollama_url: Ollama API URL
        """
        self.vector_store = vector_store or VectorStore()
        self.ollama_client = get_ollama_client(ollama_url)
        self.current_book: Optional[str] = None
    
    def set_book(self, book_name: str) -> None:
        """Set the current book for queries."""
        self.current_book = book_name
    
    def _build_context(self, query: str, n_chunks: int = 5) -> str:
        """
        Retrieve relevant chunks and build context string.
        
        Args:
            query: User's question
            n_chunks: Number of chunks to retrieve
            
        Returns:
            Formatted context string
        """
        if not self.current_book:
            return ""
        
        chunks = self.vector_store.similarity_search(
            query=query,
            book_name=self.current_book,
            n_results=n_chunks
        )
        
        if not chunks:
            return "No relevant passages found in the book."
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[Passage {i}]\n{chunk['text']}")
        
        return "\n\n".join(context_parts)
    
    def _build_messages(
        self,
        query: str,
        chat_history: List[Dict[str, str]],
        context: str
    ) -> List[Dict[str, str]]:
        """Build the message list for the LLM."""
        system_message = SYSTEM_PROMPT.format(context=context)
        
        messages = [{"role": "system", "content": system_message}]
        
        # Add chat history (limit to last 10 exchanges to manage context)
        history_limit = 20  # 10 exchanges = 20 messages
        recent_history = chat_history[-history_limit:] if chat_history else []
        messages.extend(recent_history)
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        return messages
    
    def query(
        self,
        question: str,
        model: str,
        chat_history: List[Dict[str, str]] = None,
        n_chunks: int = 5
    ) -> str:
        """
        Answer a question about the book using RAG.
        
        Args:
            question: User's question
            model: Ollama model to use
            chat_history: Previous conversation messages
            n_chunks: Number of context chunks to retrieve
            
        Returns:
            Model's response
        """
        if not self.current_book:
            return "Please upload a book first before asking questions."
        
        # Retrieve relevant context
        context = self._build_context(question, n_chunks)
        
        # Build messages
        messages = self._build_messages(
            query=question,
            chat_history=chat_history or [],
            context=context
        )
        
        # Get response from Ollama
        response = self.ollama_client.chat(model=model, messages=messages)
        
        return response
    
    def query_stream(
        self,
        question: str,
        model: str,
        chat_history: List[Dict[str, str]] = None,
        n_chunks: int = 5
    ) -> Generator[str, None, None]:
        """
        Stream answer to a question about the book using RAG.
        
        Args:
            question: User's question
            model: Ollama model to use
            chat_history: Previous conversation messages
            n_chunks: Number of context chunks to retrieve
            
        Yields:
            Response text chunks
        """
        if not self.current_book:
            yield "Please upload a book first before asking questions."
            return
        
        # Retrieve relevant context
        context = self._build_context(question, n_chunks)
        
        # Build messages
        messages = self._build_messages(
            query=question,
            chat_history=chat_history or [],
            context=context
        )
        
        # Stream response from Ollama
        for chunk in self.ollama_client.chat_stream(model=model, messages=messages):
            yield chunk
    
    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models."""
        return self.ollama_client.list_models()
    
    def is_ollama_available(self) -> bool:
        """Check if Ollama is running."""
        return self.ollama_client.is_available()
