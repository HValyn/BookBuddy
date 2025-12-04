"""
Ollama Client Module
Wrapper for Ollama API communication.
"""

import json
from typing import List, Dict, Optional, Generator
import requests


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama client.
        
        Args:
            base_url: Ollama API base URL
        """
        self.base_url = base_url.rstrip('/')
    
    def is_available(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def list_models(self) -> List[str]:
        """
        List available models.
        
        Returns:
            List of model names
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
        except requests.exceptions.RequestException:
            return []
    
    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False
    ) -> str:
        """
        Send a chat request to Ollama.
        
        Args:
            model: Model name to use
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream the response
            
        Returns:
            Assistant's response text
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            data = response.json()
            return data.get('message', {}).get('content', '')
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to communicate with Ollama: {e}")
    
    def chat_stream(
        self,
        model: str,
        messages: List[Dict[str, str]]
    ) -> Generator[str, None, None]:
        """
        Stream a chat response from Ollama.
        
        Args:
            model: Model name to use
            messages: List of message dicts with 'role' and 'content'
            
        Yields:
            Response text chunks
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": True
        }
        
        try:
            with requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=120
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        content = data.get('message', {}).get('content', '')
                        if content:
                            yield content
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to communicate with Ollama: {e}")


# Singleton instance
_ollama_client = None


def get_ollama_client(base_url: str = "http://localhost:11434") -> OllamaClient:
    """Get or create the Ollama client singleton."""
    global _ollama_client
    if _ollama_client is None or _ollama_client.base_url != base_url:
        _ollama_client = OllamaClient(base_url)
    return _ollama_client
