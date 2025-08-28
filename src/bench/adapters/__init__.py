"""
Adaptadores para diferentes proveedores de modelos de lenguaje.
"""

from .base import ModelAdapter
from .openai_adapter import OpenAIAdapter
from .hf_adapter import HuggingFaceAdapter
from .ollama_adapter import OllamaAdapter
from .mock_adapter import MockAdapter

__all__ = [
    "ModelAdapter",
    "OpenAIAdapter", 
    "HuggingFaceAdapter",
    "OllamaAdapter",
    "MockAdapter"
]
