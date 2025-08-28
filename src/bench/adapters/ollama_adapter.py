"""
Adaptador para modelos de Ollama usando HTTP requests.
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional
import logging

import httpx
from .base import ModelAdapter
from ..utils import count_tokens_heuristic, calculate_cost_estimate

logger = logging.getLogger(__name__)


class OllamaAdapter(ModelAdapter):
    """
    Adaptador para modelos de Ollama usando HTTP requests.
    """
    
    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, **kwargs)
        
        # Obtener URL base
        base_url_env = kwargs.get('base_url_env', 'OLLAMA_BASE_URL')
        self.base_url = os.getenv(base_url_env, 'http://localhost:11434')
        
        # Configurar cliente HTTP
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=30.0
        )
        
        logger.info(f"Cliente Ollama inicializado para modelo: {self.model_id} en {self.base_url}")
    
    async def generate(
        self, 
        prompt: str, 
        temperature: float = 0.2, 
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Genera texto usando la API de Ollama.
        """
        # Validar parámetros
        temperature = self.validate_temperature(temperature)
        max_tokens = self.validate_max_tokens(max_tokens)
        
        try:
            # Preparar payload
            payload = {
                "model": self.model_id,
                "prompt": prompt,
                "temperature": temperature,
                "num_predict": max_tokens,
                "stream": False
            }
            
            # Llamar a la API
            response = await self.client.post("/api/generate", json=payload)
            response.raise_for_status()
            
            # Procesar respuesta
            data = response.json()
            response_text = data.get("response", "")
            
            # Ollama no proporciona conteo de tokens, usar heurística
            token_info = count_tokens_heuristic(prompt + response_text)
            prompt_tokens = count_tokens_heuristic(prompt)["estimated_tokens"]
            completion_tokens = count_tokens_heuristic(response_text)["estimated_tokens"]
            total_tokens = prompt_tokens + completion_tokens
            
            # Calcular coste estimado (Ollama es gratuito localmente)
            cost_estimate = calculate_cost_estimate(
                prompt_tokens, completion_tokens, self.pricing
            )
            
            return {
                "text": response_text,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost_estimate": cost_estimate,
                "metadata": {
                    "model_type": self.model_type,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "eval_count": data.get("eval_count"),
                    "eval_duration": data.get("eval_duration")
                }
            }
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Error HTTP en Ollama API: {e.response.status_code} - {e.response.text}")
            raise RuntimeError(f"Error HTTP en Ollama API: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Error en Ollama API: {e}")
            raise RuntimeError(f"Error al generar texto con Ollama: {e}")
    
    async def close(self):
        """Cierra el cliente HTTP."""
        await self.client.aclose()
    
    def __del__(self):
        """Destructor para asegurar que el cliente se cierre."""
        try:
            asyncio.create_task(self.close())
        except:
            pass
