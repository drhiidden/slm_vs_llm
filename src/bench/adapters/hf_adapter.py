"""
Adaptador para modelos de Hugging Face Inference Endpoints.
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


class HuggingFaceAdapter(ModelAdapter):
    """
    Adaptador para modelos de Hugging Face Inference Endpoints.
    """
    
    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, **kwargs)
        
        # Obtener API key y endpoint
        api_key_env = kwargs.get('api_key_env', 'HUGGINGFACE_API_KEY')
        endpoint_env = kwargs.get('endpoint_env', 'HUGGINGFACE_ENDPOINT_URL')
        
        self.api_key = os.getenv(api_key_env)
        self.endpoint_url = os.getenv(endpoint_env)
        
        if not self.api_key:
            raise ValueError(f"API key de Hugging Face no encontrada en {api_key_env}")
        
        if not self.endpoint_url:
            raise ValueError(f"Endpoint URL de Hugging Face no encontrada en {endpoint_env}")
        
        # Configurar cliente HTTP
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        
        logger.info(f"Cliente Hugging Face inicializado para modelo: {self.model_id}")
    
    async def generate(
        self, 
        prompt: str, 
        temperature: float = 0.2, 
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Genera texto usando Hugging Face Inference Endpoints.
        """
        # Validar parámetros
        temperature = self.validate_temperature(temperature)
        max_tokens = self.validate_max_tokens(max_tokens)
        
        try:
            # Preparar payload según el formato esperado por HF
            payload = {
                "inputs": prompt,
                "parameters": {
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                    "do_sample": temperature > 0,
                    "return_full_text": False
                }
            }
            
            # Llamar a la API
            response = await self.client.post(self.endpoint_url, json=payload)
            response.raise_for_status()
            
            # Procesar respuesta
            data = response.json()
            
            # HF puede devolver diferentes formatos de respuesta
            if isinstance(data, list) and len(data) > 0:
                response_text = data[0].get("generated_text", "")
                # Si return_full_text=False, solo devuelve la nueva parte
                if not payload["parameters"]["return_full_text"]:
                    response_text = response_text[len(prompt):]
            elif isinstance(data, dict):
                response_text = data.get("generated_text", "")
            else:
                response_text = str(data)
            
            # HF no proporciona conteo de tokens, usar heurística
            prompt_tokens = count_tokens_heuristic(prompt)["estimated_tokens"]
            completion_tokens = count_tokens_heuristic(response_text)["estimated_tokens"]
            total_tokens = prompt_tokens + completion_tokens
            
            # Calcular coste estimado
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
                    "endpoint": self.endpoint_url
                }
            }
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Error HTTP en Hugging Face API: {e.response.status_code} - {e.response.text}")
            raise RuntimeError(f"Error HTTP en Hugging Face API: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Error en Hugging Face API: {e}")
            raise RuntimeError(f"Error al generar texto con Hugging Face: {e}")
    
    async def close(self):
        """Cierra el cliente HTTP."""
        await self.client.aclose()
    
    def __del__(self):
        """Destructor para asegurar que el cliente se cierre."""
        try:
            asyncio.create_task(self.close())
        except:
            pass
