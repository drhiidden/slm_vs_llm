"""
Adaptador para modelos de OpenAI usando el SDK oficial.
"""

import os
import asyncio
from typing import Dict, Any, Optional
import logging

from openai import AsyncOpenAI
from .base import ModelAdapter
from ..utils import calculate_cost_estimate

logger = logging.getLogger(__name__)


class OpenAIAdapter(ModelAdapter):
    """
    Adaptador para modelos de OpenAI usando el SDK oficial.
    """
    
    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, **kwargs)
        
        # Obtener API key
        api_key_env = kwargs.get('api_key_env', 'OPENAI_API_KEY')
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"API key de OpenAI no encontrada en {api_key_env}")
        
        # Inicializar cliente
        self.client = AsyncOpenAI(api_key=api_key)
        
        logger.info(f"Cliente OpenAI inicializado para modelo: {self.model_id}")
    
    async def generate(
        self, 
        prompt: str, 
        temperature: float = 0.2, 
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Genera texto usando la API de OpenAI.
        """
        # Validar parámetros
        temperature = self.validate_temperature(temperature)
        max_tokens = self.validate_max_tokens(max_tokens)
        
        try:
            # Llamar a la API
            response = await self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=30
            )
            
            # Extraer respuesta
            response_text = response.choices[0].message.content or ""
            
            # Obtener información de tokens
            usage = response.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens
            
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
                    "finish_reason": response.choices[0].finish_reason
                }
            }
            
        except Exception as e:
            logger.error(f"Error en OpenAI API: {e}")
            raise RuntimeError(f"Error al generar texto con OpenAI: {e}")
