"""
Interfaz base para adaptadores de modelos de lenguaje.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ModelAdapter(ABC):
    """
    Interfaz base para adaptadores de modelos de lenguaje.
    
    Todos los adaptadores deben implementar esta interfaz para
    proporcionar una API unificada para diferentes proveedores.
    """
    
    def __init__(self, model_id: str, **kwargs):
        """
        Inicializa el adaptador.
        
        Args:
            model_id: Identificador del modelo
            **kwargs: Parámetros adicionales específicos del proveedor
        """
        self.model_id = model_id
        self.name = kwargs.get('name', model_id)
        self.model_type = kwargs.get('type', 'unknown')
        self.max_tokens = kwargs.get('max_tokens', 2048)
        self.temperature_range = kwargs.get('temperature_range', [0.0, 1.0])
        self.pricing = kwargs.get('pricing')
        
        logger.info(f"Adaptador inicializado: {self.name} ({self.model_type})")
    
    @abstractmethod
    async def generate(
        self, 
        prompt: str, 
        temperature: float = 0.2, 
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Genera texto usando el modelo.
        
        Args:
            prompt: Texto de entrada
            temperature: Temperatura para la generación (0.0-2.0)
            max_tokens: Máximo número de tokens a generar
            
        Returns:
            Diccionario con:
            - text: Texto generado
            - prompt_tokens: Tokens del prompt
            - completion_tokens: Tokens de la respuesta
            - total_tokens: Total de tokens
            - cost_estimate: Coste estimado en USD (opcional)
            - metadata: Metadatos adicionales
        """
        pass
    
    def validate_temperature(self, temperature: float) -> float:
        """Valida y ajusta la temperatura al rango permitido."""
        min_temp, max_temp = self.temperature_range
        if temperature < min_temp:
            logger.warning(f"Temperatura {temperature} ajustada a {min_temp}")
            return min_temp
        elif temperature > max_temp:
            logger.warning(f"Temperatura {temperature} ajustada a {max_temp}")
            return max_temp
        return temperature
    
    def validate_max_tokens(self, max_tokens: Optional[int]) -> int:
        """Valida y ajusta el máximo de tokens."""
        if max_tokens is None:
            return self.max_tokens
        if max_tokens > self.max_tokens:
            logger.warning(f"max_tokens {max_tokens} ajustado a {self.max_tokens}")
            return self.max_tokens
        return max_tokens
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna información del modelo."""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "type": self.model_type,
            "max_tokens": self.max_tokens,
            "temperature_range": self.temperature_range,
            "pricing": self.pricing
        }
    
    def __str__(self) -> str:
        return f"{self.name} ({self.model_type})"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_id='{self.model_id}', name='{self.name}')"
