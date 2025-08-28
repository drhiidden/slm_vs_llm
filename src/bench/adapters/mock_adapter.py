"""
Adaptador mock para testing y desarrollo sin APIs reales.
"""

import asyncio
import random
import re
from typing import Dict, Any, Optional
import logging

from .base import ModelAdapter
from ..utils import count_tokens_heuristic, calculate_cost_estimate

logger = logging.getLogger(__name__)


class MockAdapter(ModelAdapter):
    """
    Adaptador mock que simula respuestas de modelos sin usar APIs reales.
    
    Útil para testing, desarrollo y demostraciones.
    """
    
    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, **kwargs)
        
        # Configurar latencia simulada
        self.latency_range_ms = kwargs.get('latency_range_ms', [50, 200])
        
        # Respuestas predefinidas para diferentes tipos de tareas
        self._setup_mock_responses()
    
    def _setup_mock_responses(self):
        """Configura respuestas mock para diferentes tipos de tareas."""
        self.mock_responses = {
            # Exact QA
            "capital": ["París", "Madrid", "Londres", "Berlín", "Roma"],
            "año": ["1914", "1945", "1492", "1789", "1066"],
            "elemento": ["Oxígeno", "Hidrógeno", "Carbono", "Nitrógeno", "Hierro"],
            "autor": ["Miguel de Cervantes", "William Shakespeare", "Gabriel García Márquez"],
            "planeta": ["Júpiter", "Saturno", "Neptuno", "Urano", "Tierra"],
            
            # Math
            "suma": lambda x, y: str(x + y),
            "multiplicación": lambda x, y: str(x * y),
            "resta": lambda x, y: str(x - y),
            "división": lambda x, y: str(x // y),
            "potencia": lambda x, y: str(x ** 2 + y),
            
            # Classification
            "positivo": ["pos", "positive", "bueno"],
            "negativo": ["neg", "negative", "malo"],
            
            # JSON responses
            "json_simple": '{"nombre": "Ana", "edad": 28}',
            "json_complex": '{"ciudad": "Madrid", "pais": "España", "poblacion": 3223000}',
            
            # Reasoning
            "slm_ventajas": [
                "Los SLMs ofrecen menor latencia que los LLMs.",
                "El coste de operación es significativamente menor.",
                "Son suficientes para tareas repetitivas y simples."
            ],
            "privacidad": [
                "La privacidad en IA protege los datos personales.",
                "Genera confianza en los usuarios del sistema.",
                "Es necesaria para cumplir regulaciones legales."
            ]
        }
    
    def _extract_numbers(self, text: str) -> tuple:
        """Extrae números del texto para cálculos matemáticos."""
        numbers = re.findall(r'\d+', text)
        if len(numbers) >= 2:
            return int(numbers[0]), int(numbers[1])
        return 0, 0
    
    def _generate_math_response(self, prompt: str) -> str:
        """Genera respuesta para problemas matemáticos."""
        prompt_lower = prompt.lower()
        
        if "suma" in prompt_lower or "+" in prompt:
            x, y = self._extract_numbers(prompt)
            return self.mock_responses["suma"](x, y)
        elif "multiplicación" in prompt_lower or "×" in prompt or "*" in prompt:
            x, y = self._extract_numbers(prompt)
            return self.mock_responses["multiplicación"](x, y)
        elif "resta" in prompt_lower or "-" in prompt:
            x, y = self._extract_numbers(prompt)
            return self.mock_responses["resta"](x, y)
        elif "división" in prompt_lower or "÷" in prompt or "/" in prompt:
            x, y = self._extract_numbers(prompt)
            return self.mock_responses["división"](x, y)
        elif "²" in prompt or "potencia" in prompt_lower:
            x, y = self._extract_numbers(prompt)
            return self.mock_responses["potencia"](x, y)
        
        return "42"  # Respuesta por defecto
    
    def _generate_qa_response(self, prompt: str) -> str:
        """Genera respuesta para preguntas factuales."""
        prompt_lower = prompt.lower()
        
        if "capital" in prompt_lower:
            return random.choice(self.mock_responses["capital"])
        elif "año" in prompt_lower or "guerra" in prompt_lower:
            return random.choice(self.mock_responses["año"])
        elif "elemento" in prompt_lower or "símbolo" in prompt_lower:
            return random.choice(self.mock_responses["elemento"])
        elif "escribió" in prompt_lower or "autor" in prompt_lower:
            return random.choice(self.mock_responses["autor"])
        elif "planeta" in prompt_lower:
            return random.choice(self.mock_responses["planeta"])
        
        return "Respuesta mock"
    
    def _generate_classification_response(self, prompt: str) -> str:
        """Genera respuesta para clasificación."""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ["encantó", "perfectamente", "excelente"]):
            return random.choice(self.mock_responses["positivo"])
        elif any(word in prompt_lower for word in ["terrible", "recomiendo", "malo"]):
            return random.choice(self.mock_responses["negativo"])
        
        return random.choice(self.mock_responses["positivo"])
    
    def _generate_json_response(self, prompt: str) -> str:
        """Genera respuesta JSON."""
        if "nombre" in prompt and "edad" in prompt:
            return self.mock_responses["json_simple"]
        else:
            return self.mock_responses["json_complex"]
    
    def _generate_reasoning_response(self, prompt: str) -> str:
        """Genera respuesta de razonamiento."""
        prompt_lower = prompt.lower()
        
        if "slm" in prompt_lower and "llm" in prompt_lower:
            return " ".join(self.mock_responses["slm_ventajas"])
        elif "privacidad" in prompt_lower:
            return " ".join(self.mock_responses["privacidad"])
        else:
            return " ".join(self.mock_responses["slm_ventajas"])
    
    async def generate(
        self, 
        prompt: str, 
        temperature: float = 0.2, 
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Genera respuesta mock simulando latencia real.
        """
        # Validar parámetros
        temperature = self.validate_temperature(temperature)
        max_tokens = self.validate_max_tokens(max_tokens)
        
        # Simular latencia
        latency_ms = random.uniform(*self.latency_range_ms)
        await asyncio.sleep(latency_ms / 1000)
        
        # Generar respuesta basada en el tipo de tarea
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ["calcula", "cuánto", "resultado", "número"]):
            response_text = self._generate_math_response(prompt)
        elif any(word in prompt_lower for word in ["capital", "año", "elemento", "escribió", "planeta"]):
            response_text = self._generate_qa_response(prompt)
        elif any(word in prompt_lower for word in ["etiqueta", "pos/neg", "positivo", "negativo"]):
            response_text = self._generate_classification_response(prompt)
        elif "json" in prompt_lower:
            response_text = self._generate_json_response(prompt)
        elif any(word in prompt_lower for word in ["frases", "resume", "explica"]):
            response_text = self._generate_reasoning_response(prompt)
        else:
            response_text = "Respuesta mock generada"
        
        # Limitar respuesta según max_tokens
        if len(response_text) > max_tokens * 4:  # Aproximación: 4 chars por token
            response_text = response_text[:max_tokens * 4]
        
        # Contar tokens
        prompt_tokens = len(prompt) // 4
        completion_tokens = len(response_text) // 4
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
                "latency_ms": latency_ms
            }
        }
