"""
Utilidades para el benchmarking de modelos.
"""

import time
import random
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import numpy as np

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Establece la semilla para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Semilla establecida: {seed}")


def timeit(func):
    """Decorador para medir el tiempo de ejecución de funciones."""
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        return result, elapsed_ms
    return wrapper


async def timeit_async(func):
    """Decorador para medir el tiempo de ejecución de funciones asíncronas."""
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        return result, elapsed_ms
    return wrapper


def estimate_tokens(text: str) -> int:
    """
    Estimación simple de tokens basada en caracteres.
    Aproximación: 1 token ≈ 4 caracteres para español/inglés.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def count_tokens_heuristic(text: str) -> Dict[str, int]:
    """
    Conteo heurístico de tokens para casos donde el SDK no lo proporciona.
    """
    # Limpiar espacios extra y saltos de línea
    cleaned_text = " ".join(text.split())
    
    # Estimación basada en palabras y caracteres
    words = len(cleaned_text.split())
    chars = len(cleaned_text)
    
    # Aproximación: 1 token ≈ 0.75 palabras o 4 caracteres
    estimated_tokens = min(words * 1.33, chars / 4)
    
    return {
        "estimated_tokens": int(estimated_tokens),
        "words": words,
        "characters": chars
    }


def load_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Carga datos desde un archivo JSONL."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: Union[str, Path]) -> None:
    """Guarda datos en formato JSONL."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Carga configuración desde archivo YAML."""
    import yaml
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Guarda datos en formato YAML."""
    import yaml
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def create_cache_key(model_id: str, prompt: str, temperature: float = 0.2) -> str:
    """Crea una clave única para el caché basada en el modelo y prompt."""
    content = f"{model_id}:{prompt}:{temperature}"
    return hashlib.md5(content.encode()).hexdigest()


def ensure_dir(path: Union[str, Path]) -> Path:
    """Asegura que el directorio existe, creándolo si es necesario."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_duration(ms: float) -> str:
    """Formatea duración en milisegundos de forma legible."""
    if ms < 1000:
        return f"{ms:.1f}ms"
    elif ms < 60000:
        return f"{ms/1000:.2f}s"
    else:
        return f"{ms/60000:.1f}m"


def format_cost(cost_usd: Optional[float]) -> str:
    """Formatea coste en USD de forma legible."""
    if cost_usd is None:
        return "N/A"
    if cost_usd < 0.001:
        return f"${cost_usd*1000:.3f}m"
    elif cost_usd < 1:
        return f"${cost_usd:.4f}"
    else:
        return f"${cost_usd:.2f}"


def calculate_cost_estimate(
    prompt_tokens: int,
    completion_tokens: int,
    pricing: Optional[Dict[str, float]]
) -> Optional[float]:
    """
    Calcula el coste estimado basado en tokens y precios.
    
    Args:
        prompt_tokens: Número de tokens de entrada
        completion_tokens: Número de tokens de salida
        pricing: Diccionario con precios por 1K tokens {"input": float, "output": float}
    
    Returns:
        Coste estimado en USD o None si no hay pricing disponible
    """
    if not pricing:
        return None
    
    input_cost = (prompt_tokens / 1000) * pricing.get("input", 0)
    output_cost = (completion_tokens / 1000) * pricing.get("output", 0)
    
    return input_cost + output_cost
