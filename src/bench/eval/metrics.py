"""
Métricas de evaluación para diferentes tipos de tareas.
"""

import re
import json
from typing import Dict, Any, List, Union, Optional
import logging

from jsonschema import validate, ValidationError
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)


def exact_match(prediction: str, gold: str) -> float:
    """
    Calcula si la predicción coincide exactamente con la respuesta de oro.
    
    Args:
        prediction: Texto predicho
        gold: Texto de referencia
        
    Returns:
        1.0 si coinciden exactamente, 0.0 en caso contrario
    """
    # Normalizar: convertir a minúsculas y eliminar espacios extra
    pred_clean = prediction.strip().lower()
    gold_clean = gold.strip().lower()
    
    return 1.0 if pred_clean == gold_clean else 0.0


def rouge_l(prediction: str, gold: str) -> float:
    """
    Calcula la métrica ROUGE-L (Longest Common Subsequence).
    
    Args:
        prediction: Texto predicho
        gold: Texto de referencia
        
    Returns:
        Score ROUGE-L entre 0.0 y 1.0
    """
    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(gold, prediction)
        return scores['rougeL'].fmeasure
    except Exception as e:
        logger.warning(f"Error calculando ROUGE-L: {e}")
        return 0.0


def math_accuracy(prediction: str, gold: str, tolerance: float = 0.01) -> float:
    """
    Evalúa la precisión de cálculos matemáticos.
    
    Args:
        prediction: Respuesta predicha
        gold: Respuesta correcta
        tolerance: Tolerancia para comparaciones numéricas
        
    Returns:
        1.0 si el resultado es correcto, 0.0 en caso contrario
    """
    try:
        # Extraer números de ambas respuestas
        pred_numbers = re.findall(r'-?\d+\.?\d*', prediction)
        gold_numbers = re.findall(r'-?\d+\.?\d*', gold)
        
        if not pred_numbers or not gold_numbers:
            return exact_match(prediction, gold)
        
        # Comparar el primer número encontrado
        pred_num = float(pred_numbers[0])
        gold_num = float(gold_numbers[0])
        
        # Verificar si están dentro de la tolerancia
        if abs(pred_num - gold_num) <= tolerance:
            return 1.0
        else:
            return 0.0
            
    except (ValueError, IndexError) as e:
        logger.warning(f"Error en math_accuracy: {e}")
        return exact_match(prediction, gold)


def json_schema_valid(prediction: str, schema: Dict[str, Any]) -> float:
    """
    Valida si la predicción es un JSON válido según el esquema.
    
    Args:
        prediction: Texto predicho (debe ser JSON válido)
        schema: Esquema JSON para validación
        
    Returns:
        1.0 si es válido, 0.0 en caso contrario
    """
    try:
        # Intentar parsear como JSON
        parsed_json = json.loads(prediction)
        
        # Validar contra el esquema
        validate(instance=parsed_json, schema=schema)
        return 1.0
        
    except json.JSONDecodeError as e:
        logger.warning(f"JSON inválido: {e}")
        return 0.0
    except ValidationError as e:
        logger.warning(f"Esquema inválido: {e}")
        return 0.0
    except Exception as e:
        logger.warning(f"Error en validación JSON: {e}")
        return 0.0


def field_accuracy(prediction: str, gold: str, schema: Dict[str, Any]) -> float:
    """
    Calcula la precisión de campos específicos en respuestas JSON.
    
    Args:
        prediction: JSON predicho
        gold: JSON de referencia
        schema: Esquema para identificar campos requeridos
        
    Returns:
        Proporción de campos correctos entre 0.0 y 1.0
    """
    try:
        pred_json = json.loads(prediction)
        gold_json = json.loads(gold)
        
        # Obtener campos requeridos del esquema
        required_fields = schema.get("required", [])
        if not required_fields:
            # Si no hay campos requeridos, usar propiedades
            required_fields = list(schema.get("properties", {}).keys())
        
        if not required_fields:
            return exact_match(prediction, gold)
        
        correct_fields = 0
        total_fields = len(required_fields)
        
        for field in required_fields:
            if field in pred_json and field in gold_json:
                if pred_json[field] == gold_json[field]:
                    correct_fields += 1
        
        return correct_fields / total_fields if total_fields > 0 else 0.0
        
    except Exception as e:
        logger.warning(f"Error en field_accuracy: {e}")
        return 0.0


def classification_metrics(prediction: str, gold: str) -> Dict[str, float]:
    """
    Calcula métricas de clasificación (accuracy y F1).
    
    Args:
        prediction: Etiqueta predicha
        gold: Etiqueta correcta
        
    Returns:
        Diccionario con accuracy y f1_score
    """
    # Normalizar etiquetas
    pred_clean = prediction.strip().lower()
    gold_clean = gold.strip().lower()
    
    # Mapear variaciones comunes
    label_mapping = {
        'pos': 'positive',
        'neg': 'negative',
        'bueno': 'positive',
        'malo': 'negative'
    }
    
    pred_clean = label_mapping.get(pred_clean, pred_clean)
    gold_clean = label_mapping.get(gold_clean, gold_clean)
    
    # Calcular accuracy
    accuracy = 1.0 if pred_clean == gold_clean else 0.0
    
    # Calcular F1 (simplificado para clasificación binaria)
    if pred_clean == gold_clean:
        f1_score = 1.0
    else:
        f1_score = 0.0
    
    return {
        "accuracy": accuracy,
        "f1_score": f1_score
    }


def length_control(prediction: str, target_sentences: int = 3) -> float:
    """
    Evalúa el control de longitud de la respuesta.
    
    Args:
        prediction: Texto predicho
        target_sentences: Número objetivo de frases
        
    Returns:
        Score entre 0.0 y 1.0 basado en la proximidad al objetivo
    """
    try:
        # Contar frases (terminadas en . ! ?)
        sentences = re.split(r'[.!?]+', prediction)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        actual_sentences = len(sentences)
        
        # Calcular penalización por desviación
        deviation = abs(actual_sentences - target_sentences)
        
        if deviation == 0:
            return 1.0
        elif deviation == 1:
            return 0.8
        elif deviation == 2:
            return 0.6
        else:
            return max(0.0, 1.0 - (deviation * 0.2))
            
    except Exception as e:
        logger.warning(f"Error en length_control: {e}")
        return 0.5


def f1_score(prediction: str, gold: str) -> float:
    """
    Calcula el F1-score para clasificación.
    
    Args:
        prediction: Etiqueta predicha
        gold: Etiqueta correcta
        
    Returns:
        F1-score entre 0.0 y 1.0
    """
    metrics = classification_metrics(prediction, gold)
    return metrics["f1_score"]
