"""
Grader que orquesta las métricas de evaluación por tipo de tarea.
"""

import time
from typing import Dict, Any, List, Optional
import logging

from .metrics import (
    exact_match,
    rouge_l,
    math_accuracy,
    json_schema_valid,
    field_accuracy,
    classification_metrics,
    length_control
)

logger = logging.getLogger(__name__)


class Grader:
    """
    Grader que evalúa respuestas de modelos según el tipo de tarea.
    """
    
    def __init__(self):
        """Inicializa el grader."""
        self.metric_mappings = {
            "exact_qa": {
                "primary": exact_match,
                "secondary": [rouge_l]
            },
            "math": {
                "primary": math_accuracy,
                "secondary": [exact_match]
            },
            "json_toolcall": {
                "primary": json_schema_valid,
                "secondary": [field_accuracy]
            },
            "classification": {
                "primary": lambda pred, gold: classification_metrics(pred, gold)["accuracy"],
                "secondary": [lambda pred, gold: classification_metrics(pred, gold)["f1_score"]]
            },
            "reason_short": {
                "primary": rouge_l,
                "secondary": [length_control]
            }
        }
    
    def grade(
        self,
        task_type: str,
        prediction: str,
        gold: str,
        schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evalúa una predicción según el tipo de tarea.
        
        Args:
            task_type: Tipo de tarea (exact_qa, math, json_toolcall, etc.)
            prediction: Respuesta predicha
            gold: Respuesta de referencia
            schema: Esquema JSON (solo para json_toolcall)
            **kwargs: Parámetros adicionales para métricas específicas
            
        Returns:
            Diccionario con métricas calculadas
        """
        if task_type not in self.metric_mappings:
            raise ValueError(f"Tipo de tarea no soportado: {task_type}")
        
        mapping = self.metric_mappings[task_type]
        metrics = {}
        
        # Métrica primaria
        try:
            if task_type == "json_toolcall" and schema:
                primary_score = mapping["primary"](prediction, schema)
            elif task_type == "reason_short":
                target_sentences = kwargs.get("target_sentences", 3)
                primary_score = mapping["primary"](prediction, gold)
            else:
                primary_score = mapping["primary"](prediction, gold)
            
            metrics["primary_score"] = primary_score
        except Exception as e:
            logger.warning(f"Error en métrica primaria para {task_type}: {e}")
            metrics["primary_score"] = 0.0
        
        # Métricas secundarias
        secondary_scores = {}
        for i, metric_func in enumerate(mapping["secondary"]):
            try:
                if task_type == "json_toolcall" and schema:
                    score = metric_func(prediction, gold, schema)
                elif task_type == "reason_short":
                    target_sentences = kwargs.get("target_sentences", 3)
                    score = metric_func(prediction, target_sentences)
                else:
                    score = metric_func(prediction, gold)
                
                secondary_scores[f"secondary_{i+1}"] = score
            except Exception as e:
                logger.warning(f"Error en métrica secundaria {i+1} para {task_type}: {e}")
                secondary_scores[f"secondary_{i+1}"] = 0.0
        
        metrics.update(secondary_scores)
        
        # Agregar información adicional
        metrics["task_type"] = task_type
        metrics["prediction"] = prediction
        metrics["gold"] = gold
        
        return metrics
    
    def grade_batch(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Evalúa un lote de resultados.
        
        Args:
            results: Lista de resultados con task_type, prediction, gold, etc.
            
        Returns:
            Lista de resultados con métricas agregadas
        """
        graded_results = []
        
        for result in results:
            try:
                task_type = result.get("task_type")
                prediction = result.get("prediction", "")
                gold = result.get("gold", "")
                schema = result.get("schema")
                
                if not task_type:
                    logger.warning("Resultado sin task_type, saltando...")
                    continue
                
                # Evaluar
                metrics = self.grade(
                    task_type=task_type,
                    prediction=prediction,
                    gold=gold,
                    schema=schema
                )
                
                # Combinar con resultado original
                graded_result = {**result, **metrics}
                graded_results.append(graded_result)
                
            except Exception as e:
                logger.error(f"Error evaluando resultado: {e}")
                # Agregar resultado con métricas por defecto
                result.update({
                    "primary_score": 0.0,
                    "secondary_1": 0.0,
                    "error": str(e)
                })
                graded_results.append(result)
        
        return graded_results
    
    def get_task_summary(self, graded_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Genera un resumen de métricas por tipo de tarea.
        
        Args:
            graded_results: Lista de resultados evaluados
            
        Returns:
            Resumen con métricas agregadas por tarea
        """
        task_summaries = {}
        
        for result in graded_results:
            task_type = result.get("task_type")
            if not task_type:
                continue
            
            if task_type not in task_summaries:
                task_summaries[task_type] = {
                    "count": 0,
                    "primary_scores": [],
                    "secondary_scores": [],
                    "total_tokens": 0,
                    "total_latency": 0,
                    "total_cost": 0
                }
            
            summary = task_summaries[task_type]
            summary["count"] += 1
            summary["primary_scores"].append(result.get("primary_score", 0.0))
            
            # Agregar métricas secundarias
            secondary_scores = []
            for key in result.keys():
                if key.startswith("secondary_"):
                    secondary_scores.append(result[key])
            summary["secondary_scores"].extend(secondary_scores)
            
            # Agregar métricas de rendimiento
            summary["total_tokens"] += result.get("total_tokens", 0)
            summary["total_latency"] += result.get("latency_ms", 0)
            cost = result.get("cost_estimate", 0)
            if cost is not None:
                summary["total_cost"] += cost
        
        # Calcular promedios
        for task_type, summary in task_summaries.items():
            count = summary["count"]
            if count > 0:
                summary["avg_primary_score"] = sum(summary["primary_scores"]) / count
                summary["avg_secondary_score"] = sum(summary["secondary_scores"]) / len(summary["secondary_scores"]) if summary["secondary_scores"] else 0.0
                summary["avg_tokens"] = summary["total_tokens"] / count
                summary["avg_latency_ms"] = summary["total_latency"] / count
                summary["avg_cost"] = summary["total_cost"] / count if summary["total_cost"] > 0 else None
        
        return task_summaries
