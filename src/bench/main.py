"""
CLI principal para el benchmarking de SLM vs LLM.
"""

import asyncio
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv

from .adapters import (
    OpenAIAdapter,
    HuggingFaceAdapter,
    OllamaAdapter,
    MockAdapter
)
from .eval import Grader
from .utils import (
    load_yaml,
    load_jsonl,
    save_jsonl,
    ensure_dir,
    set_seed,
    format_duration,
    format_cost
)

# Cargar variables de entorno
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Ejecutor principal del benchmarking."""
    
    def __init__(self, config_dir: str = "config"):
        """Inicializa el runner con configuración."""
        self.config_dir = Path(config_dir)
        self.providers_config = load_yaml(self.config_dir / "providers.yaml")
        self.tasks_config = load_yaml(self.config_dir / "tasks.yaml")
        self.grader = Grader()
        
        # Mapeo de proveedores a adaptadores
        self.adapter_mapping = {
            "openai": OpenAIAdapter,
            "huggingface": HuggingFaceAdapter,
            "ollama": OllamaAdapter,
            "mock": MockAdapter
        }
    
    def get_adapter(self, provider: str, model_id: str) -> Any:
        """Obtiene un adaptador para el proveedor y modelo especificados."""
        if provider not in self.providers_config["providers"]:
            raise ValueError(f"Proveedor no encontrado: {provider}")
        
        provider_config = self.providers_config["providers"][provider]
        
        if model_id not in provider_config["models"]:
            raise ValueError(f"Modelo {model_id} no encontrado en proveedor {provider}")
        
        model_config = provider_config["models"][model_id]
        adapter_class = self.adapter_mapping[provider]
        
        return adapter_class(model_id, **model_config)
    
    def load_samples(self, task_set: str) -> List[Dict[str, Any]]:
        """Carga muestras de datos para el conjunto de tareas especificado."""
        samples_file = Path("data/samples.jsonl")
        if not samples_file.exists():
            raise FileNotFoundError(f"Archivo de muestras no encontrado: {samples_file}")
        
        all_samples = load_jsonl(samples_file)
        
        if task_set == "all":
            return all_samples
        
        if task_set in self.tasks_config["task_sets"]:
            task_list = self.tasks_config["task_sets"][task_set]["tasks"]
            return [s for s in all_samples if s["task"] in task_list]
        
        # Si es una tarea específica
        if task_set in self.tasks_config["tasks"]:
            return [s for s in all_samples if s["task"] == task_set]
        
        raise ValueError(f"Conjunto de tareas no válido: {task_set}")
    
    async def run_benchmark(
        self,
        provider: str,
        model_id: str,
        task_set: str,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        seed: int = 42
    ) -> List[Dict[str, Any]]:
        """Ejecuta el benchmarking completo."""
        # Configurar semilla
        set_seed(seed)
        
        # Obtener adaptador
        adapter = self.get_adapter(provider, model_id)
        
        # Cargar muestras
        samples = self.load_samples(task_set)
        
        logger.info(f"Ejecutando benchmark: {provider}/{model_id} en {len(samples)} muestras")
        
        results = []
        
        for i, sample in enumerate(samples):
            try:
                logger.info(f"Procesando muestra {i+1}/{len(samples)}: {sample['id']}")
                
                # Generar respuesta
                start_time = datetime.now()
                response = await adapter.generate(
                    prompt=sample["prompt"],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                end_time = datetime.now()
                
                # Calcular latencia
                latency_ms = (end_time - start_time).total_seconds() * 1000
                
                # Preparar resultado
                result = {
                    "id": sample["id"],
                    "task": sample["task"],
                    "model": model_id,
                    "provider": provider,
                    "prompt": sample["prompt"],
                    "gold": sample["gold"],
                    "prediction": response["text"],
                    "latency_ms": latency_ms,
                    "prompt_tokens": response["prompt_tokens"],
                    "completion_tokens": response["completion_tokens"],
                    "total_tokens": response["total_tokens"],
                    "cost_estimate": response["cost_estimate"],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "timestamp": start_time.isoformat()
                }
                
                # Agregar esquema si existe
                if "schema" in sample:
                    result["schema"] = sample["schema"]
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error procesando muestra {sample['id']}: {e}")
                # Agregar resultado con error
                result = {
                    "id": sample["id"],
                    "task": sample["task"],
                    "model": model_id,
                    "provider": provider,
                    "prompt": sample["prompt"],
                    "gold": sample["gold"],
                    "prediction": "",
                    "error": str(e),
                    "latency_ms": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cost_estimate": None,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "timestamp": datetime.now().isoformat()
                }
                results.append(result)
        
        # Evaluar resultados
        logger.info("Evaluando resultados...")
        graded_results = self.grader.grade_batch(results)
        
        return graded_results
    
    def save_results(
        self,
        results: List[Dict[str, Any]],
        output_file: str,
        format: str = "json"
    ) -> None:
        """Guarda los resultados en el formato especificado."""
        output_path = Path(output_file)
        ensure_dir(output_path.parent)
        
        if format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        elif format == "jsonl":
            save_jsonl(results, output_path)
        else:
            raise ValueError(f"Formato no soportado: {format}")
        
        logger.info(f"Resultados guardados en: {output_path}")
    
    def generate_comparison_report(
        self,
        slm_results: List[Dict[str, Any]],
        llm_results: List[Dict[str, Any]],
        output_prefix: str
    ) -> None:
        """Genera reporte de comparación entre SLM y LLM."""
        # Evaluar resultados
        slm_graded = self.grader.grade_batch(slm_results)
        llm_graded = self.grader.grade_batch(llm_results)
        
        # Generar resúmenes
        slm_summary = self.grader.get_task_summary(slm_graded)
        llm_summary = self.grader.get_task_summary(llm_graded)
        
        # Guardar resultados completos
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON completo
        comparison_data = {
            "timestamp": timestamp,
            "slm_results": slm_graded,
            "llm_results": llm_graded,
            "slm_summary": slm_summary,
            "llm_summary": llm_summary
        }
        
        json_file = f"{output_prefix}_compare_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, ensure_ascii=False, indent=2)
        
        # CSV para análisis
        csv_file = f"{output_prefix}_compare_{timestamp}.csv"
        self._save_csv_comparison(slm_graded, llm_graded, csv_file)
        
        # Reporte Markdown
        md_file = f"{output_prefix}_compare_{timestamp}.md"
        self._save_markdown_report(slm_summary, llm_summary, md_file)
        
        logger.info(f"Reporte de comparación guardado:")
        logger.info(f"  JSON: {json_file}")
        logger.info(f"  CSV: {csv_file}")
        logger.info(f"  Markdown: {md_file}")
    
    def _save_csv_comparison(
        self,
        slm_results: List[Dict[str, Any]],
        llm_results: List[Dict[str, Any]],
        csv_file: str
    ) -> None:
        """Guarda comparación en formato CSV."""
        import pandas as pd
        
        # Preparar datos para CSV
        csv_data = []
        
        for result in slm_results + llm_results:
            csv_data.append({
                "id": result["id"],
                "task": result["task"],
                "model": result["model"],
                "provider": result["provider"],
                "primary_score": result.get("primary_score", 0.0),
                "secondary_1": result.get("secondary_1", 0.0),
                "latency_ms": result.get("latency_ms", 0),
                "total_tokens": result.get("total_tokens", 0),
                "cost_estimate": result.get("cost_estimate"),
                "prediction": result.get("prediction", ""),
                "gold": result.get("gold", "")
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False, encoding='utf-8')
    
    def _save_markdown_report(
        self,
        slm_summary: Dict[str, Any],
        llm_summary: Dict[str, Any],
        md_file: str
    ) -> None:
        """Genera reporte en formato Markdown."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(f"# Reporte de Comparación SLM vs LLM\n\n")
            f.write(f"**Fecha:** {timestamp}\n\n")
            
            # Tabla de resumen
            f.write("## Resumen por Tarea\n\n")
            f.write("| Tarea | SLM Score | LLM Score | SLM Latencia | LLM Latencia | SLM Coste | LLM Coste |\n")
            f.write("|-------|-----------|-----------|--------------|--------------|-----------|-----------|\n")
            
            all_tasks = set(slm_summary.keys()) | set(llm_summary.keys())
            for task in sorted(all_tasks):
                slm_data = slm_summary.get(task, {})
                llm_data = llm_summary.get(task, {})
                
                slm_score = f"{slm_data.get('avg_primary_score', 0):.3f}"
                llm_score = f"{llm_data.get('avg_primary_score', 0):.3f}"
                slm_latency = format_duration(slm_data.get('avg_latency_ms', 0))
                llm_latency = format_duration(llm_data.get('avg_latency_ms', 0))
                slm_cost = format_cost(slm_data.get('avg_cost'))
                llm_cost = format_cost(llm_data.get('avg_cost'))
                
                f.write(f"| {task} | {slm_score} | {llm_score} | {slm_latency} | {llm_latency} | {slm_cost} | {llm_cost} |\n")
            
            # Conclusiones
            f.write("\n## Conclusiones\n\n")
            f.write("### Tareas donde SLM iguala/supera a LLM:\n")
            # Identificar tareas donde SLM tiene mejor rendimiento
            for task in all_tasks:
                slm_data = slm_summary.get(task, {})
                llm_data = llm_summary.get(task, {})
                
                slm_score = slm_data.get('avg_primary_score', 0)
                llm_score = llm_data.get('avg_primary_score', 0)
                
                if slm_score >= llm_score * 0.9:  # SLM al menos 90% del rendimiento del LLM
                    f.write(f"- **{task}**: SLM ({slm_score:.3f}) vs LLM ({llm_score:.3f})\n")
            
            f.write("\n### Tareas donde LLM es claramente necesario:\n")
            for task in all_tasks:
                slm_data = slm_summary.get(task, {})
                llm_data = llm_summary.get(task, {})
                
                slm_score = slm_data.get('avg_primary_score', 0)
                llm_score = llm_data.get('avg_primary_score', 0)
                
                if slm_score < llm_score * 0.7:  # SLM menos del 70% del rendimiento del LLM
                    f.write(f"- **{task}**: SLM ({slm_score:.3f}) vs LLM ({llm_score:.3f})\n")
            
            f.write("\n### Trade-offs identificados:\n")
            f.write("- **Latencia**: Los SLMs suelen ser más rápidos\n")
            f.write("- **Coste**: Los SLMs suelen ser más económicos\n")
            f.write("- **Calidad**: Los LLMs suelen tener mejor rendimiento en tareas complejas\n")


def list_models():
    """Lista todos los modelos disponibles."""
    runner = BenchmarkRunner()
    
    print("Modelos disponibles:\n")
    for provider, config in runner.providers_config["providers"].items():
        print(f"**{provider.upper()}**")
        for model_id, model_config in config["models"].items():
            model_type = model_config.get("type", "unknown")
            name = model_config.get("name", model_id)
            pricing = model_config.get("pricing")
            
            pricing_str = "Gratuito" if pricing is None else f"${pricing.get('input', 0):.4f}/1K input, ${pricing.get('output', 0):.4f}/1K output"
            
            print(f"  - {model_id} ({model_type})")
            print(f"    Nombre: {name}")
            print(f"    Precio: {pricing_str}")
            print()


def list_tasks():
    """Lista todas las tareas disponibles."""
    runner = BenchmarkRunner()
    
    print("Tareas disponibles:\n")
    for task_id, task_config in runner.tasks_config["tasks"].items():
        print(f"**{task_id}**")
        print(f"  Nombre: {task_config['name']}")
        print(f"  Descripción: {task_config['description']}")
        print(f"  Métrica primaria: {task_config['primary_metric']}")
        print(f"  Métricas secundarias: {', '.join(task_config['secondary_metrics'])}")
        print()
    
    print("Conjuntos de tareas:\n")
    for set_id, set_config in runner.tasks_config["task_sets"].items():
        print(f"**{set_id}**")
        print(f"  Descripción: {set_config['description']}")
        print(f"  Tareas: {', '.join(set_config['tasks'])}")
        print()


async def run_command(args):
    """Ejecuta el comando run."""
    runner = BenchmarkRunner()
    
    # Determinar proveedor desde el modelo
    provider = args.provider
    if not provider:
        # Intentar inferir del modelo
        for p, config in runner.providers_config["providers"].items():
            if args.model in config["models"]:
                provider = p
                break
        
        if not provider:
            raise ValueError(f"No se pudo determinar el proveedor para el modelo: {args.model}")
    
    results = await runner.run_benchmark(
        provider=provider,
        model_id=args.model,
        task_set=args.tasks,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed
    )
    
    # Guardar resultados
    runner.save_results(results, args.output, "json")
    
    # Mostrar resumen
    summary = runner.grader.get_task_summary(results)
    print("\nResumen de resultados:")
    for task, task_summary in summary.items():
        print(f"  {task}: {task_summary['avg_primary_score']:.3f} (n={task_summary['count']})")


async def compare_command(args):
    """Ejecuta el comando compare."""
    runner = BenchmarkRunner()
    
    # Ejecutar benchmark para SLM
    print(f"Ejecutando benchmark para SLM: {args.slm}")
    slm_results = await runner.run_benchmark(
        provider="mock" if args.mock else "ollama",  # Simplificado para demo
        model_id=args.slm,
        task_set=args.tasks,
        temperature=args.temperature,
        seed=args.seed
    )
    
    # Ejecutar benchmark para LLM
    print(f"Ejecutando benchmark para LLM: {args.llm}")
    llm_results = await runner.run_benchmark(
        provider="mock" if args.mock else "openai",  # Simplificado para demo
        model_id=args.llm,
        task_set=args.tasks,
        temperature=args.temperature,
        seed=args.seed
    )
    
    # Generar reporte
    output_prefix = "reports/compare"
    ensure_dir("reports")
    runner.generate_comparison_report(slm_results, llm_results, output_prefix)


def main():
    """Función principal del CLI."""
    parser = argparse.ArgumentParser(
        description="Benchmarking tool para comparar SLMs vs LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python -m bench.main list-models
  python -m bench.main list-tasks
  python -m bench.main run --provider mock --model mock-slm --tasks core --output results.json
  python -m bench.main compare --slm mock-slm --llm mock-llm --tasks core --md-report
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comandos disponibles')
    
    # Comando list-models
    subparsers.add_parser('list-models', help='Lista todos los modelos disponibles')
    
    # Comando list-tasks
    subparsers.add_parser('list-tasks', help='Lista todas las tareas disponibles')
    
    # Comando run
    run_parser = subparsers.add_parser('run', help='Ejecuta benchmark para un modelo')
    run_parser.add_argument('--provider', help='Proveedor del modelo')
    run_parser.add_argument('--model', required=True, help='ID del modelo')
    run_parser.add_argument('--tasks', default='core', help='Conjunto de tareas (core, all, slm_focused, llm_focused)')
    run_parser.add_argument('--output', required=True, help='Archivo de salida')
    run_parser.add_argument('--temperature', type=float, default=0.2, help='Temperatura para generación')
    run_parser.add_argument('--max-tokens', type=int, help='Máximo número de tokens')
    run_parser.add_argument('--seed', type=int, default=42, help='Semilla para reproducibilidad')
    
    # Comando compare
    compare_parser = subparsers.add_parser('compare', help='Compara SLM vs LLM')
    compare_parser.add_argument('--slm', required=True, help='ID del modelo SLM')
    compare_parser.add_argument('--llm', required=True, help='ID del modelo LLM')
    compare_parser.add_argument('--tasks', default='core', help='Conjunto de tareas')
    compare_parser.add_argument('--temperature', type=float, default=0.2, help='Temperatura para generación')
    compare_parser.add_argument('--seed', type=int, default=42, help='Semilla para reproducibilidad')
    compare_parser.add_argument('--md-report', action='store_true', help='Generar reporte Markdown')
    compare_parser.add_argument('--mock', action='store_true', help='Usar modo mock')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'list-models':
            list_models()
        elif args.command == 'list-tasks':
            list_tasks()
        elif args.command == 'run':
            asyncio.run(run_command(args))
        elif args.command == 'compare':
            asyncio.run(compare_command(args))
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
