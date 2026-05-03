# AGENTS.md — slm_vs_llm

Guía para agentes IA y el autor en futuras versiones.

---

## Identidad

| Campo | Valor |
|---|---|
| Nombre | **slm_vs_llm** |
| Repo GitHub | `drhiidden/slm_vs_llm` |
| Tagline | *Not every task needs GPT-4.* |
| Licencia | MIT |
| Estado | OSS público, estable |

---

## ¿Qué hace?

Benchmark reproducible que enfrenta SLMs (phi-3-mini, mistral-7b via Ollama) contra LLMs (GPT-4o, Claude 3.5 via API) en 5 tipos de tareas de agentes:

1. **Exact QA** — respuesta exacta de una pregunta
2. **Math** — resolución aritmética/algebraica
3. **JSON Toolcall** — generar JSON estructurado válido
4. **Classification** — categorizar texto
5. **Reasoning** — razonamiento multi-paso

Métricas: latencia (ms), calidad (Exact match / ROUGE-L / schema validation), coste estimado (USD).

---

## Stack

| Capa | Tecnología |
|---|---|
| Lenguaje | Python 3.10+ |
| Proveedores | OpenAI, Anthropic, HuggingFace, Ollama, mock |
| Config | `config/` YAML |
| Output | JSON + CSV + Markdown |
| Entrada de datos | `data/samples.jsonl` |

---

## Estructura

```
slm_vs_llm/
├── benchmarks/       ← Lógica de benchmark por tipo de tarea
├── providers/        ← Adaptadores para cada proveedor LLM/SLM
├── metrics/          ← Cálculo de Exact match, ROUGE-L, JSON schema validation
├── reports/          ← Generación de reportes Markdown/CSV/JSON
├── config/           ← Configuraciones YAML (modelos, tareas, parámetros)
├── data/
│   └── samples.jsonl ← Dataset de prompts de evaluación
└── main.py           ← CLI principal
```

---

## Reglas críticas

1. **Mock provider siempre disponible** — para tests sin API keys
2. **Semilla configurable** — reproducibilidad garantizada
3. **Añadir proveedor**: implementar interfaz en `providers/`, registrar en `config/`
4. **No hardcodear API keys** — siempre via `.env` o variables de entorno
5. **Ollama = local, gratis** — siempre testar con Ollama antes de reportar resultados con APIs de pago

---

## Cómo añadir un nuevo modelo

```python
# 1. providers/nuevo_provider.py
class NuevoProvider(BaseProvider):
    def generate(self, prompt: str, **kwargs) -> BenchmarkResult:
        ...

# 2. config/models.yaml
models:
  - name: nuevo-modelo
    provider: nuevo_provider
    cost_per_1k_tokens: 0.001
```

---

## Metodología

Desarrollado con [HCP (Human-Code-AI Protocol)](https://github.com/haletheia/human-code-ai-protocol).
