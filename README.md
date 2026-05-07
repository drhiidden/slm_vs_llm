![SLM vs LLM banner](docs/banner.png)

# SLM vs LLM

**Benchmarking Small Language Models vs Large Language Models**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![visitors](https://komarev.com/ghpvc/?username=drhiidden&repo=slm_vs_llm&color=00ff88&style=flat-square)

**Qué es**: Framework Python para comparar modelos de lenguaje pequeños (SLMs, <10B params) vs grandes (LLMs, >10B params) en tareas reales de código y reasoning.

**Objetivo**: Demostrar que SLMs (Llama 3 8B, Mistral 7B) pueden competir con LLMs (GPT-4, Claude 3.5) en tareas específicas con el prompt correcto.

---

## Por qué esto importa

**Problema**: Se asume que siempre necesitas GPT-4/Claude para tareas complejas.  
**Realidad**: SLMs bien tuneados pueden ser 10x más baratos y 5x más rápidos en tareas específicas.

**Casos de uso**:
- Code generation simple (funciones unitarias)
- Q&A sobre docs técnicas
- Clasificación de texto
- Entity extraction

**Trade-off**:
- ✅ SLM: Costo bajo, latencia baja, privacidad (local)
- ❌ LLM: Costo alto, latencia alta, requiere API externa

---

## Quick Start

### Requirements

- Python 3.11+
- 16GB RAM (para SLMs locales via Ollama)
- API keys: OpenAI (opcional), Anthropic (opcional)

### Setup

```bash
# Clonar
git clone https://github.com/drhiidden/slm_vs_llm.git
cd slm_vs_llm

# Instalar deps
pip install -r requirements.txt

# Configurar Ollama (para SLMs locales)
curl https://ollama.ai/install.sh | sh
ollama pull llama3:8b
ollama pull mistral:7b

# Configurar API keys (opcional, para LLMs)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Run Benchmarks

```bash
# Benchmark 1: Code generation
python benchmarks/code_generation.py

# Benchmark 2: Reasoning tasks
python benchmarks/reasoning.py

# Benchmark 3: Cost analysis
python benchmarks/cost_analysis.py

# Ver reporte
open reports/benchmark_results.html
```

---

## Benchmarks Incluidos

1. **Code Generation** (HumanEval subset)
   - Task: Generar funciones Python desde docstrings
   - Métrica: Pass@1, latencia, costo

2. **Reasoning** (GSM8K math problems)
   - Task: Resolver problemas matemáticos
   - Métrica: Accuracy, razonamiento correcto

3. **Text Classification** (AG News)
   - Task: Clasificar noticias en 4 categorías
   - Métrica: Accuracy, F1-score

---

## Resultados Preliminares (v0.1)

| Modelo | Params | Pass@1 | Latency | Cost/1K | Local |
|--------|--------|--------|---------|---------|-------|
| **Llama 3 8B** | 8B | 67% | 1.2s | $0 | ✅ |
| **Mistral 7B** | 7B | 63% | 1.0s | $0 | ✅ |
| GPT-4 Turbo | ~1.7T | 91% | 3.5s | $0.03 | ❌ |
| Claude 3.5 Sonnet | ? | 89% | 2.8s | $0.015 | ❌ |

**Conclusión**: Para tareas simples-medias, SLMs son 3x más rápidos y gratis (si local).

---

## Stack

Python 3.11+ · Ollama · OpenAI/Anthropic APIs · Pandas · Matplotlib

---

## Documentación

- **[Benchmarks](docs/benchmarks.md)** - Detalles de cada benchmark
- **[Analysis](docs/analysis.md)** - Análisis de resultados
- **[AGENTS.md](AGENTS.md)** - Setup técnico paso a paso
- **[CHANGELOG.md](CHANGELOG.md)** - Historial de versiones

---

## Roadmap

- **v0.2** (2 months): Más benchmarks (RAG, summarization)
- **v0.3** (4 months): Fine-tuning SLMs para casos específicos
- **v1.0** (6 months): Dashboard interactivo, reportes automáticos

---

## Licencia

MIT - Ver [LICENSE](LICENSE)

---

**Metodología**: Desarrollado con [HCP (Human-Code-AI Protocol)](https://github.com/haletheia/human-code-ai-protocol)
