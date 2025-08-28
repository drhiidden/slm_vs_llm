
from transformers import pipeline

# 🔹 SLM pequeño (más barato/rápido, para tareas simples)
slm = pipeline("text-generation", model="distilgpt2")

# 🔹 LLM más grande (más potente, para razonamiento complejo)
llm = pipeline("text-generation", model="EleutherAI/gpt-j-6B")

# 🔹 Orquestador: decide qué modelo usar según el tipo de pregunta
def orquestador(pregunta: str):
    # Si la pregunta es factual/simple → usar SLM
    if any(p in pregunta.lower() for p in ["capital", "suma", "resta", "color", "animal"]):
        respuesta = slm(pregunta, max_length=40, do_sample=True)[0]["generated_text"]
        return f"[SLM usado] {respuesta}"
    else:
        # Si es abierta/creativa → usar LLM
        respuesta = llm(pregunta, max_length=80, do_sample=True)[0]["generated_text"]
        return f"[LLM usado] {respuesta}"

# 🔹 Ejemplos de uso
print(orquestador("¿Cuál es la capital de Francia?"))   # SLM
print(orquestador("Hazme un plan de negocios para una app de delivery en Madrid"))  # LLM
