from transformers import pipeline

# 🔹 LLM grande (ejemplo: GPT-J-6B, un modelo de 6B parámetros)
llm = pipeline("text-generation", model="EleutherAI/gpt-j-6B")

# 🔹 SLM pequeño (ejemplo: DistilGPT2, mucho más liviano)
slm = pipeline("text-generation", model="distilgpt2")

# Pregunta sencilla
prompt = "¿Cuál es la capital de Francia?"

print("Respuesta LLM grande:")
print(llm(prompt, max_length=30)[0]["generated_text"])

print("\nRespuesta SLM pequeño:")
print(slm(prompt, max_length=30)[0]["generated_text"])
