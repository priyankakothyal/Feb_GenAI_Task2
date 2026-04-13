# ==============================
# 🔹 LOCAL LLM USING TRANSFORMERS
# ==============================

from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")

response = generator(
    "Explain LangChain in simple terms",
    max_length=100,
    num_return_sequences=1
)

print("\nRaw Model Output:\n", response[0]["generated_text"])


# ==============================
# 🔹 USING LANGCHAIN WRAPPER
# ==============================

from langchain.llms import HuggingFacePipeline

llm = HuggingFacePipeline(pipeline=generator)

response = llm.invoke("Explain LangChain in simple terms")

print("\nLangChain Output:\n", response)