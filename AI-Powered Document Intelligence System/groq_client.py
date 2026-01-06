import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.1-8b-instant"

def call_llm(context: str, question: str) -> str:
    prompt = f"""
You are an intelligent document assistant.
Answer ONLY using the context below.
If the answer is not present, say "Not found in document".

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()
