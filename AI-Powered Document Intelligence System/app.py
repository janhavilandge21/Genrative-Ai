import os
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ================================
# CONFIG
# ================================
MODEL_NAME = "llama-3.1-8b-instant"   # ✅ stable Groq model

# ================================
# GROQ CLIENT
# ================================
@st.cache_resource
def load_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("❌ GROQ_API_KEY not set in environment variables")
        st.stop()
    return Groq(api_key=api_key)

client = load_groq_client()

# ================================
# PDF PROCESSING
# ================================
def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=800, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ================================
# VECTOR STORE
# ================================
@st.cache_resource
def build_vector_store(chunks):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(chunks).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, embedder

def retrieve_context(question, chunks, index, embedder, k=3):
    q_embedding = embedder.encode([question]).astype("float32")
    _, indices = index.search(q_embedding, k)
    return "\n\n".join([chunks[i] for i in indices[0]])

# ================================
# LLM CALL
# ================================
def ask_groq(context, question):
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
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()

# ================================
# STREAMLIT UI
# ================================
st.set_page_config(
    page_title="AI Document Intelligence (RAG)",
    layout="wide"
)

st.title("📄 AI Document Intelligence with RAG")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Tech Stack**
    - Python
    - Streamlit
    - FAISS
    - SentenceTransformers
    - Groq (LLaMA)
    """)

# Upload PDF
uploaded_file = st.file_uploader(
    "📤 Upload a PDF document",
    type=["pdf"]
)

if uploaded_file:
    with st.spinner("Reading PDF..."):
        text = read_pdf(uploaded_file)
        chunks = chunk_text(text)

    with st.spinner("Building vector store..."):
        index, embedder = build_vector_store(chunks)

    st.success(f"✅ Document processed ({len(chunks)} chunks)")

    question = st.text_area(
        "❓ Ask a question from the document",
        placeholder="e.g. What is the main topic of this document?"
    )

    if st.button("Ask Question"):
        if not question.strip():
            st.warning("Please enter a question")
        else:
            with st.spinner("Thinking..."):
                context = retrieve_context(question, chunks, index, embedder)
                answer = ask_groq(context, question)

            st.subheader("🧠 Answer")
            st.write(answer)

            with st.expander("🔍 Retrieved Context"):
                st.write(context)
