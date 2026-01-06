import os
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="AI Document Intelligence with RAG",
    layout="wide"
)

# ======================================
# GROQ CONFIG
# ======================================
MODEL_NAME = "llama-3.1-8b-instant"

@st.cache_resource
def load_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not set in environment variables")
        st.stop()
    return Groq(api_key=api_key)

client = load_groq_client()

# ======================================
# PDF PROCESSING
# ======================================
def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def chunk_text(text, source, chunk_size=800, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append({
            "text": text[start:end],
            "source": source
        })
        start += chunk_size - overlap
    return chunks

# ======================================
# VECTOR STORE
# ======================================
@st.cache_resource
def build_vector_store(chunks):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, embedder, chunks

def retrieve_context(question, chunks, index, embedder, k=4):
    q_embedding = embedder.encode([question]).astype("float32")
    _, indices = index.search(q_embedding, k)

    retrieved = [chunks[i] for i in indices[0]]
    context = "\n\n".join([r["text"] for r in retrieved])
    sources = list(set([r["source"] for r in retrieved]))

    return context, sources

# ======================================
# LLM CALL
# ======================================
def ask_groq(context, question):
    prompt = f"""
You are an intelligent document assistant.
Answer ONLY using the context below.
If the answer is not present, say "Not found in documents".

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

# ======================================
# STREAMLIT UI
# ======================================
st.title("AI Document Intelligence with RAG")

st.write(
    "Upload multiple PDF documents and ask questions across all of them "
    "using Retrieval-Augmented Generation."
)

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    all_chunks = []

    with st.spinner("Processing documents..."):
        for file in uploaded_files:
            text = read_pdf(file)
            chunks = chunk_text(text, file.name)
            all_chunks.extend(chunks)

    with st.spinner("Building vector database..."):
        index, embedder, stored_chunks = build_vector_store(all_chunks)

    st.success(
        f"Loaded {len(uploaded_files)} documents "
        f"({len(all_chunks)} text chunks)"
    )

    question = st.text_area(
        "Ask a question",
        placeholder="Example: What are the key points mentioned across all documents?"
    )

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating answer..."):
                context, sources = retrieve_context(
                    question,
                    stored_chunks,
                    index,
                    embedder
                )
                answer = ask_groq(context, question)

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Sources")
            for src in sources:
                st.write("-", src)

            st.subheader("Retrieved Context")
            st.write(context)
