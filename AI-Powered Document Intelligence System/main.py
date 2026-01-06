from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq_client import call_llm

# -------------------------------
# 1. Load PDF
# -------------------------------
def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# -------------------------------
# 2. Chunk text
# -------------------------------
def chunk_text(text, chunk_size=800, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# -------------------------------
# 3. Build Vector Store
# -------------------------------
def build_faiss(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings, model

# -------------------------------
# 4. Retrieve relevant chunks
# -------------------------------
def retrieve(question, chunks, index, model, k=3):
    q_embedding = model.encode([question]).astype("float32")
    _, indices = index.search(q_embedding, k)
    return "\n\n".join([chunks[i] for i in indices[0]])

# -------------------------------
# 5. Main App
# -------------------------------
def main():
    print("📄 Loading document...")
    text = load_pdf("Langchain.pdf")

    print("✂️ Chunking text...")
    chunks = chunk_text(text)

    print("📦 Building vector store...")
    index, _, embed_model = build_faiss(chunks)

    print("\n✅ Document ready. Ask questions (type 'exit' to quit)\n")

    while True:
        question = input("❓ Question: ")
        if question.lower() == "exit":
            break

        context = retrieve(question, chunks, index, embed_model)
        answer = call_llm(context, question)

        print("\n🧠 Answer:\n", answer)
        print("-" * 60)

if __name__ == "__main__":
    main()
