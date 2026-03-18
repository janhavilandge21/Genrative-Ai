# 📄 AI-Powered Document Intelligence System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![LangChain](https://img.shields.io/badge/LangChain-Framework-green)
![FAISS](https://img.shields.io/badge/FAISS-VectorDB-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![LLM](https://img.shields.io/badge/LLM-Groq-purple)


![Image](https://www.researchgate.net/publication/378364457/figure/fig1/AS%3A11431281225000902%401708532903222/Retrieval-Augmented-Generation-Architecture.png)

![Image](https://global.discourse-cdn.com/streamlit/original/3X/5/f/5f935cb9df829925099da453f672c46a09da244b.gif)

![Image](https://cdn.prod.website-files.com/5ee50f2ef83ac07f0cb7fb44/669a0f3de1068f88799c3fe8_chatbot-plan-vector-database.png)

![Image](https://statusneo.com/wp-content/uploads/2024/03/image-9-1024x681.png)



An **end-to-end AI-powered document intelligence system** that allows users to upload **multiple PDF documents** and ask intelligent, context-aware questions using **Retrieval-Augmented Generation (RAG)**.

Built with **Python, Streamlit, FAISS, and Groq-powered LLaMA models**.

---

## 🚀 Project Overview

This project demonstrates how modern **enterprise AI systems** answer questions from large document collections **without hallucination** by combining:

* 🔍 **Semantic Retrieval** (Vector Search)
* 🧠 **Large Language Models (LLMs)**
* 📚 **Multi-document reasoning**

Instead of fine-tuning models, this system uses **RAG**, which is **faster, cheaper, and production-ready**.

---

## ✨ Key Features

* 📚 **Multi-PDF Upload** – Ask questions across many documents
* 🔍 **Semantic Search** – FAISS vector similarity search
* 🧠 **RAG Pipeline** – Accurate, grounded answers
* ⚡ **Low-Latency LLM** – Groq-hosted LLaMA 3.1
* 🖥️ **Streamlit Frontend** – Clean, interactive UI
* 📄 **Source Attribution** – Shows which documents were used

---

## 🧠 Architecture Diagram

📌 **Add this image to your repo**
Filename: `assets/architecture.png`

```md
![RAG Architecture](assets/architecture.png)
```

### Architecture Flow

```
Multiple PDFs
      ↓
Text Extraction
      ↓
Chunking
      ↓
Embeddings (SentenceTransformers)
      ↓
FAISS Vector Database
      ↓
Top-K Retrieval
      ↓
Groq LLaMA Model
      ↓
Final Answer
```

---

## 🖥️ Application UI (Streamlit)

📌 **Add this image to your repo**
Filename: `assets/ui.png`

```md
![Streamlit UI](assets/ui.png)
```

**UI Capabilities**

* Upload multiple PDFs
* Ask natural language questions
* View AI-generated answers
* Inspect retrieved context & sources

---

## 🛠️ Tech Stack

| Layer           | Technology                     |
| --------------- | ------------------------------ |
| Language        | Python                         |
| Frontend        | Streamlit                      |
| Embeddings      | SentenceTransformers           |
| Vector Store    | FAISS                          |
| LLM             | Groq (LLaMA 3.1)               |
| File Processing | PyPDF                          |
| GenAI Pattern   | Retrieval-Augmented Generation |

---

## 📂 Project Structure

```
rag_document_intelligence/
├── app.py
├── requirements.txt
├── assets/
│   ├── architecture.png
│   └── ui.png
└── README.md
```



---







---

## 🧪 How to Use

1. Upload **one or more PDF documents**
2. Wait for document processing
3. Enter a question
4. Receive an AI-generated answer grounded in document content
5. View sources and retrieved context

---

## 📌 Example Queries

* *What are the key points across all documents?*
* *Compare the policies mentioned in these PDFs*
* *Summarize the conclusions*

---


---

## 🔮 Future Enhancements

* Chat history & conversational memory
* Persistent vector storage
* Per-document filtering
* Support for DOCX / TXT files
* FastAPI backend integration
* Cloud deployment

---

