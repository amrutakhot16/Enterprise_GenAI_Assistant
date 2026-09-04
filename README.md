# Enterprise GenAI Assistant

An enterprise-focused Retrieval-Augmented Generation (RAG) application that allows users to ask questions about internal company documents and receive AI-generated answers with document source citations.

## Features

- PDF document ingestion and text chunking
- Semantic document search using vector embeddings
- LLM-powered question answering
- Multi-document retrieval
- Source citations with document name and page number
- Conversation history for follow-up questions
- FastAPI backend REST API
- Streamlit chat interface
- API keys managed securely through environment variables

## Tech Stack

- **Python**
- **FastAPI** – Backend API
- **Streamlit** – Frontend
- **LangChain** – RAG orchestration
- **ChromaDB** – Vector database
- **Hugging Face Sentence Transformers** – Text embeddings
- **Groq** – LLM inference
- **PyPDF** – PDF document loading

## Project Structure

```text
Enterprise_GenAI_Assistant/
│
├── app/
│   ├── api/
│   │   └── routes.py
│   │
│   ├── core/
│   │
│   ├── llm/
│   │   ├── model.py
│   │   └── prompts.py
│   │
│   ├── memory/
│   │   └── chat_memory.py
│   │
│   ├── rag/
│   │   └── vectorstore.py
│   │
│   ├── services/
│   │   └── chat_service.py
│   │
│   └── main.py
│
├── documents/
│   ├── products.pdf
│   ├── strategy.pdf
│   └── test_doc.pdf
│
├── frontend/
│   └── app.py
│
├── .gitignore
├── requirements.txt
└── README.md