# Architecture Overview

- **LLM**: `distilgpt2` via Hugging Face `pipeline` (CPU-friendly)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector DB**: Chroma (local, persisted under `./chroma_db`)
- **Memory**: LangChain `ConversationBufferMemory`
- **Chains**: `RetrievalQA`, `ConversationalRetrievalChain`
- **Web**: FastAPI backend + Gradio UI

## Data Flow (Conversational RAG)
1. User asks a question.
2. Retriever pulls relevant chunks from Chroma.
3. Chain feeds retrieved context + chat history to LLM.
4. LLM generates answer; memory updated.