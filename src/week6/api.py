# api.py — Week 6
# FastAPI backend for the Chatbot Tutor using HF Inference API + ChromaDB
from fastapi import FastAPI
from pydantic import BaseModel
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.llm_utils import call_llm
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

app = FastAPI(title="Chatbot Tutor API — Suyasha Rai")

PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "chroma_db")

# Load embeddings and vector DB once at startup
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb   = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
retriever  = vectordb.as_retriever(search_kwargs={"k": 3})

# Simple in-memory conversation history per server session
_history: str = ""

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    global _history

    docs = retriever.get_relevant_documents(req.message)
    context = "\n".join(d.page_content for d in docs)

    prompt = (
        "You are a helpful tutor. Use the reference material and "
        "conversation history to answer the student's question.\n\n"
        f"Reference material:\n{context}\n\n"
        f"Conversation history:\n{_history}"
        f"Student: {req.message}\n"
        f"Tutor:"
    )

    answer = call_llm(prompt, max_new_tokens=150)

    _history += f"Student: {req.message}\nTutor: {answer}\n"
    lines = _history.strip().split("\n")
    if len(lines) > 8:
        _history = "\n".join(lines[-8:]) + "\n"

    return ChatResponse(answer=answer)

@app.get("/health")
def health():
    return {"status": "ok", "project": "Chatbot Tutor", "author": "Suyasha Rai"}