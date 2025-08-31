# api.py - Week 6

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Chatbot Tutor API - Suyasha Rai")

_history = []


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    global _history

    if "rag" in req.message.lower():
        answer = (
            "RAG stands for Retrieval-Augmented Generation. It combines retrieval "
            "with generation to produce better answers."
        )
    else:
        answer = (
            "Generative AI creates new content such as text, images, or music."
        )

    _history.append({"message": req.message, "answer": answer})
    return ChatResponse(answer=answer)


@app.get("/health")
def health():
    return {"status": "ok", "project": "Chatbot Tutor", "author": "Suyasha Rai"}
