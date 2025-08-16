# conversational_rag_cli.py — Week 5
# Conversational RAG CLI — retrieves context from ChromaDB, answers via HF Inference API
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.llm_utils import call_llm
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "chroma_db")

def main():
    if not os.path.exists(PERSIST_DIR):
        print(f"Vector DB not found at {PERSIST_DIR}. Run src/week4/ingest.py first.")
        return

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    print("Conversational RAG Tutor — type 'exit' to quit.\n")
    history = ""

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # Retrieve relevant context
        docs = retriever.get_relevant_documents(question)
        context = "\n".join(d.page_content for d in docs)

        prompt = (
            "You are a helpful tutor. Use the reference material below and "
            "the conversation history to answer the student's question.\n\n"
            f"Reference material:\n{context}\n\n"
            f"Conversation history:\n{history}"
            f"Student: {question}\n"
            f"Tutor:"
        )

        answer = call_llm(prompt, max_new_tokens=150)
        print(f"Tutor: {answer}\n")

        history += f"Student: {question}\nTutor: {answer}\n"
        # Keep last 4 turns to stay within token limits
        lines = history.strip().split("\n")
        if len(lines) > 8:
            history = "\n".join(lines[-8:]) + "\n"

if __name__ == "__main__":
    main()