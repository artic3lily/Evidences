# query_rag.py — Week 4
# Query the ChromaDB vector store and answer questions using HF Inference API
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.llm_utils import call_llm
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "chroma_db")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True, help="Question to ask the RAG bot")
    args = parser.parse_args()

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    docs = retriever.get_relevant_documents(args.question)
    context = "\n".join(d.page_content for d in docs)

    prompt = (
        f"Use the following reference material to answer the question.\n\n"
        f"Reference:\n{context}\n\n"
        f"Question: {args.question}\n"
        f"Answer:"
    )

    answer = call_llm(prompt, max_new_tokens=150)
    print("\n[Answer]:", answer)
    print("\n[Sources]:")
    for d in docs:
        print(" -", d.metadata.get("source"))

if __name__ == "__main__":
    main()