# advanced_query_rag.py — Week 8
# Advanced retrieval with metadata filtering using HF Inference API
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.llm_utils import call_llm
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "chroma_db_advanced")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    if not os.path.exists(PERSIST_DIR):
        print(f"Advanced DB not found at {PERSIST_DIR}. Run advanced_ingest.py first.")
        return

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

    # Retrieve using metadata filter (doc_type = academic_tutor_reference)
    results = vectordb.similarity_search(
        args.question,
        k=3,
        filter={"doc_type": "academic_tutor_reference"}
    )

    context = "\n".join(d.page_content for d in results)
    prompt = (
        f"Use the following academic reference material to answer the question.\n\n"
        f"Reference:\n{context}\n\n"
        f"Question: {args.question}\n"
        f"Answer:"
    )

    answer = call_llm(prompt, max_new_tokens=150)
    print("\n[Advanced RAG Answer]:", answer)
    print("\n[Retrieved Enriched Sources]:")
    for d in results:
        print(f" - Source: {d.metadata.get('parent_source')} (Author: {d.metadata.get('author')})")

if __name__ == "__main__":
    main()
