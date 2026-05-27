# evaluate_rag.py — Week 9
# Automated RAG pipeline evaluation using cosine similarity on sentence embeddings.
# The LLM call uses the HF Inference API — no local model download required.

import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.llm_utils import call_llm
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "chroma_db")

EVAL_DATA = [
    {
        "question": "What is 3/4 as a decimal?",
        "ground_truth": "3/4 is equal to 0.75 as a decimal."
    },
    {
        "question": "What does PEMDAS stand for?",
        "ground_truth": "PEMDAS stands for Parentheses, Exponents, Multiplication/Division, Addition/Subtraction."
    },
    {
        "question": "What is a python function?",
        "ground_truth": "A function in Python groups reusable code and can accept parameters and return values."
    }
]

def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return float(dot / (n1 * n2)) if n1 and n2 else 0.0

def main():
    print("Loading sentence transformer for evaluation metrics...")
    eval_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    if not os.path.exists(PERSIST_DIR):
        print(f"Vector DB not found at {PERSIST_DIR}. Run ingest.py first.")
        return

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})

    print("\n--- Starting RAG Evaluation ---")
    results = []

    for item in EVAL_DATA:
        q, gt = item["question"], item["ground_truth"]
        print(f"\nEvaluating: '{q}'")

        docs = retriever.get_relevant_documents(q)
        context = " ".join(d.page_content for d in docs)

        prompt = (
            f"Reference:\n{context}\n\n"
            f"Question: {q}\nAnswer:"
        )
        answer = call_llm(prompt, max_new_tokens=80)

        q_emb  = eval_model.encode(q)
        ctx_emb = eval_model.encode(context)
        ans_emb = eval_model.encode(answer)
        gt_emb  = eval_model.encode(gt)

        ctx_rel  = cosine_similarity(q_emb, ctx_emb)
        ans_sim  = cosine_similarity(ans_emb, gt_emb)

        results.append({"question": q, "context_relevance": ctx_rel, "answer_similarity": ans_sim})
        print(f"  Generated: '{answer[:120]}...'")
        print(f"  Context Relevance : {ctx_rel:.4f}")
        print(f"  Answer Similarity : {ans_sim:.4f}")

    print("\n--- Summary ---")
    print(f"Avg Context Relevance : {np.mean([r['context_relevance'] for r in results]):.4f}")
    print(f"Avg Answer Similarity : {np.mean([r['answer_similarity'] for r in results]):.4f}")
    print("Evaluation complete.")

if __name__ == "__main__":
    main()
