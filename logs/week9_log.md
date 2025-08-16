# Week 9 (Days 71–80) — Automated RAG Evaluation & Logging

**Focus:** Quantitative measurement of RAG pipeline performance (Retrieval Relevance, Answer Similarity) and logging systems.  
**Deliverables:** RAG evaluation script utilizing cosine similarity metrics on local sentence embeddings.

## Step-by-step

1) **Set up evaluation data**
   - Curated a benchmark dataset consisting of standard math/Python questions and high-quality human-written reference answers (ground truth) inside `src/week9/evaluate_rag.py`.

2) **Embedding-based Cosine Similarity metrics**
   - File: `src/week9/evaluate_rag.py`
   - Formulated mathematical functions to calculate vectors and compute the cosine angle between sentence pairs using the local `sentence-transformers/all-MiniLM-L6-v2` model.
   - Measures:
     - **Context Relevance:** Semantic similarity between the user query and the retrieved vector-store chunks (verifies retriever recall).
     - **Answer Similarity:** Semantic similarity between the generated LLM text and the human reference ground truth (verifies generation quality).

3) **Run evaluation pipeline**
   - Execute the test suite locally:
     ```bash
     python src/week9/evaluate_rag.py
     ```

## Challenges and Resolutions
- **LLM Hallucinations on CPU:** The small `distilgpt2` model frequently generated redundant trailing text. Resolved by filtering out the inputs and splitting generated text cleanly at the `Answer:` token.
