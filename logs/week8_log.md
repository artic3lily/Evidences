# Week 8 (Days 61–70) — Advanced RAG Optimization & Containerization

**Focus:** Hierarchical chunking strategies, metadata enrichment, query filtering, and containerization using Docker.  
**Deliverables:** Advanced ingestion and query scripts; root-level Dockerfile for FastAPI deployment.

## Step-by-step

1) **Advanced Ingestion Strategy**
   - File: `src/week8/advanced_ingest.py`
   - Added metadata fields (such as `doc_type` and `author` set to "Suyasha Rai") to chunks.
   - Implemented hierarchical chunking where child chunks carry context paths back to their parent files, preventing loss of context when working with short token widths.
   - Run advanced ingestion:
     ```bash
     python src/week8/advanced_ingest.py
     ```

2) **Query Filtering with Metadata**
   - File: `src/week8/advanced_query_rag.py`
   - Configured retriever to filter retrieval queries based on document metadata attributes (e.g. retrieving only chunks classified as `academic_tutor_reference`).
   - Run query script:
     ```bash
     python src/week8/advanced_query_rag.py --question "What are Python basics?"
     ```

3) **Docker Setup**
   - File: `Dockerfile`
   - Set up a lightweight `python:3.11-slim` container base.
   - Exposed port 8000 and mapped the entrypoint command to run the Uvicorn-hosted API backend (`src.week6.api:app`).

## Challenges and Resolutions
- **Chroma Filter Matching:** Ensuring database filters matched exact key types in ChromaDB. Resolved by casting input metadata explicitly and initializing Chroma with matching embeddings before persistence.
