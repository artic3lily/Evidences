# Week 4 (Days 31–40) — Vector DB & Retrieval-Augmented Generation (RAG)

**Focus:** Build a tiny domain KB, embed it, store locally with Chroma, and query via RAG.  
**Deliverables:** Prepared knowledge base, ingestion + query scripts.

## Step-by-step

1) **Prepare small knowledge base**
- Edit or add `.md` files in `src/week4/data/sample_knowledge/`.

2) **Ingest (chunk + embed + persist)**
```bash
python src/week4/ingest.py
```

3) **Query via RetrievalQA**
```bash
python src/week4/query_rag.py --question "Explain lists vs dicts"
```

