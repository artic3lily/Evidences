# Week 5 (Days 41–50) — Combining RAG & Memory

**Focus:** Merge RAG retrieval with conversation memory for smoother multi-turn help.  
**Deliverables:** ConversationalRetrievalChain-based CLI tutor.

## Step-by-step

1) **Prereq: make sure Week 4 DB exists**
```bash
python src/week4/ingest.py
```

2) **Run Conversational RAG CLI**
```bash
python src/week5/conversational_rag_cli.py
```

3) **Experiment**
- Ask questions that need external facts (KB) and that build on prior turns.

