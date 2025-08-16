# AI Chatbot Tutor Portfolio (10-Week Internship Collaboration)

Welcome to our consolidated portfolio for our 10-week engineering internship. This repository houses all files, source code, data analysis, and weekly logs documenting the collaborative development of an **AI Chatbot Tutor**. 

**Interns:** 
- Suyasha Rai (SuyashaRai)
- Reeyash (atlantic-hibiscuss)

**Project:** Local Retrieval-Augmented Generation (RAG) Chatbot Tutor using Open-Source Small Language Models (SLMs).

---

## What is this project about?

This project is an interactive, context-aware AI Chatbot designed to act as a personalized tutor for students in subjects like mathematics and programming. 

To ensure user privacy and allow for local execution, the system leverages a localized Small Language Model (`distilgpt2`) paired with a vector database (ChromaDB) to feed precise, educational context to the chatbot so it can answer questions accurately.

### Tech Stack
- **Core Logic:** Python 3.11+, LangChain, Hugging Face Transformers
- **Database:** ChromaDB (Vector Store), Sentence Transformers (`all-MiniLM-L6-v2`)
- **API & Interface:** FastAPI, Gradio, Uvicorn
- **DevOps & Testing:** Docker, GitHub Actions CI/CD, PyTest

---

## Detailed Step-by-Step Run Guide

### 1. Prerequisites and Setup
Create and activate a virtual environment, then install all project requirements:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Week 1: Python and ML Foundations
Run basic scripts:
```bash
python src/week1/data_structures.py
python src/week1/oop.py
python src/week1/async_example.py
```
Open Jupyter to review data visualization exercises:
```bash
jupyter notebook src/week1/Suyasha_Rai_Log_1.ipynb
```

### 3. Week 4: Base Document Ingestion
Ingest the sample mathematics corpus into the standard ChromaDB database:
```bash
python src/week4/ingest.py
```

### 4. Week 5: RAG Command Line Chat
Query the database with local history tracking:
```bash
python src/week5/conversational_rag_cli.py
```

### 5. Week 6: Web APIs & Frontend GUI
Start the FastAPI server in one terminal:
```bash
uvicorn src.week6.api:app --reload --port 8000
```
Start the Gradio interface in another terminal:
```bash
python src/week6/ui_app.py
```
Visit http://127.0.0.1:7860 to interact with the Chatbot Tutor visually.

### 6. Week 8: Advanced Metadata Ingestion & Filtering
Ingest documents using hierarchical metadata fields:
```bash
python src/week8/advanced_ingest.py
```
Query using active metadata filtering parameters:
```bash
python src/week8/advanced_query_rag.py --question "What are python functions?"
```

### 7. Week 9: Automated RAG Pipeline Evaluation
Run semantic metrics evaluation using local sentence embeddings:
```bash
python src/week9/evaluate_rag.py
```

### 8. Containerized Deployment
Build and run the API server using Docker:
```bash
docker build -t suyasharai/chatbot-tutor-api:latest .
docker run -p 8000:8000 suyasharai/chatbot-tutor-api:latest
```
