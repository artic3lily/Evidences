# Week 6 (Days 51–57) — Web Interface & Deployment Basics

**Focus:** Expose the tutor via a simple API, add a web UI, and discuss deployment.  
**Deliverables:** FastAPI backend + Gradio front-end that talk to your LangChain backend.

## Step-by-step

1) **Start the API**
```bash
uvicorn src.week6.api:app --reload
```

2) **Run the UI**
```bash
python src/week6/ui_app.py
```

3) **Try chatting**
- The API holds the memory on the server side; the UI just sends messages.

4) **(Optional) Containerization discussion**
- Create a Dockerfile later if you want to containerize; push to a registry for cloud deploy.

