# advanced_ingest.py
# Week 8: Advanced RAG Ingestion with Metadata Enrichment and Parent Mapping simulation

import os
import glob
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Base paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "week4", "data", "sample_knowledge")
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "chroma_db_advanced")

def load_docs(paths: List[str]):
    docs = []
    for p in paths:
        loader = TextLoader(p, encoding="utf-8")
        loaded_docs = loader.load()
        # Add metadata enrichment: document classification/type
        for d in loaded_docs:
            d.metadata["doc_type"] = "academic_tutor_reference"
            d.metadata["author"] = "Suyasha Rai"
        docs.extend(loaded_docs)
    return docs

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Data directory {DATA_DIR} does not exist. Creating default directory and files...")
        os.makedirs(DATA_DIR, exist_ok=True)
        # Write dummy knowledge if missing
        with open(os.path.join(DATA_DIR, "python_basics.md"), "w", encoding="utf-8") as f:
            f.write("# Python Basics\nFunctions group code.\nLists are mutable.\n")

    files = glob.glob(os.path.join(DATA_DIR, "*.md"))
    print("Loading reference documents for advanced ingestion:", files)
    docs = load_docs(files)

    # Use hierarchical chunking strategy (parent-child relationship)
    # Child splitter for embedding generation
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
    child_chunks = child_splitter.split_documents(docs)

    # Attach parent text summary or source indicators to child chunk metadata
    for i, chunk in enumerate(child_chunks):
        chunk.metadata["chunk_id"] = f"child_{i}"
        chunk.metadata["parent_source"] = chunk.metadata.get("source")
        # Simulate parent document context inclusion
        chunk.page_content = f"[Context: {chunk.metadata['parent_source']}] " + chunk.page_content

    print(f"Generated {len(child_chunks)} child chunks with metadata enrichment.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(child_chunks, embedding=embeddings, persist_directory=PERSIST_DIR)
    vectordb.persist()
    print(f"Persisted advanced vector DB to: {PERSIST_DIR}")

if __name__ == "__main__":
    main()
