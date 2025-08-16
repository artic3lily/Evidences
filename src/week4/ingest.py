import os, glob
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "sample_knowledge")
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "chroma_db")

def load_docs(paths:List[str]):
    docs = []
    for p in paths:
        loader = TextLoader(p, encoding="utf-8")
        docs.extend(loader.load())
    return docs

def main():
    files = glob.glob(os.path.join(DATA_DIR, "*.md"))
    print("Loading files:", files)
    docs = load_docs(files)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=PERSIST_DIR)
    vectordb.persist()
    print(f"Persisted vector DB to: {PERSIST_DIR}")

if __name__ == "__main__":
    main()