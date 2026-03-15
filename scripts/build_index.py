import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
load_dotenv()

from ingestion import load_documents
from chunking import split_documents
from embeddings import get_embedding_model
from vector_store import create_vector_store
from config import FAISS_INDEX_PATH


def build():
    with open("data/urls.txt") as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"Loading {len(urls)} URLs...")
    docs = load_documents(urls)
    print(f"Loaded {len(docs)} documents. Chunking...")

    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks. Building index...")

    embeddings = get_embedding_model()
    vs = create_vector_store(embeddings)
    vs.add_documents(chunks)

    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    vs.save_local(FAISS_INDEX_PATH)
    print(f"Index saved to {FAISS_INDEX_PATH}")


if __name__ == "__main__":
    build()
