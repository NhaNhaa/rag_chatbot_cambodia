"""
Create embeddings and FAISS vector index from raw documents.
Saves index to vectorstore/ folder for later retrieval.
Now includes metadata: source_type (e.g., 'wikipedia'), category (e.g., 'temple').
"""

import json
import pickle
from typing import List, Dict, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import (
    DATA_FILE,
    VECTOR_INDEX_PATH,
    VECTOR_METADATA_PATH,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping chunks by character count.
    Returns list of text chunks.
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)

        if end == text_length:
            break

        start = start + chunk_size - overlap

    return chunks


def load_raw_documents(json_path: str) -> List[Dict]:
    """
    Load documents from JSON file created by data_loader.py.
    Each dict has 'title', 'url', 'content'.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as file:
            docs = json.load(file)
        if isinstance(docs, list):
            return docs
        else:
            print(f"Error: {json_path} does not contain a list.")
            return []
    except FileNotFoundError:
        print(f"Error: {json_path} not found. Run data_loader.py first.")
        return []
    except json.JSONDecodeError:
        print(f"Error: {json_path} is corrupted.")
        return []


def infer_category(title: str) -> str:
    """
    Infer category from document title using keyword matching.
    Returns one of: 'temple', 'city', 'history', 'culture', 'geography', 'general'
    """
    title_lower = title.lower()
    if any(keyword in title_lower for keyword in ["angkor", "wat", "temple", "bayon", "preah vihear"]):
        return "temple"
    if any(keyword in title_lower for keyword in ["phnom penh", "siem reap", "kampot", "city"]):
        return "city"
    if "history" in title_lower:
        return "history"
    if any(keyword in title_lower for keyword in ["cuisine", "food", "culture"]):
        return "culture"
    if any(keyword in title_lower for keyword in ["tonle sap", "lake", "river", "mountain"]):
        return "geography"
    return "general"


def create_chunks_from_documents(
    documents: List[Dict], chunk_size: int, overlap: int
) -> Tuple[List[Dict], List[str]]:
    """
    Convert each document into overlapping text chunks.
    Returns:
        - metadata list: each item has 'title', 'url', 'chunk_index', 'source_doc',
          'source_type' (always 'wikipedia' for now), 'category'
        - chunk_texts list: plain text strings
    """
    all_metadata = []
    all_chunks = []

    for doc in documents:
        title = doc.get("title", "Unknown")
        url = doc.get("url", "")
        content = doc.get("content", "")
        if not content:
            continue

        category = infer_category(title)

        chunks = chunk_text(content, chunk_size, overlap)

        for idx, chunk in enumerate(chunks):
            all_metadata.append({
                "title": title,
                "url": url,
                "chunk_index": idx,
                "source_doc": title,
                "source_type": "wikipedia",   # all current docs are from Wikipedia
                "category": category,
            })
            all_chunks.append(chunk)

    return all_metadata, all_chunks


def build_faiss_index(
    chunks: List[str], model_name: str
) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    """
    Create embeddings for all chunks and build FAISS inner product index.
    Returns (faiss_index, embeddings_array).
    """
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Creating embeddings for {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print(f"FAISS index built with {index.ntotal} vectors")
    return index, embeddings


def save_vectorstore(index: faiss.IndexFlatIP, metadata: List[Dict], chunks: List[str]):
    """
    Save FAISS index and metadata to vectorstore/ directory.
    """
    try:
        faiss.write_index(index, VECTOR_INDEX_PATH)
        print(f"Saved FAISS index to {VECTOR_INDEX_PATH}")

        with open(VECTOR_METADATA_PATH, "wb") as f:
            pickle.dump({"metadata": metadata, "chunks": chunks}, f)
        print(f"Saved metadata to {VECTOR_METADATA_PATH}")
    except Exception as error:
        print(f"Error saving vectorstore: {error}")


def load_vectorstore() -> Tuple[faiss.IndexFlatIP, List[Dict], List[str]]:
    """
    Load previously saved FAISS index and metadata.
    Returns (index, metadata, chunks) or (None, [], []) if not found.
    """
    try:
        index = faiss.read_index(VECTOR_INDEX_PATH)
        with open(VECTOR_METADATA_PATH, "rb") as f:
            data = pickle.load(f)
        metadata = data.get("metadata", [])
        chunks = data.get("chunks", [])
        print(f"Loaded FAISS index with {index.ntotal} vectors")
        print(f"Loaded {len(metadata)} metadata entries")
        return index, metadata, chunks
    except FileNotFoundError:
        print("Vectorstore not found. Run embedder.py first to build it.")
        return None, [], []
    except Exception as error:
        print(f"Error loading vectorstore: {error}")
        return None, [], []


def vectorstore_exists() -> bool:
    """Check if both FAISS index and metadata pickle exist."""
    from pathlib import Path
    return Path(VECTOR_INDEX_PATH).exists() and Path(VECTOR_METADATA_PATH).exists()


# ========== Main execution ==========
if __name__ == "__main__":
    print("=== Cambodia Tourism RAG – Embedding Builder ===\n")

    docs = load_raw_documents(DATA_FILE)
    if not docs:
        print("No documents found. Exiting.")
        exit(1)

    print(f"Loaded {len(docs)} documents.")

    metadata, chunks = create_chunks_from_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    if not chunks:
        print("No chunks created. Exiting.")
        exit(1)

    index, embeddings = build_faiss_index(chunks, EMBEDDING_MODEL_NAME)
    save_vectorstore(index, metadata, chunks)

    print("\n✓ Vectorstore build complete.")