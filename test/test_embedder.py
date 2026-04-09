"""
Test suite for embedder.py – chunking, metadata creation, and FAISS operations (mocked).
Updated to check for new metadata fields: source_type and category.
"""

import sys
import os
import json
import tempfile
import pickle
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import faiss
import embedder
from config import CHUNK_SIZE, CHUNK_OVERLAP


def test_chunk_text_basic():
    text = "a" * 1000
    chunks = embedder.chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) >= 3
    assert len(chunks) <= 5
    assert all(isinstance(c, str) for c in chunks)


def test_chunk_text_empty():
    assert embedder.chunk_text("", 100, 10) == []


def test_chunk_text_overlap():
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = embedder.chunk_text(text, chunk_size=10, overlap=5)
    assert chunks[0] == text[0:10]
    assert chunks[1] == text[5:15]
    assert chunks[0][-5:] == chunks[1][:5]


def test_load_raw_documents_success():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        test_docs = [{"title": "A", "url": "x", "content": "hello"}]
        json.dump(test_docs, tmp)
        tmp_path = tmp.name
    docs = embedder.load_raw_documents(tmp_path)
    assert docs == test_docs
    os.unlink(tmp_path)


def test_load_raw_documents_not_found():
    docs = embedder.load_raw_documents("nonexistent.json")
    assert docs == []


def test_infer_category():
    assert embedder.infer_category("Angkor Wat") == "temple"
    assert embedder.infer_category("Phnom Penh") == "city"
    assert embedder.infer_category("History of Cambodia") == "history"
    assert embedder.infer_category("Cambodian cuisine") == "culture"
    assert embedder.infer_category("Tonle Sap") == "geography"
    assert embedder.infer_category("Random Topic") == "general"


def test_create_chunks_from_documents():
    docs = [
        {"title": "Angkor Wat", "url": "http://a.com", "content": "abcde" * 100},
        {"title": "Phnom Penh", "url": "http://b.com", "content": "12345" * 100},
    ]
    metadata, chunks = embedder.create_chunks_from_documents(docs, chunk_size=100, overlap=20)
    assert len(metadata) == len(chunks)
    assert len(chunks) >= 8
    # Check new metadata fields
    assert "source_type" in metadata[0]
    assert metadata[0]["source_type"] == "wikipedia"
    assert "category" in metadata[0]
    assert metadata[0]["category"] in ["temple", "city", "history", "culture", "geography", "general"]
    assert metadata[0]["title"] == "Angkor Wat"
    assert metadata[0]["chunk_index"] == 0


def test_build_faiss_index_mock():
    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.rand(5, 384).astype(np.float32)
    with patch("embedder.SentenceTransformer", return_value=mock_model):
        chunks = ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]
        index, embeddings = embedder.build_faiss_index(chunks, "mock-model")
    assert index.ntotal == 5
    assert embeddings.shape == (5, 384)


def test_save_and_load_vectorstore():
    dimension = 128
    index = faiss.IndexFlatIP(dimension)
    vectors = np.random.rand(10, dimension).astype(np.float32)
    faiss.normalize_L2(vectors)
    index.add(vectors)

    metadata = [{"title": f"Doc{i}", "chunk_index": i, "source_type": "wikipedia", "category": "general"} for i in range(10)]
    chunks = [f"chunk{i}" for i in range(10)]

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("embedder.VECTOR_INDEX_PATH", os.path.join(tmpdir, "index.faiss")):
            with patch("embedder.VECTOR_METADATA_PATH", os.path.join(tmpdir, "index.pkl")):
                embedder.save_vectorstore(index, metadata, chunks)
                loaded_index, loaded_metadata, loaded_chunks = embedder.load_vectorstore()
                assert loaded_index.ntotal == 10
                assert loaded_metadata == metadata
                assert loaded_chunks == chunks


def test_vectorstore_exists():
    with tempfile.TemporaryDirectory() as tmpdir:
        idx_path = os.path.join(tmpdir, "index.faiss")
        pkl_path = os.path.join(tmpdir, "index.pkl")
        with patch("embedder.VECTOR_INDEX_PATH", idx_path):
            with patch("embedder.VECTOR_METADATA_PATH", pkl_path):
                assert embedder.vectorstore_exists() is False
                open(idx_path, "w").close()
                open(pkl_path, "w").close()
                assert embedder.vectorstore_exists() is True