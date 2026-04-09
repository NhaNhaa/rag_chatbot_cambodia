"""
Test suite for hybrid_retriever.py – verifies BM25 + FAISS combination.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import faiss
from hybrid_retriever import HybridRetriever


def test_tokenize():
    # Use a non-empty chunks list to avoid BM25 zero division
    chunks = ["dummy chunk"]  # BM25 requires at least one document
    retriever = HybridRetriever(chunks, [{"title": "dummy"}], None)
    tokens = retriever._tokenize("Hello, World! How are you?")
    assert tokens == ["hello,", "world!", "how", "are", "you?"]


def test_normalize_scores():
    scores = np.array([10, 20, 30])
    normalized = HybridRetriever._normalize_scores(scores)
    expected = np.array([0.0, 0.5, 1.0])
    assert np.allclose(normalized, expected)


def test_normalize_scores_constant():
    scores = np.array([5, 5, 5])
    normalized = HybridRetriever._normalize_scores(scores)
    assert np.allclose(normalized, [1.0, 1.0, 1.0])


def test_hybrid_search_mock():
    # Create dummy chunks
    chunks = [
        "Angkor Wat is a temple in Cambodia.",
        "Phnom Penh is the capital city.",
        "The Mekong River flows through Cambodia.",
    ]
    metadata = [{"title": "Angkor Wat"}, {"title": "Phnom Penh"}, {"title": "Mekong"}]

    # Create dummy FAISS index
    dim = 384
    index = faiss.IndexFlatIP(dim)
    vectors = np.random.rand(len(chunks), dim).astype(np.float32)
    faiss.normalize_L2(vectors)
    index.add(vectors)

    retriever = HybridRetriever(chunks, metadata, index)

    # Mock the embedding model to return fixed vector for query
    class MockModel:
        def encode(self, queries):
            return np.random.rand(1, dim).astype(np.float32)

    retriever.embed_model = MockModel()

    # Test search
    retrieved_chunks, retrieved_metadata, scores = retriever.search(
        "temple", top_k=2, faiss_weight=0.7, bm25_weight=0.3
    )
    assert len(retrieved_chunks) == 2
    assert len(retrieved_metadata) == 2
    assert len(scores) == 2
    assert all(isinstance(s, float) for s in scores)


def test_hybrid_search_weights():
    """Test that BM25 and FAISS contributions are combined as expected."""
    # Simple controlled test with known BM25 and FAISS scores
    chunks = ["apple banana", "banana cherry", "dog cat"]
    metadata = [{"title": "A"}, {"title": "B"}, {"title": "C"}]

    # Create dummy FAISS index with pre-defined scores via fixed embeddings
    dim = 2
    index = faiss.IndexFlatIP(dim)
    # We'll set embeddings so that FAISS scores for query "apple" are: [high, low, low]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.5]], dtype=np.float32)
    faiss.normalize_L2(vectors)
    index.add(vectors)

    retriever = HybridRetriever(chunks, metadata, index)

    # Mock embedding model to return query embedding that matches first vector
    class MockModel:
        def encode(self, queries):
            # Return embedding that will give highest dot with first vector
            return np.array([[1.0, 0.0]], dtype=np.float32)

    retriever.embed_model = MockModel()

    # Query "apple" – BM25 should favor first chunk (contains "apple")
    retrieved_chunks, _, scores = retriever.search("apple", top_k=3, faiss_weight=0.5, bm25_weight=0.5)
    # First chunk should have highest combined score
    assert retrieved_chunks[0] == chunks[0]