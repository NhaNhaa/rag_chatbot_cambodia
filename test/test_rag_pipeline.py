"""
Test suite for rag_pipeline.py – mocking FAISS, Groq, and hybrid retriever.
"""

import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import faiss
import rag_pipeline
from config import TOP_K_RESULTS

def test_build_prompt_with_chunks():
    query = "What is Angkor Wat?"
    chunks = ["Angkor Wat is a temple complex in Cambodia."]
    prompt = rag_pipeline.build_prompt(query, chunks)
    assert "based ONLY on the following" in prompt
    assert "Angkor Wat is a temple complex" in prompt
    assert query in prompt

def test_build_prompt_no_chunks():
    query = "Fake question"
    prompt = rag_pipeline.build_prompt(query, [])
    assert "No relevant information" in prompt
    assert "don't have enough information" in prompt

def test_retrieve_relevant_chunks_pure_faiss():
    """Test pure FAISS retrieval (fallback)."""
    dim = 384
    index = faiss.IndexFlatIP(dim)
    vectors = np.random.rand(10, dim).astype(np.float32)
    faiss.normalize_L2(vectors)
    index.add(vectors)
    metadata = [{"title": f"Doc{i}"} for i in range(10)]
    chunks = [f"chunk{i}" for i in range(10)]
    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.rand(1, dim).astype(np.float32)
    retrieved_chunks, retrieved_metadata, ret_time = rag_pipeline.retrieve_relevant_chunks(
        "test query", index, metadata, chunks, mock_model, top_k=3, use_hybrid=False
    )
    assert len(retrieved_chunks) == 3
    assert len(retrieved_metadata) == 3
    assert ret_time >= 0

def test_retrieve_relevant_chunks_hybrid():
    """Test hybrid retrieval (mocking HybridRetriever)."""
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = (["chunk1", "chunk2"], [{"title": "A"}, {"title": "B"}], [0.9, 0.8])
    with patch("rag_pipeline.get_hybrid_retriever", return_value=mock_retriever):
        retrieved_chunks, retrieved_metadata, ret_time = rag_pipeline.retrieve_relevant_chunks(
            "test query", None, [], [], None, top_k=2, use_hybrid=True
        )
    assert retrieved_chunks == ["chunk1", "chunk2"]
    assert len(retrieved_metadata) == 2
    assert ret_time >= 0

def test_answer_without_rag_mock():
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content="Mock answer"))]
    mock_client.chat.completions.create.return_value = mock_completion
    result = rag_pipeline.answer_without_rag("What is Phnom Penh?", mock_client)
    assert result["answer"] == "Mock answer"
    assert result["num_chunks"] == 0

def test_answer_with_rag_mock():
    dim = 384
    index = faiss.IndexFlatIP(dim)
    vectors = np.random.rand(5, dim).astype(np.float32)
    faiss.normalize_L2(vectors)
    index.add(vectors)
    metadata = [{"title": f"Doc{i}"} for i in range(5)]
    chunks = [f"Content {i}" for i in range(5)]
    mock_embed = MagicMock()
    mock_embed.encode.return_value = np.random.rand(1, dim).astype(np.float32)
    mock_groq = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content="RAG answer"))]
    mock_groq.chat.completions.create.return_value = mock_completion
    # Use pure FAISS to avoid hybrid retriever dependency
    result = rag_pipeline.answer_with_rag(
        "test query", index, metadata, chunks, mock_embed, mock_groq, top_k=2, use_hybrid=False
    )
    assert result["answer"] == "RAG answer"
    assert result["num_chunks"] == 2
    assert "retrieval_time" in result
    assert "generation_time" in result
    assert "total_time" in result