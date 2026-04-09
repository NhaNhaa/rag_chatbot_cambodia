"""
Test suite for config.py – verifies constants exist and have correct types.
"""
import sys
import os
# Add parent directory to path so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import config


def test_constants_exist():
    """Check that all required constants are defined."""
    required_constants = [
        "DATA_FILE",
        "VECTOR_INDEX_PATH",
        "VECTOR_METADATA_PATH",
        "EMBEDDING_MODEL_NAME",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "TOP_K_RESULTS",
        "LLM_MODEL_NAME",
        "GROQ_API_BASE",
        "PAGE_TITLE",
        "PAGE_ICON",
        "DEFAULT_QUESTION",
    ]
    for const_name in required_constants:
        assert hasattr(config, const_name), f"Missing constant: {const_name}"


def test_paths_are_strings():
    """File paths should be non‑empty strings."""
    assert isinstance(config.DATA_FILE, str)
    assert config.DATA_FILE.endswith(".json")
    assert isinstance(config.VECTOR_INDEX_PATH, str)
    assert config.VECTOR_INDEX_PATH.endswith(".faiss")
    assert isinstance(config.VECTOR_METADATA_PATH, str)
    assert config.VECTOR_METADATA_PATH.endswith(".pkl")


def test_numeric_constants():
    """Chunk size, overlap, and top‑k should be positive integers."""
    assert isinstance(config.CHUNK_SIZE, int)
    assert config.CHUNK_SIZE > 0
    assert isinstance(config.CHUNK_OVERLAP, int)
    assert config.CHUNK_OVERLAP >= 0
    assert config.CHUNK_OVERLAP < config.CHUNK_SIZE
    assert isinstance(config.TOP_K_RESULTS, int)
    assert config.TOP_K_RESULTS > 0


def test_model_names_are_strings():
    """Model names should be non‑empty strings."""
    assert isinstance(config.EMBEDDING_MODEL_NAME, str)
    assert len(config.EMBEDDING_MODEL_NAME) > 0
    assert isinstance(config.LLM_MODEL_NAME, str)
    assert len(config.LLM_MODEL_NAME) > 0
    assert isinstance(config.GROQ_API_BASE, str)
    assert config.GROQ_API_BASE.startswith("https://")


def test_ui_constants():
    """UI strings should be non‑empty strings."""
    assert isinstance(config.PAGE_TITLE, str)
    assert len(config.PAGE_TITLE) > 0
    assert isinstance(config.PAGE_ICON, str)
    assert isinstance(config.DEFAULT_QUESTION, str)
    assert len(config.DEFAULT_QUESTION) > 0