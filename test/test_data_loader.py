"""
Test suite for data_loader.py – verifies fetching and saving logic.
Uses mocking to avoid hitting Wikipedia network during tests.
"""

import sys
import os
import json
import tempfile
from unittest.mock import patch, Mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import data_loader
from config import DATA_FILE


def test_fetch_wikipedia_page_success():
    """Mock a successful Wikipedia API response."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "query": {
            "pages": {
                "123": {
                    "pageid": 123,
                    "title": "Angkor Wat",
                    "extract": "Angkor Wat is a temple complex in Cambodia."
                }
            }
        }
    }

    with patch("requests.get", return_value=mock_response):
        result = data_loader.fetch_wikipedia_page("Angkor Wat")

    assert result["title"] == "Angkor Wat"
    assert "temple complex" in result["content"]
    assert result["url"] == "https://en.wikipedia.org/wiki/Angkor_Wat"


def test_fetch_wikipedia_page_not_found():
    """Wikipedia returns page ID -1 when page doesn't exist."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "query": {
            "pages": {
                "-1": {"title": "Fake Page", "missing": ""}
            }
        }
    }

    with patch("requests.get", return_value=mock_response):
        result = data_loader.fetch_wikipedia_page("Fake Page")

    assert result == {}


def test_fetch_wikipedia_page_network_error():
    """Network error should return empty dict."""
    with patch("requests.get", side_effect=Exception("Connection error")):
        result = data_loader.fetch_wikipedia_page("Angkor Wat")
    assert result == {}


def test_fetch_all_topics():
    """fetch_all_topics should call fetch for each topic and return list."""
    def mock_fetch(title):
        if "Wat" in title:
            return {"title": title, "url": "http://example.com", "content": "Some content"}
        return {}

    with patch("data_loader.fetch_wikipedia_page", side_effect=mock_fetch):
        topics = ["Angkor Wat", "Fake Topic", "Another Wat"]
        docs = data_loader.fetch_all_topics(topics)

    assert len(docs) == 2
    assert docs[0]["title"] == "Angkor Wat"
    assert docs[1]["title"] == "Another Wat"


def test_save_and_load_documents():
    """Test saving to JSON and loading back."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    test_docs = [
        {"title": "Doc1", "url": "http://a.com", "content": "Hello"},
        {"title": "Doc2", "url": "http://b.com", "content": "World"},
    ]

    success = data_loader.save_documents(test_docs, tmp_path)
    assert success is True

    loaded = data_loader.load_documents_from_file(tmp_path)
    assert loaded == test_docs

    os.unlink(tmp_path)


def test_load_documents_file_not_found():
    """Loading a missing file returns empty list."""
    result = data_loader.load_documents_from_file("does_not_exist.json")
    assert result == []


def test_load_documents_corrupted_json():
    """If JSON is malformed, return empty list."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        tmp.write("this is not json")
        tmp_path = tmp.name

    result = data_loader.load_documents_from_file(tmp_path)
    assert result == []
    os.unlink(tmp_path)


def test_save_documents_empty_list():
    """save_documents should return False and not raise error for empty list."""
    result = data_loader.save_documents([], "any.json")
    assert result is False