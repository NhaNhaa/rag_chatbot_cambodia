"""
Test suite for metadata_filter.py.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import metadata_filter


def test_filter_by_source_type():
    metadata = [
        {"source_type": "wikipedia", "title": "A"},
        {"source_type": "faq", "title": "B"},
        {"source_type": "wikipedia", "title": "C"},
    ]
    chunks = ["chunk A", "chunk B", "chunk C"]
    filtered_meta, filtered_chunks = metadata_filter.filter_by_source_type(
        metadata, chunks, ["wikipedia"]
    )
    assert len(filtered_meta) == 2
    assert filtered_meta[0]["title"] == "A"
    assert filtered_meta[1]["title"] == "C"
    assert filtered_chunks == ["chunk A", "chunk C"]


def test_filter_by_category():
    metadata = [
        {"category": "temple", "title": "A"},
        {"category": "city", "title": "B"},
        {"category": "temple", "title": "C"},
    ]
    chunks = ["chunk A", "chunk B", "chunk C"]
    filtered_meta, filtered_chunks = metadata_filter.filter_by_category(
        metadata, chunks, ["temple"]
    )
    assert len(filtered_meta) == 2
    assert filtered_meta[0]["title"] == "A"
    assert filtered_meta[1]["title"] == "C"
    assert filtered_chunks == ["chunk A", "chunk C"]


def test_apply_filters_both():
    metadata = [
        {"source_type": "wikipedia", "category": "temple", "title": "A"},
        {"source_type": "faq", "category": "temple", "title": "B"},
        {"source_type": "wikipedia", "category": "city", "title": "C"},
    ]
    chunks = ["chunk A", "chunk B", "chunk C"]
    filtered_meta, filtered_chunks = metadata_filter.apply_filters(
        metadata, chunks, source_types=["wikipedia"], categories=["temple"]
    )
    assert len(filtered_meta) == 1
    assert filtered_meta[0]["title"] == "A"
    assert filtered_chunks == ["chunk A"]


def test_apply_filters_empty_allowed():
    # Empty allowed list should return nothing
    metadata = [{"source_type": "wikipedia", "title": "A"}]
    chunks = ["chunk A"]
    filtered_meta, filtered_chunks = metadata_filter.filter_by_source_type(
        metadata, chunks, []
    )
    assert filtered_meta == []
    assert filtered_chunks == []