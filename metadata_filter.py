"""
Metadata filtering utilities for RAG retrieval.
Allows filtering chunks by source_type (e.g., 'wikipedia', 'faq') or category (e.g., 'temple', 'history').
"""

from typing import List, Dict, Optional


def filter_by_source_type(
    metadata_list: List[Dict], chunks: List[str], allowed_sources: List[str]
) -> tuple[List[Dict], List[str]]:
    """
    Keep only chunks whose source_type is in allowed_sources.
    Returns filtered (metadata, chunks) pairs in the same order.
    """
    filtered_meta = []
    filtered_chunks = []
    for meta, chunk in zip(metadata_list, chunks):
        if meta.get("source_type", "unknown") in allowed_sources:
            filtered_meta.append(meta)
            filtered_chunks.append(chunk)
    return filtered_meta, filtered_chunks


def filter_by_category(
    metadata_list: List[Dict], chunks: List[str], allowed_categories: List[str]
) -> tuple[List[Dict], List[str]]:
    """
    Keep only chunks whose category is in allowed_categories.
    """
    filtered_meta = []
    filtered_chunks = []
    for meta, chunk in zip(metadata_list, chunks):
        if meta.get("category", "general") in allowed_categories:
            filtered_meta.append(meta)
            filtered_chunks.append(chunk)
    return filtered_meta, filtered_chunks


def apply_filters(
    metadata_list: List[Dict],
    chunks: List[str],
    source_types: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
) -> tuple[List[Dict], List[str]]:
    """
    Apply both source_type and category filters (AND logic).
    If a filter list is None or empty, no filtering is applied for that dimension.
    """
    if source_types:
        metadata_list, chunks = filter_by_source_type(metadata_list, chunks, source_types)
    if categories:
        metadata_list, chunks = filter_by_category(metadata_list, chunks, categories)
    return metadata_list, chunks