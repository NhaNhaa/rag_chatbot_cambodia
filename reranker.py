"""
Re-ranker for retrieved chunks using a tiny cross-encoder model.
Uses a small model (~17 MB) to fit in limited disk space.
"""

from typing import List, Tuple
import numpy as np
from sentence_transformers import CrossEncoder


class Reranker:
    """
    Re-ranks a list of chunks based on their semantic relevance to the query.
    Uses a very small cross-encoder model to save disk space.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2"):
        """
        Initialize the cross-encoder model (TinyBERT, ~17 MB).
        """
        print(f"Loading tiny reranker model: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, chunks: List[str], top_k: int = 10
    ) -> Tuple[List[str], List[float]]:
        """
        Re-rank the list of chunks by relevance to the query.
        Returns the top_k chunks (sorted) and their scores.
        """
        if not chunks:
            return [], []

        pairs = [[query, chunk] for chunk in chunks]
        scores = self.model.predict(pairs)
        sorted_indices = np.argsort(scores)[::-1]
        top_indices = sorted_indices[:top_k]

        reranked_chunks = [chunks[i] for i in top_indices]
        reranked_scores = [float(scores[i]) for i in top_indices]

        return reranked_chunks, reranked_scores