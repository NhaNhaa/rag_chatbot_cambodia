"""
Hybrid retriever: combines FAISS (semantic) and BM25 (keyword) search.
Uses weighted average of normalized scores from both methods.
"""

import math
from typing import List, Tuple, Dict

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import TOP_K_RESULTS, EMBEDDING_MODEL_NAME


class HybridRetriever:
    """
    Hybrid search engine that combines BM25 keyword matching and FAISS semantic similarity.
    """

    def __init__(self, chunks: List[str], metadata: List[Dict], faiss_index: faiss.IndexFlatIP):
        """
        Initialize with text chunks, metadata, and pre-built FAISS index.
        Also builds BM25 index from chunks.
        """
        self.chunks = chunks
        self.metadata = metadata
        self.faiss_index = faiss_index
        self.embed_model = None  # lazy load

        # Tokenize chunks for BM25 (simple whitespace + lowercase)
        tokenized_chunks = [self._tokenize(chunk) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer: lowercase and split on whitespace."""
        return text.lower().split()

    def _get_embed_model(self):
        """Lazy load sentence transformer model."""
        if self.embed_model is None:
            self.embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return self.embed_model

    def search(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
        faiss_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Perform hybrid search.
        Returns:
            - list of chunk texts
            - list of metadata dicts
            - list of combined scores
        """
        # 1. BM25 scores
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)  # raw scores

        # 2. FAISS scores (semantic)
        embed_model = self._get_embed_model()
        query_embedding = embed_model.encode([query])
        faiss.normalize_L2(query_embedding)
        faiss_scores, indices = self.faiss_index.search(query_embedding, len(self.chunks))

        # Convert to 1D array
        faiss_scores = faiss_scores[0]

        # 3. Normalize scores to [0,1] range for fair combination
        norm_bm25 = self._normalize_scores(bm25_scores)
        norm_faiss = self._normalize_scores(faiss_scores)

        # 4. Combine scores
        combined_scores = [
            faiss_weight * norm_faiss[i] + bm25_weight * norm_bm25[i]
            for i in range(len(self.chunks))
        ]

        # 5. Get top_k indices by combined score
        sorted_indices = np.argsort(combined_scores)[::-1][:top_k]

        retrieved_chunks = []
        retrieved_metadata = []
        retrieved_scores = []
        for idx in sorted_indices:
            retrieved_chunks.append(self.chunks[idx])
            retrieved_metadata.append(self.metadata[idx])
            retrieved_scores.append(combined_scores[idx])

        return retrieved_chunks, retrieved_metadata, retrieved_scores

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range using min-max scaling."""
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score == min_score:
            return np.ones_like(scores)  # all equal
        return (scores - min_score) / (max_score - min_score)


# Optional: convenience function to create retriever from existing vectorstore
def create_hybrid_retriever_from_store(
    index: faiss.IndexFlatIP, metadata: List[Dict], chunks: List[str]
) -> HybridRetriever:
    """Factory function to build HybridRetriever from loaded vectorstore components."""
    return HybridRetriever(chunks, metadata, index)