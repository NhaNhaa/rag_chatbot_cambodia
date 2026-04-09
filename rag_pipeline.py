"""
RAG pipeline: retrieve relevant chunks using hybrid search (BM25 + FAISS),
with query expansion and cross‑encoder re‑ranking, then generate answers using Groq LLM.
"""

import time
import re
from typing import List, Tuple, Dict, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os

from config import (
    VECTOR_INDEX_PATH,
    VECTOR_METADATA_PATH,
    EMBEDDING_MODEL_NAME,
    TOP_K_RESULTS,
    LLM_MODEL_NAME,
    FAISS_WEIGHT,
    BM25_WEIGHT,
)
from embedder import load_vectorstore, vectorstore_exists
from hybrid_retriever import HybridRetriever
from metadata_filter import apply_filters
from reranker import Reranker

load_dotenv()

# Global caches
_hybrid_retriever = None
_reranker = None

def get_groq_client():
    """Initialize and return Groq client using API key from environment or Streamlit secrets."""
    import streamlit as st
    import os
    from dotenv import load_dotenv

    api_key = None
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        print(f"✅ Secret loaded, first 10 chars: {api_key[:10]}...")
    except Exception as e:
        print(f"❌ Failed to read st.secrets: {e}")
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            print("✅ API key loaded from .env")
        else:
            print("❌ No API key found in .env")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found in Streamlit secrets or .env file.")
    return Groq(api_key=api_key)

def get_hybrid_retriever():
    """Load or create a singleton HybridRetriever from the vectorstore."""
    global _hybrid_retriever
    if _hybrid_retriever is not None:
        return _hybrid_retriever
    if not vectorstore_exists():
        raise FileNotFoundError("Vectorstore not found. Run embedder.py first.")
    index, metadata, chunks = load_vectorstore()
    if index is None:
        raise ValueError("Failed to load vectorstore.")
    _hybrid_retriever = HybridRetriever(chunks, metadata, index)
    return _hybrid_retriever

def get_reranker():
    """Load or create a singleton Reranker."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker

def expand_query(query: str) -> List[str]:
    """
    Split a complex question into simpler sub‑queries.
    Splits on 'and' and '?' to get atomic questions.
    """
    clean = query.rstrip('?.').strip()
    parts = re.split(r'\s+and\s+|\?', clean)
    sub_queries = [p.strip() + '?' for p in parts if p.strip()]
    if not sub_queries:
        return [query]
    if query not in sub_queries:
        sub_queries.append(query)
    return sub_queries

def retrieve_relevant_chunks(
    query: str,
    index: faiss.IndexFlatIP,
    metadata: List[Dict],
    chunks: List[str],
    model: SentenceTransformer,
    top_k: int,
    use_hybrid: bool = True,
    faiss_weight: float = FAISS_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
    filter_categories: Optional[List[str]] = None,
    filter_sources: Optional[List[str]] = None,
    use_query_expansion: bool = True,
    use_reranker: bool = True,
    rerank_top_k: int = 5,
) -> Tuple[List[str], List[Dict], float]:
    """
    Retrieve top-k most relevant chunks.
    - use_query_expansion: break question into sub‑queries and merge.
    - use_reranker: re‑score the retrieved chunks with a cross‑encoder.
    """
    start_time = time.time()

    # ----- Step 1: retrieve a larger candidate set -----
    candidate_k = top_k * 3 if use_reranker else top_k

    if use_query_expansion:
        sub_queries = expand_query(query)
        all_chunks = []
        all_metadata = []
        all_scores = []

        for sub_q in sub_queries:
            if use_hybrid:
                retriever = get_hybrid_retriever()
                sub_chunks, sub_meta, sub_scores = retriever.search(
                    sub_q, top_k=candidate_k,
                    faiss_weight=faiss_weight, bm25_weight=bm25_weight
                )
            else:
                q_emb = model.encode([sub_q])
                faiss.normalize_L2(q_emb)
                scores, indices = index.search(q_emb, candidate_k)
                sub_chunks = []
                sub_meta = []
                sub_scores = []
                for idx, s in zip(indices[0], scores[0]):
                    if idx != -1 and idx < len(chunks):
                        sub_chunks.append(chunks[idx])
                        sub_meta.append(metadata[idx])
                        sub_scores.append(s)

            # Deduplicate by (title, chunk_index)
            for c, m, s in zip(sub_chunks, sub_meta, sub_scores):
                key = (m.get('title', ''), m.get('chunk_index', 0))
                if key not in [(m2.get('title',''), m2.get('chunk_index',0)) for m2 in all_metadata]:
                    all_chunks.append(c)
                    all_metadata.append(m)
                    all_scores.append(s)

        # Sort by raw score (BM25+FAISS) and take top candidate_k
        if all_scores:
            sorted_indices = np.argsort(all_scores)[::-1][:candidate_k]
            candidate_chunks = [all_chunks[i] for i in sorted_indices]
            candidate_metadata = [all_metadata[i] for i in sorted_indices]
        else:
            candidate_chunks, candidate_metadata = [], []
    else:
        # Single‑query retrieval
        if use_hybrid:
            retriever = get_hybrid_retriever()
            candidate_chunks, candidate_metadata, _ = retriever.search(
                query, top_k=candidate_k, faiss_weight=faiss_weight, bm25_weight=bm25_weight
            )
        else:
            q_emb = model.encode([query])
            faiss.normalize_L2(q_emb)
            scores, indices = index.search(q_emb, candidate_k)
            candidate_chunks = []
            candidate_metadata = []
            for idx in indices[0]:
                if idx != -1 and idx < len(chunks):
                    candidate_chunks.append(chunks[idx])
                    candidate_metadata.append(metadata[idx])

    # ----- Step 2: re‑rank with cross‑encoder -----
    if use_reranker and candidate_chunks:
        reranker = get_reranker()
        reranked_chunks, reranked_scores = reranker.rerank(query, candidate_chunks, top_k=rerank_top_k)
        # Align metadata with reranked chunks
        chunk_to_meta = {chunk: meta for chunk, meta in zip(candidate_chunks, candidate_metadata)}
        reranked_metadata = [chunk_to_meta[chunk] for chunk in reranked_chunks if chunk in chunk_to_meta]
        retrieved_chunks = reranked_chunks[:top_k]
        retrieved_metadata = reranked_metadata[:top_k]
    else:
        retrieved_chunks = candidate_chunks[:top_k]
        retrieved_metadata = candidate_metadata[:top_k]

    # ----- Step 3: apply filters (if any) -----
    if filter_categories or filter_sources:
        retrieved_metadata, retrieved_chunks = apply_filters(
            retrieved_metadata, retrieved_chunks,
            source_types=filter_sources,
            categories=filter_categories
        )

    retrieval_time = time.time() - start_time
    return retrieved_chunks, retrieved_metadata, retrieval_time

def build_prompt(query: str, retrieved_chunks: List[str]) -> str:
    """Build prompt instructing LLM to answer based ONLY on retrieved context, allowing partial answers."""
    if not retrieved_chunks:
        return f"""You are a helpful assistant for Cambodia tourism.

Question: {query}

No relevant information was found in the knowledge base. Please say:
"I don't have enough information to answer that question about Cambodia tourism."

Answer based ONLY on the above. Do not make up facts."""
    context = "\n\n---\n\n".join(retrieved_chunks)
    prompt = f"""You are a helpful assistant for Cambodia tourism. Answer the question based ONLY on the following retrieved information. If the information is present, answer directly. If only part of the question can be answered, provide that partial answer and note what is missing. Do not refuse to answer if some relevant information exists.

Retrieved information:
{context}

Question: {query}

Answer (using only the retrieved information):"""
    return prompt

def generate_answer(prompt: str, client: Groq, model_name: str) -> Tuple[str, float]:
    """Send prompt to Groq LLM and return answer + generation time."""
    start_time = time.time()
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=800,
        )
        answer = completion.choices[0].message.content
        generation_time = time.time() - start_time
        return answer.strip(), generation_time
    except Exception as error:
        print(f"LLM error: {error}")
        return f"Error generating answer: {error}", 0.0

def answer_with_rag(
    query: str,
    index: faiss.IndexFlatIP,
    metadata: List[Dict],
    chunks: List[str],
    embed_model: SentenceTransformer,
    groq_client: Groq,
    top_k: int = TOP_K_RESULTS,
    use_hybrid: bool = True,
    use_query_expansion: bool = True,
    use_reranker: bool = True,
    filter_categories: Optional[List[str]] = None,
    filter_sources: Optional[List[str]] = None,
) -> Dict:
    """Complete RAG pipeline: retrieve (with expansion & reranking) -> filter -> prompt -> generate."""
    retrieved_chunks, retrieved_metadata, retrieval_time = retrieve_relevant_chunks(
        query, index, metadata, chunks, embed_model, top_k,
        use_hybrid=use_hybrid,
        faiss_weight=FAISS_WEIGHT,
        bm25_weight=BM25_WEIGHT,
        filter_categories=filter_categories,
        filter_sources=filter_sources,
        use_query_expansion=use_query_expansion,
        use_reranker=use_reranker,
        rerank_top_k=top_k,
    )
    prompt = build_prompt(query, retrieved_chunks)
    answer, generation_time = generate_answer(prompt, groq_client, LLM_MODEL_NAME)
    return {
        "query": query,
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "retrieved_metadata": retrieved_metadata,
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "total_time": retrieval_time + generation_time,
        "num_chunks": len(retrieved_chunks),
    }

def answer_without_rag(query: str, groq_client: Groq) -> Dict:
    """Direct LLM answer without retrieval (to demonstrate hallucination)."""
    start_time = time.time()
    prompt = f"You are a helpful assistant. Answer this question about Cambodia tourism: {query}"
    try:
        completion = groq_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )
        answer = completion.choices[0].message.content
        generation_time = time.time() - start_time
        return {
            "query": query,
            "answer": answer,
            "generation_time": generation_time,
            "num_chunks": 0,
        }
    except Exception as error:
        return {"query": query, "answer": f"Error: {error}", "generation_time": 0.0, "num_chunks": 0}

if __name__ == "__main__":
    if not vectorstore_exists():
        print("Vectorstore not found. Run embedder.py first.")
        exit(1)
    index, metadata, chunks = load_vectorstore()
    if index is None:
        print("Failed to load vectorstore.")
        exit(1)
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    try:
        groq_client = get_groq_client()
    except ValueError as e:
        print(e)
        exit(1)
    test_query = "How did the seasonal flooding of the Tonlé Sap lake influence both the agriculture of the Khmer Empire and the location of major temples like Angkor Wat and Bayon?"
    print(f"\nQuery: {test_query}")
    result = answer_with_rag(
        test_query, index, metadata, chunks, embed_model, groq_client,
        use_hybrid=True, use_query_expansion=True, use_reranker=True
    )
    print(f"\nAnswer: {result['answer']}")
    print(f"\nRetrieved {result['num_chunks']} chunks in {result['retrieval_time']:.2f}s")
    print(f"Generated in {result['generation_time']:.2f}s")
    print("\nSources:")
    for meta in result['retrieved_metadata']:
        print(f"  - {meta['title']}")