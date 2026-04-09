"""
Configuration constants for Cambodia Tourism RAG Chatbot.
All fixed values are defined here to keep other files clean.
"""

# ========== File paths ==========
DATA_FILE = "data/documents.json"
VECTOR_INDEX_PATH = "vectorstore/index.faiss"
VECTOR_METADATA_PATH = "vectorstore/index.pkl"

# ========== Embedding model ==========
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ========== Text chunking ==========
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

# ========== Retrieval ==========
TOP_K_RESULTS = 5

# ========== LLM via Groq ==========
LLM_MODEL_NAME = "openai/gpt-oss-120b"
GROQ_API_BASE = "https://api.groq.com/openai/v1"

# ========== Streamlit UI ==========
PAGE_TITLE = "Cambodia Tourism RAG Chatbot"
PAGE_ICON = "🏝️"
DEFAULT_QUESTION = "What is Angkor Wat?"

# ========== Hybrid search weights ==========
FAISS_WEIGHT = 0.3
BM25_WEIGHT = 0.7