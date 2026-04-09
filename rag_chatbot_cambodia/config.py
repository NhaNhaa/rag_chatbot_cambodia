"""
Configuration constants for Cambodia Tourism RAG Chatbot.
All fixed values are defined here to keep other files clean.
"""

# ========== File paths ==========
DATA_FILE = "data/documents.json"
VECTOR_INDEX_PATH = "vectorstore/index.faiss"
VECTOR_METADATA_PATH = "vectorstore/index.pkl"

# ========== Embedding model ==========
# Using a small, fast sentence-transformer model (works offline after first download)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ========== Text chunking ==========
CHUNK_SIZE = 500        # characters per chunk
CHUNK_OVERLAP = 50      # characters overlap between chunks

# ========== Retrieval ==========
TOP_K_RESULTS = 4       # number of chunks to retrieve per query

# ========== LLM via Groq ==========
# Load API key from environment variable GROQ_API_KEY (use .env file)
# Model is Llama 3 8B (fast, free tier)
LLM_MODEL_NAME = "llama3-8b-8192"
GROQ_API_BASE = "https://api.groq.com/openai/v1"

# ========== Streamlit UI ==========
PAGE_TITLE = "Cambodia Tourism RAG Chatbot"
PAGE_ICON = "🏝️"
DEFAULT_QUESTION = "What is Angkor Wat?"