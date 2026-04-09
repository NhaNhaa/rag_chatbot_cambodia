# 🏝️ Cambodia Tourism RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about Cambodia tourism using **Wikipedia articles, province data, and travel FAQs**. The system retrieves relevant information from a local knowledge base and generates grounded answers with **source transparency** — reducing hallucinations compared to plain LLM calls.

## 🎯 Purpose

Tourists often struggle to find reliable, consolidated information about Cambodia. This chatbot provides **fact‑based answers** using retrieved documents, and it demonstrates:

- RAG architecture (retrieval + generation)
- Hybrid search (BM25 + FAISS)
- Query expansion & cross‑encoder reranking
- Streamlit UI with example questions and source visibility
- Comparison of **RAG vs. no‑RAG** to show hallucination reduction

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | FAISS (local) |
| Keyword search | BM25 via `rank-bm25` |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Groq (`openai/gpt-oss-120b` or any Groq model) |
| UI | Streamlit |
| Data sources | Wikipedia API, static CSV (provinces), synthetic FAQ |

## 📁 Project Structure

```
rag_chatbot_cambodia/
├── advanced_data_loader.py   # Fetches extra Wikipedia pages (Khmer Empire, Tonlé Sap, etc.)
├── app.py                    # Streamlit UI – user interface, no settings, just ask
├── config.py                 # Constants: paths, chunk size, model names, hybrid weights
├── csv_loader.py             # Loads province data (CSV) into documents.json
├── data/                     # Directory for raw data files
│   ├── documents.json        # All knowledge base documents (Wikipedia, FAQ, CSV)
│   ├── faq.txt               # Simple travel FAQ (used by advanced_data_loader)
│   └── provinces.csv         # Province list with population, capital, area (from official stats)
├── data_loader.py            # Basic Wikipedia fetcher (extract API) and JSON save/load
├── embedder.py               # Chunks documents, generates embeddings, builds FAISS index
├── hybrid_retriever.py       # Combines BM25 (keyword) + FAISS (semantic) search
├── metadata_filter.py        # Filters retrieved chunks by source_type or category
├── rag_pipeline.py           # Core RAG pipeline: retrieval → reranking → prompt → LLM generation
├── README.md                 # Project documentation (setup, usage, architecture)
├── requirements.txt          # Python dependencies
├── reranker.py               # Cross‑encoder reranker to reorder retrieved chunks by relevance
├── test/                     # Unit tests for all modules (pytest)
│   ├── test_config.py
│   ├── test_data_loader.py
│   ├── test_embedder.py
│   ├── test_hybrid_retriever.py
│   ├── test_metadata_filter.py
│   ├── test_rag_pipeline.py
│   └── __init__.py
└── __init__.py               # Makes the root a Python package (optional)

```

## 🚀 Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/NhaNhaa/rag_chatbot_cambodia.git
cd rag_chatbot_cambodia
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your Groq API key

- Get a free API key from [Groq Console](https://console.groq.com)
- Copy `.env.example` to `.env` and add your key:

```bash
cp .env.example .env
# Edit .env and paste your GROQ_API_KEY
```

### 5. (Optional) Build or refresh the knowledge base

The repository already contains pre‑built `data/documents.json` and `vectorstore/`. To rebuild from scratch:

```bash
python data_loader.py          # fetch basic Wikipedia articles
python advanced_data_loader.py # add extra topics & FAQ
python csv_loader.py           # add province data
python embedder.py             # chunk, embed, build FAISS index
```

### 6. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## 🧪 Example Questions

| Category | Question |
|----------|----------|
| Temple | *"What is the significance of the bas‑reliefs at Angkor Wat?"* |
| Geography | *"Why does the Tonlé Sap lake reverse its flow each year?"* |
| History | *"What was the Khmer Empire, and during which centuries did it flourish?"* |
| Cuisine | *"What are three traditional Cambodian dishes and their main ingredients?"* |
| Province | *"Which province has the smallest population, and what is its capital?"* |
| Culture | *"What is Pchum Ben, and when is it celebrated?"* |
| Modern | *"What was the Khmer Rouge, and what impact did it have on Cambodia?"* |
| Comparison | *"How does Bayon temple differ architecturally from Angkor Wat?"* |
| Travel | *"Best time to visit Cambodia and what currency to use?"* |
| Causal | *"How did Tonlé Sap flooding support the construction of Angkor?"* |

## 🧠 How It Works

1. **User asks a question** (via Streamlit).
2. **Query expansion** splits complex questions into sub‑queries (e.g., on “and” or “?”).
3. **Hybrid retrieval**:
   - BM25 (keyword) scores chunks.
   - FAISS (semantic) scores chunks.
   - Scores are normalized and combined (weights from `config.py`).
4. **Reranking**: a small cross‑encoder re‑scores the top candidates for better relevance.
5. **Prompt construction**: retrieved chunks are inserted into a strict prompt that instructs the LLM to answer **only from the provided text**.
6. **LLM generation**: Groq’s model (configurable) produces the final answer.
7. **Display**: answer, performance metrics, and collapsible sources are shown.

## 📈 Performance & Improvements

- **Hallucination reduction**: The LLM correctly says “I don’t have enough information” when the answer is missing.
- **Speed**: Retrieval typically < 3 seconds, generation < 2 seconds (depends on model).
- **Accuracy**: Verified on 10+ hard questions (history, geography, cuisine, comparisons).

## 🧪 Testing

Run the test suite:

```bash
pytest test/ -v
```

Tests cover chunking, metadata inference, hybrid search, reranking, and the RAG pipeline (mocked).

## 🌐 Deployment (Streamlit Cloud)

1. Push your code to GitHub.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud), connect your repo.
3. Set `GROQ_API_KEY` as a secret in the app settings.
4. Deploy. The app will be live with a public URL.

> **Note**: The `vectorstore/` folder is excluded from git (binary files). Streamlit Cloud will rebuild it on first run if you include `data/documents.json`. Alternatively, you can pre‑build and commit the index (not recommended for large files).

## 📝 License

This project is for educational and portfolio purposes. Feel free to use and adapt it.

## 🙏 Acknowledgements

- Wikipedia for the articles
- Groq for free LLM access
- Sentence‑Transformers & FAISS communities

## 👤 Author

PANHA LIM – [GitHub](https://github.com/NhaNhaa)

---