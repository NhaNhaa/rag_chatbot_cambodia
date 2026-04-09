"""
Streamlit UI for Cambodia Tourism RAG Chatbot – Super simple, no confusing settings.
Just type a question or click an example.
Uses advanced RAG: hybrid search + query expansion + reranking.
"""

import streamlit as st

from config import PAGE_TITLE, PAGE_ICON, DEFAULT_QUESTION, TOP_K_RESULTS, LLM_MODEL_NAME
from embedder import load_vectorstore, vectorstore_exists
from rag_pipeline import answer_with_rag, answer_without_rag, get_groq_client

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="centered")

st.title(f"{PAGE_ICON} {PAGE_TITLE}")
st.markdown("Ask anything about **Cambodia** – I'll search my knowledge base and give you a reliable answer.")

with st.expander("ℹ️ How this works (click to read)"):
    st.markdown("""
    - I look up information from **Wikipedia articles, travel FAQs, and province data** about Cambodia.
    - Then I use an AI to write an answer **based only on what I found**.
    - This means I won't make up fake facts.
    - **No settings to worry about** – just ask!
    """)

# ========== Question input ==========
if "input_question" not in st.session_state:
    st.session_state.input_question = DEFAULT_QUESTION

query = st.text_area(
    "📝 **Your question**",
    value=st.session_state.input_question,
    height=100,
    placeholder="e.g., What is Angkor Wat? Where can I try Cambodian food?",
    label_visibility="visible",
    key="question_input"
)

col_ask, _ = st.columns([1, 5])
with col_ask:
    ask_button = st.button("🔍 Ask now", type="primary", use_container_width=True)

# ========== Example questions (click to fill AND ask) ==========
st.markdown("---")
st.markdown("### 🧪 Try these examples (click any)")
example_questions = [
    "What is Angkor Wat?",
    "Where is Phnom Penh?",
    "Tell me about Cambodian cuisine.",
    "What are the major provinces of Cambodia?",
]

cols = st.columns(2)
for i, q in enumerate(example_questions):
    col_idx = i % 2
    if cols[col_idx].button(q, key=f"ex_{i}", use_container_width=True):
        st.session_state.input_question = q
        st.session_state.auto_ask = True
        st.rerun()

st.markdown("---")

# ========== Determine whether to process ==========
should_process = False
if ask_button and query.strip():
    should_process = True
elif st.session_state.get("auto_ask", False) and st.session_state.input_question.strip():
    should_process = True
    st.session_state.auto_ask = False

if should_process:
    @st.cache_resource
    def load_model():
        from sentence_transformers import SentenceTransformer
        with st.spinner("🔄 Loading AI model (first time only, ~10 sec)..."):
            return SentenceTransformer("all-MiniLM-L6-v2")

    @st.cache_resource
    def load_index():
        if not vectorstore_exists():
            return None, None, None
        return load_vectorstore()

    @st.cache_resource
    def load_groq():
        try:
            return get_groq_client()
        except Exception:
            return None

    embed_model = load_model()
    index, metadata, chunks = load_index()
    groq_client = load_groq()

    if groq_client is None:
        st.error("❌ Missing API key. Please add GROQ_API_KEY to Streamlit secrets.")
        st.stop()

    if index is None:
        st.error("❌ Knowledge base not found. Run `python embedder.py` first.")
        st.stop()

    with st.spinner("🔍 Searching knowledge base and writing answer..."):
        result = answer_with_rag(
            query=st.session_state.input_question.strip(),
            index=index,
            metadata=metadata,
            chunks=chunks,
            embed_model=embed_model,
            groq_client=groq_client,
            top_k=TOP_K_RESULTS,
            use_hybrid=True,
            use_query_expansion=True,
            use_reranker=True,
            filter_categories=None,
            filter_sources=None,
        )

    st.markdown("### ✅ Answer")
    st.write(result["answer"])

    if result["num_chunks"] > 0:
        with st.expander("📚 Sources I used (click to see the original text)"):
            for i, meta in enumerate(result["retrieved_metadata"]):
                st.markdown(f"**Source {i+1}:** *{meta['title']}*")
                st.caption(f"Category: {meta.get('category', 'general')}")
                st.markdown(result["retrieved_chunks"][i])
                st.divider()
    else:
        st.info("ℹ️ No relevant information found. Try rephrasing your question.")

    with st.expander("⚡ Performance (how fast it worked)"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Search time", f"{result['retrieval_time']:.2f} sec")
        col2.metric("AI writing time", f"{result['generation_time']:.2f} sec")
        col3.metric("Chunks used", result["num_chunks"])

elif (ask_button or st.session_state.get("auto_ask", False)) and not st.session_state.input_question.strip():
    st.warning("⚠️ Please type a question first.")
    if st.session_state.get("auto_ask", False):
        st.session_state.auto_ask = False

st.markdown("---")
st.caption(f"📖 **Knowledge sources:** Wikipedia articles, Cambodia Travel FAQ, province data | 🤖 AI model: {LLM_MODEL_NAME} via Groq | 🔎 Search: hybrid + query expansion + reranking")