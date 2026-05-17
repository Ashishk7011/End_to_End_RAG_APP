from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from src.config import RagSettings, build_embeddings
from src.document_loader import load_uploaded_files, split_documents
from src.rag_graph import build_rag_graph
from src.retrieval import HybridRetriever


APP_DIR = Path(__file__).parent
INDEX_DIR = APP_DIR / ".rag_index"


st.set_page_config(
    page_title="Advanced LangGraph RAG",
    page_icon="magnifying_glass_tilted_right",
    layout="wide",
)


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("retriever", None)
    st.session_state.setdefault("graph", None)
    st.session_state.setdefault("indexed_file_names", [])


def provider_defaults(provider: str) -> tuple[str, str]:
    if provider == "OpenAI":
        return "gpt-4.1-mini", "text-embedding-3-small"
    return "qwen2.5", "nomic-embed-text"


@st.cache_resource(show_spinner=False)
def build_index(
    file_signature: tuple[str, ...],
    settings: RagSettings,
    _uploaded_files,
) -> tuple[HybridRetriever, int]:
    raw_docs = load_uploaded_files(_uploaded_files)
    chunks = split_documents(
        raw_docs,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    embeddings = build_embeddings(settings)
    retriever = HybridRetriever(
        documents=chunks,
        embeddings=embeddings,
        persist_directory=str(INDEX_DIR),
        top_k=settings.top_k,
    )
    return retriever, len(chunks)


init_state()

with st.sidebar:
    st.title("RAG Control")

    provider = st.selectbox("Model provider", ["OpenAI", "Ollama"])
    default_chat, default_embedding = provider_defaults(provider)
    chat_model = st.text_input("Chat model", value=default_chat)
    embedding_model = st.text_input("Embedding model", value=default_embedding)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)

    st.divider()
    uploaded_files = st.file_uploader(
        "Knowledge files",
        type=["pdf", "txt", "md", "markdown", "docx"],
        accept_multiple_files=True,
    )
    chunk_size = st.slider("Chunk size", 400, 1800, 900, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, 25)
    top_k = st.slider("Retrieved chunks", 2, 12, 6, 1)
    relevance_threshold = st.slider("Vector relevance threshold", 0.0, 1.0, 0.18, 0.01)

    settings = RagSettings(
        provider=provider,
        chat_model=chat_model,
        embedding_model=embedding_model,
        temperature=temperature,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        relevance_threshold=relevance_threshold,
    )

    index_clicked = st.button("Build knowledge index", type="primary", use_container_width=True)
    if provider == "OpenAI" and not os.getenv("OPENAI_API_KEY"):
        st.warning("Set OPENAI_API_KEY in .env or your environment before querying.")

st.title("Advanced LangGraph RAG")
st.caption("Upload documents, build an index, then ask grounded questions with citations.")

if index_clicked:
    if not uploaded_files:
        st.sidebar.error("Upload at least one supported file first.")
    else:
        signature = tuple(f"{file.name}:{file.size}" for file in uploaded_files)
        with st.spinner("Embedding documents and compiling the RAG graph..."):
            try:
                retriever, chunk_count = build_index(signature, settings, uploaded_files)
                st.session_state.retriever = retriever
                st.session_state.graph = build_rag_graph(settings, retriever)
                st.session_state.indexed_file_names = [file.name for file in uploaded_files]
                st.sidebar.success(f"Indexed {chunk_count} chunks from {len(uploaded_files)} files.")
            except Exception as exc:
                st.sidebar.error(f"Indexing failed: {exc}")

if st.session_state.indexed_file_names:
    with st.expander("Indexed sources", expanded=False):
        for name in st.session_state.indexed_file_names:
            st.write(f"- {name}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("citations"):
            with st.expander("Sources"):
                st.dataframe(message["citations"], use_container_width=True)

question = st.chat_input("Ask a question about your uploaded documents")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if st.session_state.graph is None:
        answer = "Please upload documents and build the knowledge index first."
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
    else:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.spinner("Running the LangGraph RAG workflow..."):
                result = st.session_state.graph.invoke({"question": question})

            answer = result.get("answer", "I could not produce an answer.")
            confidence = result.get("confidence", 0.0)
            grounded = result.get("grounded", False)
            citations = result.get("citations", [])

            placeholder.markdown(answer)
            st.caption(f"Grounded: {'yes' if grounded else 'uncertain'} | Confidence: {confidence:.2f}")
            if citations:
                with st.expander("Sources", expanded=True):
                    st.dataframe(citations, use_container_width=True)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "citations": citations,
            }
        )
