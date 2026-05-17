from __future__ import annotations

import json
from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from src.config import RagSettings, build_chat_model
from src.retrieval import HybridRetriever


class RagState(TypedDict, total=False):
    question: str
    rewritten_question: str
    documents: list[Document]
    graded_documents: list[Document]
    context: str
    answer: str
    citations: list[dict[str, Any]]
    grounded: bool
    confidence: float
    retry_count: int
    messages: list[str]


REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user question into a precise search query for document retrieval. "
            "Preserve named entities, numbers, and constraints. Return only the query.",
        ),
        ("human", "{question}"),
    ]
)

GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You grade whether a document chunk is useful for answering a question. "
            "Return compact JSON with keys relevant: boolean and reason: string.",
        ),
        ("human", "Question:\n{question}\n\nDocument chunk:\n{document}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful RAG assistant. Answer only from the provided context. "
            "Cite sources inline like [source p.page] or [source chunk N]. "
            "If the context is insufficient, say what is missing and do not invent facts.",
        ),
        ("human", "Question:\n{question}\n\nContext:\n{context}"),
    ]
)

VALIDATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Check if the answer is grounded in the context. Return compact JSON with "
            "keys grounded: boolean, confidence: number from 0 to 1, reason: string.",
        ),
        ("human", "Context:\n{context}\n\nAnswer:\n{answer}"),
    ]
)


def build_rag_graph(settings: RagSettings, retriever: HybridRetriever):
    llm = build_chat_model(settings)
    text_parser = StrOutputParser()

    def rewrite_query(state: RagState) -> RagState:
        question = state["question"].strip()
        try:
            rewritten = (REWRITE_PROMPT | llm | text_parser).invoke({"question": question})
        except Exception:
            rewritten = question
        return {"rewritten_question": rewritten.strip() or question, "retry_count": 0}

    def retrieve(state: RagState) -> RagState:
        query = state.get("rewritten_question") or state["question"]
        return {"documents": retriever.retrieve(query)}

    def grade_documents(state: RagState) -> RagState:
        question = state["question"]
        graded: list[Document] = []

        for doc in state.get("documents", []):
            if float(doc.metadata.get("retrieval_score", 0.0)) >= settings.relevance_threshold:
                graded.append(doc)
                continue

            try:
                raw = (GRADE_PROMPT | llm | text_parser).invoke(
                    {"question": question, "document": doc.page_content[:2200]}
                )
                payload = _parse_json(raw)
                if payload.get("relevant") is True:
                    graded.append(doc)
            except Exception:
                if question.lower() in doc.page_content.lower():
                    graded.append(doc)

        return {"graded_documents": graded or state.get("documents", [])[:3]}

    def assemble_context(state: RagState) -> RagState:
        docs = state.get("graded_documents", [])
        context_blocks = []
        citations = []
        for index, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "uploaded document")
            page = doc.metadata.get("page")
            chunk_id = doc.metadata.get("chunk_id")
            label = f"{source} p.{page}" if page else f"{source} chunk {chunk_id}"
            context_blocks.append(f"[{index}] {label}\n{doc.page_content}")
            citations.append(
                {
                    "source": source,
                    "page": page,
                    "chunk_id": chunk_id,
                    "score": round(float(doc.metadata.get("retrieval_score", 0.0)), 3),
                }
            )
        return {"context": "\n\n".join(context_blocks), "citations": citations}

    def generate_answer(state: RagState) -> RagState:
        if not state.get("context"):
            return {
                "answer": "I could not find relevant context in the uploaded documents.",
                "confidence": 0.0,
                "grounded": False,
            }

        answer = (ANSWER_PROMPT | llm | text_parser).invoke(
            {"question": state["question"], "context": state["context"]}
        )
        return {"answer": answer}

    def validate_answer(state: RagState) -> RagState:
        if not state.get("answer") or not state.get("context"):
            return {"grounded": False, "confidence": 0.0}

        try:
            raw = (VALIDATE_PROMPT | llm | text_parser).invoke(
                {"context": state["context"], "answer": state["answer"]}
            )
            payload = _parse_json(raw)
            return {
                "grounded": bool(payload.get("grounded", False)),
                "confidence": float(payload.get("confidence", 0.5)),
            }
        except Exception:
            score = 0.65 if state.get("citations") else 0.3
            return {"grounded": score >= 0.6, "confidence": score}

    def tighten_context(state: RagState) -> RagState:
        retry_count = int(state.get("retry_count", 0)) + 1
        docs = state.get("graded_documents", [])[:3]
        return {"graded_documents": docs, "retry_count": retry_count}

    def should_retry(state: RagState) -> str:
        if state.get("grounded") or int(state.get("retry_count", 0)) >= 1:
            return "done"
        return "retry"

    graph = StateGraph(RagState)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("assemble_context", assemble_context)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("validate_answer", validate_answer)
    graph.add_node("tighten_context", tighten_context)

    graph.add_edge(START, "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_edge("grade_documents", "assemble_context")
    graph.add_edge("assemble_context", "generate_answer")
    graph.add_edge("generate_answer", "validate_answer")
    graph.add_conditional_edges(
        "validate_answer",
        should_retry,
        {"retry": "tighten_context", "done": END},
    )
    graph.add_edge("tighten_context", "assemble_context")
    return graph.compile()


def _parse_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.removeprefix("json").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start : end + 1]
    return json.loads(text)
