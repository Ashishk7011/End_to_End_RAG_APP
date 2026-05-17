from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


def collection_name_for(documents: list[Document]) -> str:
    digest = hashlib.sha1()
    for doc in documents:
        digest.update(doc.page_content[:500].encode("utf-8", errors="ignore"))
        digest.update(str(doc.metadata).encode("utf-8", errors="ignore"))
    return f"rag_{digest.hexdigest()[:16]}"


class HybridRetriever:
    def __init__(
        self,
        *,
        documents: list[Document],
        embeddings,
        persist_directory: str,
        top_k: int,
    ) -> None:
        self.documents = documents
        self.top_k = top_k
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            ids=[_doc_key(document) for document in documents],
            collection_name=collection_name_for(documents),
            persist_directory=persist_directory,
        )
        self.keyword_retriever = BM25Retriever.from_documents(documents)
        self.keyword_retriever.k = top_k

    def retrieve(self, query: str) -> list[Document]:
        vector_hits = self.vectorstore.similarity_search_with_relevance_scores(
            query,
            k=self.top_k,
        )
        keyword_hits = self.keyword_retriever.invoke(query)

        ranked: dict[str, Document] = {}
        for rank, (doc, score) in enumerate(vector_hits):
            merged = _clone_document(doc)
            merged.metadata["retrieval_score"] = float(score)
            merged.metadata["retrieval_rank"] = rank + 1
            ranked[_doc_key(merged)] = merged

        for rank, doc in enumerate(keyword_hits):
            key = _doc_key(doc)
            if key in ranked:
                ranked[key].metadata["keyword_rank"] = rank + 1
            else:
                merged = _clone_document(doc)
                merged.metadata["keyword_rank"] = rank + 1
                merged.metadata["retrieval_score"] = 0.0
                ranked[key] = merged

        return sorted(
            ranked.values(),
            key=lambda item: (
                -float(item.metadata.get("retrieval_score", 0.0)),
                int(item.metadata.get("keyword_rank", 999)),
            ),
        )[: self.top_k]


def _doc_key(doc: Document) -> str:
    return f"{doc.metadata.get('source')}:{doc.metadata.get('page')}:{doc.metadata.get('chunk_id')}"


def _clone_document(doc: Document) -> Document:
    if hasattr(doc, "model_copy"):
        return doc.model_copy(deep=True)
    return doc.copy(deep=True)
