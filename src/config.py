from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

Provider = Literal["OpenAI", "Ollama"]


@dataclass(frozen=True)
class RagSettings:
    provider: Provider
    chat_model: str
    embedding_model: str
    temperature: float = 0.1
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 6
    relevance_threshold: float = 0.18


def build_chat_model(settings: RagSettings):
    if settings.provider == "OpenAI":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.chat_model, temperature=settings.temperature)

    from langchain_ollama import ChatOllama

    return ChatOllama(model=settings.chat_model, temperature=settings.temperature)


def build_embeddings(settings: RagSettings):
    if settings.provider == "OpenAI":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.embedding_model)

    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=settings.embedding_model)
