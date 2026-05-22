"""
Vector store baseado em ChromaDB para os protocolos médicos.

Estratégia:
- Indexa todos os `.md` de `src/data/protocolos/` em uma collection persistente.
- Embeddings via `sentence-transformers` (modelo multilíngue leve).
- Reindexação automática quando o número de arquivos mudar.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownTextSplitter
from langchain_core.documents import Document

from src.config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    PROTOCOLS_DIR,
    RAG_TOP_K,
    VECTORSTORE_DIR,
)

_embeddings: HuggingFaceEmbeddings | None = None
_vectorstore: Chroma | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def _load_protocol_documents() -> List[Document]:
    """Carrega todos os .md de protocolos e retorna chunks prontos."""
    docs: List[Document] = []
    for md_file in sorted(Path(PROTOCOLS_DIR).glob("*.md")):
        loader = TextLoader(str(md_file), encoding="utf-8")
        loaded = loader.load()
        for d in loaded:
            d.metadata["source"] = md_file.name
            d.metadata["protocolo"] = md_file.stem
        docs.extend(loaded)

    splitter = MarkdownTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(docs)


def get_vectorstore(force_rebuild: bool = False) -> Chroma:
    """Retorna a vector store, criando/atualizando se necessário."""
    global _vectorstore

    VECTORSTORE_DIR.mkdir(exist_ok=True)
    embeddings = _get_embeddings()

    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )

    existing_count = store._collection.count()  # noqa: SLF001
    chunks = _load_protocol_documents()

    if force_rebuild or existing_count == 0:
        if existing_count > 0:
            store.delete_collection()
            store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=str(VECTORSTORE_DIR),
            )
        if chunks:
            store.add_documents(chunks)

    _vectorstore = store
    return store


def search_protocols(query: str, k: int = RAG_TOP_K) -> List[Document]:
    """Busca semântica nos protocolos. Retorna chunks ordenados por relevância."""
    store = get_vectorstore()
    return store.similarity_search(query, k=k)


def format_context(docs: List[Document]) -> str:
    """Formata documentos recuperados como bloco de contexto para o prompt."""
    if not docs:
        return "(Nenhum protocolo correspondente encontrado.)"
    blocks = []
    for d in docs:
        src = d.metadata.get("source", "desconhecido")
        blocks.append(f"[FONTE: {src}]\n{d.page_content.strip()}")
    return "\n\n---\n\n".join(blocks)
