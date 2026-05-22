"""
Vector store baseado em FAISS para os protocolos médicos.

Estratégia:
- Indexa todos os `.md` de `src/data/protocolos/` em um índice FAISS persistente.
- Embeddings via `sentence-transformers` (modelo multilíngue leve).
- Reindexação automática quando a quantidade de chunks mudar.

FAISS foi escolhido em vez de Chroma para evitar dependência de
compilador C++ no Windows (chroma-hnswlib não tem wheel pré-compilada).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownTextSplitter

from src.config import (
    EMBEDDING_MODEL,
    PROTOCOLS_DIR,
    RAG_TOP_K,
    VECTORSTORE_DIR,
)

_embeddings: HuggingFaceEmbeddings | None = None
_vectorstore: FAISS | None = None

_INDEX_NAME = "protocolos"


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


def _index_file() -> Path:
    return VECTORSTORE_DIR / f"{_INDEX_NAME}.faiss"


def get_vectorstore(force_rebuild: bool = False) -> FAISS:
    """Retorna a vector store, criando/atualizando se necessário."""
    global _vectorstore

    if _vectorstore is not None and not force_rebuild:
        return _vectorstore

    VECTORSTORE_DIR.mkdir(exist_ok=True)
    embeddings = _get_embeddings()
    chunks = _load_protocol_documents()

    # Tenta carregar índice persistido
    if _index_file().exists() and not force_rebuild:
        try:
            store = FAISS.load_local(
                str(VECTORSTORE_DIR),
                embeddings,
                index_name=_INDEX_NAME,
                allow_dangerous_deserialization=True,
            )
            # Se o nº de chunks mudou, força reindex
            if store.index.ntotal == len(chunks):
                _vectorstore = store
                return _vectorstore
        except Exception:  # noqa: BLE001
            pass  # cai no rebuild abaixo

    if not chunks:
        raise RuntimeError(
            f"Nenhum protocolo encontrado em {PROTOCOLS_DIR}. "
            "Verifique se os arquivos .md estão no diretório."
        )

    _vectorstore = FAISS.from_documents(chunks, embeddings)
    _vectorstore.save_local(str(VECTORSTORE_DIR), index_name=_INDEX_NAME)
    return _vectorstore


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


def index_size() -> int:
    """Retorna o número de vetores no índice (para diagnóstico)."""
    store = get_vectorstore()
    return store.index.ntotal if store.index is not None else 0
