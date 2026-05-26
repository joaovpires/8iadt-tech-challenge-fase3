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


_KEYWORD_PROTOCOL_MAP = {
    # ── Sepse ────────────────────────────────────────────────────────────
    "sepse": "sepse.md",
    "séptico": "sepse.md",
    "séptica": "sepse.md",
    "septico": "sepse.md",
    "septica": "sepse.md",
    "qsofa": "sepse.md",
    "lactato": "sepse.md",
    "bundle": "sepse.md",
    "noradrenalina": "sepse.md",
    "choque séptico": "sepse.md",
    "choque septico": "sepse.md",
    "vasopressor": "sepse.md",
    "hemocultura": "sepse.md",
    # sintomas vitais clássicos de sepse (usados pela triagem)
    "hipotensão": "sepse.md",
    "hipotensao": "sepse.md",
    "taquicardia": "sepse.md",
    # ── AVE / AVC ─────────────────────────────────────────────────────────
    "ave ": "ave_isquemico.md",
    "avc ": "ave_isquemico.md",
    " ave": "ave_isquemico.md",
    " avc": "ave_isquemico.md",
    "isquêmico": "ave_isquemico.md",
    "isquemico": "ave_isquemico.md",
    "acidente vascular": "ave_isquemico.md",
    "nihss": "ave_isquemico.md",
    "trombólise": "ave_isquemico.md",
    "trombolise": "ave_isquemico.md",
    "rt-pa": "ave_isquemico.md",
    "fast ": "ave_isquemico.md",
    "afasia": "ave_isquemico.md",
    "paralisia facial": "ave_isquemico.md",
    "déficit neurológico": "ave_isquemico.md",
    "deficit neurologico": "ave_isquemico.md",
    # ── Cetoacidose ───────────────────────────────────────────────────────
    "cetoacidose": "cetoacidose.md",
    "cad": "cetoacidose.md",
    "insulina": "cetoacidose.md",
    "cetonemia": "cetoacidose.md",
    "cetonúria": "cetoacidose.md",
    "cetonuria": "cetoacidose.md",
    "glicemia alta": "cetoacidose.md",
    "hiperglicemia": "cetoacidose.md",
    "diabético": "cetoacidose.md",
    "diabetico": "cetoacidose.md",
    "diabetes descompensado": "cetoacidose.md",
    "bicarbonato baixo": "cetoacidose.md",
    # ── Dor Torácica ──────────────────────────────────────────────────────
    "dor torácica": "dor_toracica.md",
    "dor toracica": "dor_toracica.md",
    "troponina": "dor_toracica.md",
    "ecg": "dor_toracica.md",
    "infarto": "dor_toracica.md",
    "sca": "dor_toracica.md",
    "angina": "dor_toracica.md",
    "supra de st": "dor_toracica.md",
    "supradesnivelamento": "dor_toracica.md",
    "dissecção aórtica": "dor_toracica.md",
    "disseccao aortica": "dor_toracica.md",
    "tromboembolismo": "dor_toracica.md",
    "tep": "dor_toracica.md",
    # ── Crise Hipertensiva ────────────────────────────────────────────────
    "hipertensiva": "crise_hipertensiva.md",
    "hipertensão": "crise_hipertensiva.md",
    "hipertensao": "crise_hipertensiva.md",
    "pressão alta": "crise_hipertensiva.md",
    "pressao alta": "crise_hipertensiva.md",
    "pa elevada": "crise_hipertensiva.md",
    "encefalopatia hipertensiva": "crise_hipertensiva.md",
    # ── Antibioticoterapia ────────────────────────────────────────────────
    "antibiótico": "antibioticoterapia.md",
    "antibiotico": "antibioticoterapia.md",
    "antimicrobiano": "antibioticoterapia.md",
    "pneumonia": "antibioticoterapia.md",
    "infecção urinária": "antibioticoterapia.md",
    "infeccao urinaria": "antibioticoterapia.md",
    "penicilina": "antibioticoterapia.md",
    "ceftriaxona": "antibioticoterapia.md",
    "vancomicina": "antibioticoterapia.md",
    "piperacilina": "antibioticoterapia.md",
    "resistência bacteriana": "antibioticoterapia.md",
    "desescalonamento": "antibioticoterapia.md",
    # ── Prevenção / Exames ────────────────────────────────────────────────
    "preventivo": "prevencao_exames.md",
    "colonoscopia": "prevencao_exames.md",
    "rastreamento": "prevencao_exames.md",
    "exame de rotina": "prevencao_exames.md",
    "mamografia": "prevencao_exames.md",
    "pap": "prevencao_exames.md",
    "preventiva": "prevencao_exames.md",
}


def search_protocols(query: str, k: int = RAG_TOP_K) -> List[Document]:
    """Busca semântica nos protocolos com boost por palavras-chave."""
    store = get_vectorstore()
    docs = store.similarity_search(query, k=k)

    # Garante inclusão do protocolo correto quando palavra-chave é detectada
    q_lower = query.lower()
    forced_sources = {
        proto for kw, proto in _KEYWORD_PROTOCOL_MAP.items() if kw in q_lower
    }
    existing_sources = {d.metadata.get("source") for d in docs}
    missing = forced_sources - existing_sources

    if missing:
        all_chunks = _load_protocol_documents()
        for src in missing:
            candidates = [c for c in all_chunks if c.metadata.get("source") == src]
            if candidates:
                docs = [candidates[0]] + docs  # insere na frente (maior prioridade)

    return docs


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
