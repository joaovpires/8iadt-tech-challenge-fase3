"""
Configuração central do assistente médico.
Todos os caminhos e parâmetros ficam aqui para facilitar manutenção.
"""
from __future__ import annotations

import os
from pathlib import Path

# ───────────────────────── Paths ─────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "src" / "data"
PROTOCOLS_DIR = DATA_DIR / "protocolos"
PATIENTS_FILE = DATA_DIR / "pacientes.json"
VECTORSTORE_DIR = ROOT_DIR / ".chroma"
LOGS_DIR = ROOT_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# ───────────────────────── LLM (Ollama) ─────────────────────────
# Modelo da mesma família do fine-tuning (TinyLlama 1.1B).
# Para baixar: `ollama pull tinyllama`
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "tinyllama")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT", "400"))

# ───────────────────────── Embeddings / RAG ─────────────────────────
# Modelo leve e multilíngue (PT/EN), ~120MB, roda em CPU.
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
COLLECTION_NAME = "protocolos_medicos"

# ───────────────────────── Segurança ─────────────────────────
# Termos que disparam recusa imediata (nunca prescrever).
BLOCKED_PATTERNS = [
    r"\bprescrev[ao]\b",
    r"\breceit[ao]\b.*\bobrigat",
    r"\bdose\s+exata\b",
]

DISCLAIMER = (
    "⚠️ **Aviso:** Este assistente é uma ferramenta de apoio educacional. "
    "Toda conduta clínica deve ser validada pelo médico responsável."
)
