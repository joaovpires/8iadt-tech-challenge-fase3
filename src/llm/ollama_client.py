"""
Cliente LLM via Ollama.

Usa `langchain-ollama` que conversa com o servidor local do Ollama
(`http://localhost:11434` por padrão).

Pré-requisito:
    1. Instalar o Ollama: https://ollama.com/download
    2. `ollama pull tinyllama`     (~640 MB — mesma família do fine-tuning)
    3. `ollama serve`              (geralmente roda como serviço automaticamente)
"""
from __future__ import annotations

import logging
from functools import lru_cache

from langchain_ollama import ChatOllama

from src.config import LLM_NUM_PREDICT, LLM_TEMPERATURE, OLLAMA_BASE_URL, OLLAMA_MODEL

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    """Retorna um cliente ChatOllama configurado (singleton)."""
    log.info("Inicializando ChatOllama: model=%s base=%s", OLLAMA_MODEL, OLLAMA_BASE_URL)
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=LLM_TEMPERATURE,
        num_predict=LLM_NUM_PREDICT,
    )


def ping() -> tuple[bool, str]:
    """Testa se o Ollama está respondendo."""
    try:
        resp = get_llm().invoke("Responda apenas: OK")
        return True, (resp.content if hasattr(resp, "content") else str(resp))[:200]
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
