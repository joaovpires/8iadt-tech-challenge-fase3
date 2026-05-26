"""
Cliente LLM via Ollama (padrão) ou modelo local fine-tunado (opcional).

Usa `langchain-ollama` por padrão. Quando USE_LOCAL_MODEL=true e o adapter
LoRA estiver em outputs/tinyllama-medico-local/, usa o modelo fine-tunado
carregado diretamente via transformers/PEFT.

Pré-requisito (modo Ollama):
    1. Instalar o Ollama: https://ollama.com/download
    2. `ollama pull tinyllama`     (~640 MB — mesma família do fine-tuning)
    3. `ollama serve`              (geralmente roda como serviço automaticamente)

Pré-requisito (modo local):
    1. Executar 03_finetuning_local.ipynb até o fim
    2. Confirmar que outputs/tinyllama-medico-local/adapter_config.json existe
    3. Iniciar o app com: USE_LOCAL_MODEL=true streamlit run app.py
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache

from langchain_ollama import ChatOllama

from src.config import (
    LOCAL_ADAPTER_DIR,
    LLM_NUM_PREDICT,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    USE_LOCAL_MODEL,
)

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_ollama() -> ChatOllama:
    """Singleton do cliente Ollama."""
    log.info("Inicializando ChatOllama: model=%s base=%s", OLLAMA_MODEL, OLLAMA_BASE_URL)
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=LLM_TEMPERATURE,
        num_predict=LLM_NUM_PREDICT,
    )


def get_llm():
    """
    Retorna o cliente LLM ativo:
    - USE_LOCAL_MODEL=true + adapter disponível → ChatHuggingFace (LoRA local)
    - Caso contrário → ChatOllama (padrão)
    """
    # Lê em tempo de execução (não no import) para honrar o toggle da sidebar
    use_local = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"
    if use_local and _adapter_available():
        log.info("Usando modelo fine-tunado local: %s", LOCAL_ADAPTER_DIR)
        from src.llm.local_llm_client import get_local_llm
        return get_local_llm()
    return _get_ollama()


def clear_llm_cache() -> None:
    """Limpa o cache do cliente LLM (chamado ao trocar de modo na sidebar)."""
    _get_ollama.cache_clear()
    try:
        from src.llm.local_llm_client import get_local_llm
        get_local_llm.cache_clear()
    except Exception:  # noqa: BLE001
        pass


def _adapter_available() -> bool:
    required = ["adapter_config.json", "adapter_model.safetensors"]
    return all((LOCAL_ADAPTER_DIR / f).exists() for f in required)


def ping() -> tuple[bool, str]:
    """
    Testa se o backend LLM está respondendo.
    Em modo local, verifica se o adapter existe (sem fazer inferência pesada).
    """
    if USE_LOCAL_MODEL and _adapter_available():
        return True, f"Modelo local: {LOCAL_ADAPTER_DIR.name}"
    try:
        resp = _get_ollama().invoke("Responda apenas: OK")
        return True, (resp.content if hasattr(resp, "content") else str(resp))[:200]
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
