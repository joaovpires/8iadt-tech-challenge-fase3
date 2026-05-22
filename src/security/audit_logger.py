"""
Logger estruturado para auditoria de todas as interações.

Cada chamada gera uma linha JSON em `logs/audit.jsonl` com:
- timestamp, paciente_id, pergunta, urgência, fontes, validação, resposta.

Esse log atende ao requisito de "logging detalhado para rastreamento e auditoria".
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict

from src.config import LOGS_DIR

AUDIT_FILE = LOGS_DIR / "audit.jsonl"

# Logger técnico (console)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("assistente_medico")


def log_interaction(event: str, payload: Dict[str, Any]) -> None:
    """Grava um evento de auditoria no JSONL."""
    record = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "event": event,
        **payload,
    }
    try:
        with open(AUDIT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:  # log filesystem issues but don't break the app
        log.warning("Falha ao gravar audit log: %s", exc)


def read_recent(n: int = 20) -> list[dict]:
    """Lê as últimas N interações registradas."""
    if not AUDIT_FILE.exists():
        return []
    with open(AUDIT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [json.loads(l) for l in lines[-n:]]
