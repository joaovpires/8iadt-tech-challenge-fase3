"""
Guardrails simples — validação pós-resposta da LLM.

Objetivo: bloquear respostas que tentem prescrever de forma definitiva ou
omitam o aviso de validação médica.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from src.config import BLOCKED_PATTERNS, DISCLAIMER

PRESCRIPTION_HINTS = [
    r"\btome\s+\d+",
    r"\buse\s+\d+\s*mg\b",
    r"\bdose\s+definitiva\b",
    r"\bprescrev[oa]\b\s+\w+",
]


@dataclass
class ValidationResult:
    ok: bool
    issues: List[str]
    fixed_answer: str


def validate(answer: str) -> ValidationResult:
    """Valida e ajusta a resposta da LLM."""
    issues: List[str] = []
    fixed = answer.strip()

    # 1) Padrões bloqueados (configuração)
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, fixed, flags=re.IGNORECASE):
            issues.append(f"Padrão bloqueado detectado: {pattern}")

    # 2) Heurística de prescrição direta
    for pattern in PRESCRIPTION_HINTS:
        if re.search(pattern, fixed, flags=re.IGNORECASE):
            issues.append(f"Possível prescrição direta: {pattern}")

    # 3) Garantir disclaimer
    if "médic" not in fixed.lower() and "medic" not in fixed.lower():
        fixed = f"{fixed}\n\n{DISCLAIMER}"
        issues.append("Disclaimer ausente — adicionado automaticamente.")

    # Se houver problemas críticos, substituir por mensagem segura
    has_critical = any("bloqueado" in i or "prescrição direta" in i for i in issues)
    if has_critical:
        fixed = (
            "⚠️ Resposta bloqueada pelo módulo de segurança.\n\n"
            "O conteúdo gerado continha indicação direta de prescrição/dose, "
            "o que viola as regras do assistente. Reformule a pergunta solicitando "
            "orientação geral baseada em protocolo ou consulte o médico assistente.\n\n"
            f"{DISCLAIMER}"
        )

    return ValidationResult(ok=not has_critical, issues=issues, fixed_answer=fixed)
