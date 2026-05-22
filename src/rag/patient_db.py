"""Acesso à base estruturada de pacientes (JSON sintético)."""
from __future__ import annotations

import json
from datetime import date, datetime
from typing import Dict, List, Optional

from src.config import PATIENTS_FILE

_cache: Optional[List[Dict]] = None


def _load() -> List[Dict]:
    global _cache
    if _cache is None:
        with open(PATIENTS_FILE, "r", encoding="utf-8") as f:
            _cache = json.load(f)
    return _cache


def list_patients() -> List[Dict]:
    """Retorna todos os pacientes (campos resumidos)."""
    return [
        {"id": p["id"], "nome": p["nome"], "idade": p["idade"], "sexo": p["sexo"]}
        for p in _load()
    ]


def get_patient(patient_id: str) -> Optional[Dict]:
    """Busca paciente por ID."""
    for p in _load():
        if p["id"].upper() == patient_id.upper():
            return p
    return None


def format_patient_context(patient: Dict) -> str:
    """Formata dados do paciente como contexto textual para o LLM."""
    if not patient:
        return "(Paciente não informado.)"

    comorb = ", ".join(patient.get("comorbidades", [])) or "nenhuma"
    alerg = ", ".join(patient.get("alergias", [])) or "nenhuma conhecida"
    meds = ", ".join(patient.get("medicacoes_uso", [])) or "nenhuma"

    pendentes = patient.get("exames_pendentes", [])
    pend_str = (
        "\n".join(f"  - {e['exame']} (solicitado em {e['solicitado_em']})" for e in pendentes)
        if pendentes
        else "  (nenhum)"
    )

    ultimos = patient.get("ultimos_exames", [])
    ult_str = (
        "\n".join(f"  - {e['exame']}: {e['valor']} ({e['data']})" for e in ultimos)
        if ultimos
        else "  (nenhum registrado)"
    )

    return (
        f"PACIENTE: {patient['nome']} (ID {patient['id']})\n"
        f"Idade: {patient['idade']} | Sexo: {patient['sexo']}\n"
        f"Comorbidades: {comorb}\n"
        f"Alergias: {alerg}\n"
        f"Medicações em uso: {meds}\n"
        f"Exames pendentes:\n{pend_str}\n"
        f"Últimos exames:\n{ult_str}\n"
        f"Histórico: {patient.get('historico_resumo', '-')}"
    )


def overdue_exams(patient: Dict, days_threshold: int = 60) -> List[Dict]:
    """Lista exames solicitados há mais de X dias sem resultado."""
    overdue = []
    today = date.today()
    for e in patient.get("exames_pendentes", []):
        try:
            req = datetime.strptime(e["solicitado_em"], "%Y-%m-%d").date()
            if (today - req).days >= days_threshold:
                overdue.append({**e, "dias_em_aberto": (today - req).days})
        except (ValueError, KeyError):
            continue
    return overdue
