"""
Fluxo de triagem clínica automatizada com LangGraph.

Etapas:
    sintomas → classificar_urgencia → verificar_exames_pendentes
            → consultar_protocolos → gerar_conduta → alerta_equipe (se grave)

Estado é um dicionário tipado que percorre os nós, acumulando informação.
"""
from __future__ import annotations

from typing import List, Literal, Optional, TypedDict

from langgraph.graph import END, StateGraph

from src.chains.medical_chain import ask
from src.rag.patient_db import get_patient, overdue_exams


# ───────────────────────── Estado ─────────────────────────
class TriageState(TypedDict, total=False):
    # entrada
    sintomas: str
    paciente_id: Optional[str]

    # derivado
    urgencia: Literal["BAIXA", "MEDIA", "ALTA"]
    motivo_urgencia: str
    exames_pendentes: List[dict]
    conduta: str
    fontes: List[str]
    alerta: Optional[str]
    trace: List[str]


# ───────────────────────── Heurística simples de urgência ─────────────────────────
RED_FLAGS = [
    "dor torácica", "dor toracica", "falta de ar", "dispneia", "dispnéia",
    "perda de consciência", "perda de consciencia", "convulsão", "convulsao",
    "sangramento", "hemorragia", "rebaixamento", "déficit neurológico",
    "deficit neurologico", "paralisia", "anafilaxia", "trauma grave",
]
YELLOW_FLAGS = [
    "febre alta", "vômito persistente", "vomito persistente", "dor intensa",
    "tontura", "desidratação", "desidratacao",
]


def _append_trace(state: TriageState, msg: str) -> List[str]:
    trace = list(state.get("trace", []))
    trace.append(msg)
    return trace


# ───────────────────────── Nós ─────────────────────────
def node_classify_urgency(state: TriageState) -> TriageState:
    s = state["sintomas"].lower()
    if any(rf in s for rf in RED_FLAGS):
        urg, motivo = "ALTA", "Sinal de alarme (red flag) identificado nos sintomas."
    elif any(yf in s for yf in YELLOW_FLAGS):
        urg, motivo = "MEDIA", "Sintomas sugestivos de avaliação prioritária."
    else:
        urg, motivo = "BAIXA", "Sem red flags evidentes na descrição."

    return {
        **state,
        "urgencia": urg,
        "motivo_urgencia": motivo,
        "trace": _append_trace(state, f"[classify_urgency] urgência={urg}"),
    }


def node_check_pending_exams(state: TriageState) -> TriageState:
    pid = state.get("paciente_id")
    pendentes: List[dict] = []
    if pid:
        patient = get_patient(pid)
        if patient:
            pendentes = overdue_exams(patient, days_threshold=60)
    return {
        **state,
        "exames_pendentes": pendentes,
        "trace": _append_trace(
            state, f"[check_pending_exams] {len(pendentes)} exame(s) em atraso"
        ),
    }


def node_consult_assistant(state: TriageState) -> TriageState:
    pergunta = (
        f"Triagem clínica.\n"
        f"Sintomas relatados: {state['sintomas']}\n"
        f"Urgência classificada: {state.get('urgencia')}.\n"
        f"Sugira conduta inicial baseada nos protocolos disponíveis."
    )
    resp = ask(pergunta, patient_id=state.get("paciente_id"))
    return {
        **state,
        "conduta": resp.answer,
        "fontes": resp.sources,
        "trace": _append_trace(state, f"[consult_assistant] {len(resp.sources)} fonte(s)"),
    }


def node_alert_team(state: TriageState) -> TriageState:
    alerta = None
    if state.get("urgencia") == "ALTA":
        alerta = (
            "🚨 ALERTA CRÍTICO: caso com sinais de alarme. "
            "Acionar equipe médica imediatamente para avaliação presencial."
        )
    return {
        **state,
        "alerta": alerta,
        "trace": _append_trace(state, f"[alert_team] alerta={'sim' if alerta else 'não'}"),
    }


# ───────────────────────── Roteamento condicional ─────────────────────────
def route_after_urgency(state: TriageState) -> str:
    # qualquer urgência segue pela mesma cadeia; mantemos o nó para evidenciar a decisão
    return "check_pending_exams"


# ───────────────────────── Build do grafo ─────────────────────────
def build_triage_graph():
    g = StateGraph(TriageState)

    g.add_node("classify_urgency", node_classify_urgency)
    g.add_node("check_pending_exams", node_check_pending_exams)
    g.add_node("consult_assistant", node_consult_assistant)
    g.add_node("alert_team", node_alert_team)

    g.set_entry_point("classify_urgency")
    g.add_conditional_edges("classify_urgency", route_after_urgency)
    g.add_edge("check_pending_exams", "consult_assistant")
    g.add_edge("consult_assistant", "alert_team")
    g.add_edge("alert_team", END)

    return g.compile()


_graph = None


def run_triage(sintomas: str, paciente_id: Optional[str] = None) -> TriageState:
    """Executa o fluxo completo de triagem e retorna o estado final."""
    global _graph
    if _graph is None:
        _graph = build_triage_graph()
    initial: TriageState = {"sintomas": sintomas, "paciente_id": paciente_id, "trace": []}
    return _graph.invoke(initial)
