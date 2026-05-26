"""
Chain principal do assistente médico:
- Recupera protocolos relevantes (RAG).
- Carrega contexto do paciente (se houver).
- Chama a LLM com prompt estruturado.
- Retorna resposta + fontes (explainability).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.config import DISCLAIMER
from src.llm.ollama_client import get_llm
from src.rag.patient_db import format_patient_context, get_patient
from src.rag.vectorstore import format_context, search_protocols
from src.security.audit_logger import log_interaction
from src.security.guardrails import validate

SYSTEM_PROMPT = """Você é um assistente clínico de apoio à decisão médica.

REGRAS INEGOCIÁVEIS:
1. NUNCA prescreva medicação, dose ou esquema terapêutico de forma definitiva.
2. SEMPRE indique que a conduta final é responsabilidade do médico assistente.
3. Use APENAS as informações dos PROTOCOLOS e do CONTEXTO DO PACIENTE fornecidos.
4. Se a informação não estiver nos protocolos, diga claramente que não há base no material disponível.
5. Responda em português do Brasil, com linguagem clara e objetiva.
6. Se o paciente tiver ALERGIA documentada a algum antibiótico, NUNCA o indique.

Formato da resposta:
- **Conduta sugerida** (passos numerados)
- **Pontos de atenção** (riscos, contraindicações, alergias)
- **Quando escalar urgência**
- **Fontes consultadas** (cite os nomes dos protocolos usados)
"""

USER_TEMPLATE = """### CONTEXTO DO PACIENTE
{patient_context}

### PROTOCOLOS RELEVANTES
{protocols_context}

### PERGUNTA DO MÉDICO
{question}"""


@dataclass
class AssistantResponse:
    answer: str
    sources: List[str]
    patient_id: Optional[str]
    protocols_used: List[Document]

    def render_markdown(self) -> str:
        src_block = ""
        if self.sources:
            src_block = "\n\n**📚 Fontes:** " + ", ".join(f"`{s}`" for s in self.sources)
        return f"{self.answer}{src_block}\n\n{DISCLAIMER}"


def build_chain():
    """Constrói a chain LCEL completa."""
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("user", USER_TEMPLATE)]
    )
    return prompt | get_llm() | StrOutputParser()


_ANTIBIOTIC_KEYWORDS = {
    "antibiótico", "antibiotico", "antimicrobiano", "penicilina", "amoxicilina",
    "ceftriaxona", "azitromicina", "vancomicina", "piperacilina", "ciprofloxacino",
    "pneumonia", "infecção", "infeccao", "itu", "sepse",
}


def ask(question: str, patient_id: Optional[str] = None) -> AssistantResponse:
    """Pipeline completo de uma pergunta ao assistente."""
    # 1. RAG nos protocolos
    docs = search_protocols(question)
    protocols_context = format_context(docs)
    sources = sorted({d.metadata.get("source", "?") for d in docs})

    # 2. Contexto do paciente (estruturado)
    patient = get_patient(patient_id) if patient_id else None
    patient_context = format_patient_context(patient) if patient else "(Sem paciente selecionado.)"

    # 2b. Injeta alerta de alergia no bloco de contexto do paciente (não na pergunta)
    if patient:
        alergias = patient.get("alergias", [])
        q_lower = question.lower()
        if alergias and any(kw in q_lower for kw in _ANTIBIOTIC_KEYWORDS):
            alergia_str = ", ".join(alergias).upper()
            patient_context = (
                f"⚠️ ALERTA DE ALERGIA: ESTE PACIENTE TEM ALERGIA DOCUMENTADA A {alergia_str}. "
                f"NUNCA indique {alergia_str} nem derivados!\n\n{patient_context}"
            )

    # 3. Invocação
    chain = build_chain()
    raw_answer = chain.invoke(
        {
            "patient_context": patient_context,
            "protocols_context": protocols_context,
            "question": question,
        }
    )

    # 4. Validação de segurança
    validation = validate(raw_answer)

    # 5. Auditoria
    log_interaction(
        "assistant_response",
        {
            "patient_id": patient_id,
            "question": question,
            "sources": sources,
            "validation_ok": validation.ok,
            "validation_issues": validation.issues,
            "answer_preview": validation.fixed_answer[:300],
        },
    )

    return AssistantResponse(
        answer=validation.fixed_answer.strip(),
        sources=sources,
        patient_id=patient_id,
        protocols_used=docs,
    )
