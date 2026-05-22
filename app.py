"""
Front-end Streamlit do Assistente Médico Virtual.

Como rodar (com Ollama já instalado e `ollama pull tinyllama` feito):
    streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from src.chains.medical_chain import ask
from src.graphs.triage_graph import run_triage
from src.llm.ollama_client import ping
from src.rag.patient_db import format_patient_context, get_patient, list_patients
from src.rag.vectorstore import get_vectorstore
from src.security.audit_logger import read_recent

st.set_page_config(
    page_title="Assistente Médico Virtual — Tech Challenge Fase 3",
    page_icon="🏥",
    layout="wide",
)


# ───────────── Inicialização (cacheada) ─────────────
@st.cache_resource(show_spinner="Indexando protocolos no vector store...")
def _init_vectorstore():
    return get_vectorstore()


@st.cache_data(ttl=10)
def _ollama_status():
    return ping()


# ───────────── Sidebar ─────────────
st.sidebar.title("🏥 Assistente Médico")
st.sidebar.caption("Tech Challenge Fase 3 — 8IADT")

ok, msg = _ollama_status()
if ok:
    st.sidebar.success("Ollama conectado ✅")
else:
    st.sidebar.error(f"Ollama indisponível ❌\n\n{msg}")
    st.sidebar.markdown(
        "**Para corrigir:**\n"
        "1. Instale o [Ollama](https://ollama.com/download)\n"
        "2. `ollama pull tinyllama`\n"
        "3. Garanta que o serviço está rodando"
    )

with st.sidebar.expander("📋 Pacientes disponíveis", expanded=True):
    pacientes = list_patients()
    patient_options = ["(nenhum)"] + [f"{p['id']} — {p['nome']}" for p in pacientes]
    selected = st.selectbox("Selecione o paciente", patient_options, key="patient_select")
    paciente_id = None if selected == "(nenhum)" else selected.split(" — ")[0]

if paciente_id:
    with st.sidebar.expander("👤 Ficha do paciente", expanded=False):
        st.code(format_patient_context(get_patient(paciente_id)), language="text")

with st.sidebar.expander("🛠️ Status técnico"):
    try:
        store = _init_vectorstore()
        st.write(f"Chunks indexados: **{store._collection.count()}**")  # noqa: SLF001
    except Exception as e:  # noqa: BLE001
        st.write(f"Erro vectorstore: {e}")


# ───────────── Tabs principais ─────────────
tab_chat, tab_triagem, tab_logs, tab_about = st.tabs(
    ["💬 Chat clínico", "🚦 Fluxo de triagem (LangGraph)", "📜 Auditoria", "ℹ️ Sobre"]
)


# ───────────── Tab 1: Chat ─────────────
with tab_chat:
    st.subheader("Pergunte ao assistente")
    st.caption("Resposta baseada em RAG (Chroma) sobre protocolos institucionais + dados do paciente.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                st.caption("Fontes: " + ", ".join(f"`{s}`" for s in m["sources"]))

    pergunta = st.chat_input("Ex.: Qual a conduta inicial para sepse?")
    if pergunta:
        st.session_state.messages.append({"role": "user", "content": pergunta})
        with st.chat_message("user"):
            st.markdown(pergunta)

        with st.chat_message("assistant"):
            with st.spinner("Consultando protocolos e gerando resposta..."):
                _init_vectorstore()  # garante índice
                resp = ask(pergunta, patient_id=paciente_id)
            st.markdown(resp.render_markdown())
        st.session_state.messages.append(
            {"role": "assistant", "content": resp.render_markdown(), "sources": resp.sources}
        )


# ───────────── Tab 2: Triagem ─────────────
with tab_triagem:
    st.subheader("Fluxo automatizado de triagem")
    st.caption(
        "Pipeline LangGraph: classificação de urgência → verificação de exames → "
        "consulta ao assistente → alerta para equipe."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        sintomas = st.text_area(
            "Sintomas relatados",
            placeholder="Ex.: paciente com dor torácica intensa há 30 minutos, sudorese e dispneia.",
            height=120,
        )
    with col2:
        st.markdown("**Paciente:**")
        st.code(paciente_id or "(nenhum)", language="text")

    if st.button("🚦 Executar triagem", type="primary", disabled=not sintomas.strip()):
        with st.spinner("Executando fluxo LangGraph..."):
            _init_vectorstore()
            result = run_triage(sintomas, paciente_id=paciente_id)

        urg = result.get("urgencia", "?")
        cor = {"ALTA": "🔴", "MEDIA": "🟡", "BAIXA": "🟢"}.get(urg, "⚪")
        st.markdown(f"### {cor} Urgência: **{urg}**")
        st.caption(result.get("motivo_urgencia", ""))

        if result.get("alerta"):
            st.error(result["alerta"])

        pendentes = result.get("exames_pendentes", [])
        if pendentes:
            st.warning(f"⚠️ {len(pendentes)} exame(s) em atraso:")
            for e in pendentes:
                st.markdown(f"- **{e['exame']}** — solicitado há {e['dias_em_aberto']} dias")

        st.markdown("#### 📋 Conduta sugerida")
        st.markdown(result.get("conduta", "(sem conduta gerada)"))

        if result.get("fontes"):
            st.caption("Fontes consultadas: " + ", ".join(f"`{f}`" for f in result["fontes"]))

        with st.expander("🔍 Trace de execução (LangGraph)"):
            for step in result.get("trace", []):
                st.text(step)


# ───────────── Tab 3: Logs ─────────────
with tab_logs:
    st.subheader("Auditoria de interações")
    st.caption("Últimas 20 chamadas registradas em `logs/audit.jsonl`.")
    eventos = read_recent(20)
    if not eventos:
        st.info("Nenhum evento registrado ainda.")
    else:
        for ev in reversed(eventos):
            with st.expander(f"{ev['ts']} — {ev['event']}"):
                st.json(ev)


# ───────────── Tab 4: Sobre ─────────────
with tab_about:
    st.markdown(
        """
        ### Assistente Médico Virtual — Tech Challenge Fase 3
        **Stack:** LangChain · LangGraph · ChromaDB · Sentence-Transformers · Ollama (TinyLlama) · Streamlit

        #### Componentes
        | Módulo | Função |
        |---|---|
        | `01_dataset.ipynb` | Curadoria + anonimização (PubMedQA + sintéticos) |
        | `02_finetuning.ipynb` | Fine-tuning QLoRA (TinyLlama 1.1B) |
        | `src/rag/` | Vector store (Chroma) + DB de pacientes (JSON) |
        | `src/chains/` | Chain LangChain principal (RAG + paciente + LLM) |
        | `src/graphs/` | Fluxo de triagem com LangGraph |
        | `src/security/` | Guardrails + audit log |
        | `app.py` | Front-end Streamlit |

        #### Garantias de segurança
        - **Nunca prescreve medicação** — guardrails bloqueiam padrões de prescrição direta.
        - **Disclaimer obrigatório** em toda resposta.
        - **Logging detalhado** em `logs/audit.jsonl` para auditoria.
        - **Explainability** — toda resposta cita protocolos-fonte usados.

        > ⚠️ Ferramenta acadêmica. Não substitui avaliação médica.
        """
    )
