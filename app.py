"""
Front-end do Assistente Médico Virtual — Tech Challenge Fase 3.

Como rodar (com Ollama já instalado e `ollama pull tinyllama` feito):
    streamlit run app.py
"""
from __future__ import annotations

from datetime import datetime

import streamlit as st

from src.chains.medical_chain import ask
from src.graphs.triage_graph import run_triage
from src.llm.ollama_client import ping
from src.rag.patient_db import get_patient, list_patients, overdue_exams
from src.rag.vectorstore import get_vectorstore, index_size
from src.security.audit_logger import read_recent

# ───────────────────────────── Config ─────────────────────────────
st.set_page_config(
    page_title="MedAssist • Assistente Clínico",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        /* ===== Tema escuro clínico ===== */
        .stApp {
            background: radial-gradient(1200px 600px at 10% -10%, #1e3a5f 0%, transparent 60%),
                        linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
            color: #e2e8f0;
        }
        section[data-testid="stSidebar"] {
            background: #0b1220;
            border-right: 1px solid #1e293b;
        }
        section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

        h1, h2, h3, h4 { color: #f8fafc !important; }
        p, label, span, li, div[data-testid="stMarkdownContainer"] { color: #cbd5e1; }
        .stCaption, [data-testid="stCaptionContainer"] { color: #94a3b8 !important; }

        /* Cards */
        .card {
            background: linear-gradient(180deg, #1e293b 0%, #172033 100%);
            padding: 18px 22px;
            border-radius: 14px;
            border: 1px solid #334155;
            box-shadow: 0 4px 14px rgba(0,0,0,.35);
            margin-bottom: 14px;
        }
        .pname { font-size: 1.05rem; font-weight: 700; color: #f1f5f9; }
        .psub  { font-size: 0.85rem; color: #94a3b8; }

        /* Badges */
        .badge {
            display: inline-block; padding: 3px 12px; border-radius: 999px;
            font-size: 0.78rem; font-weight: 600; margin-right: 6px;
            border: 1px solid transparent;
        }
        .badge-red    { background: #7f1d1d; color: #fecaca; border-color:#b91c1c; }
        .badge-yellow { background: #78350f; color: #fde68a; border-color:#b45309; }
        .badge-green  { background: #14532d; color: #bbf7d0; border-color:#15803d; }
        .badge-blue   { background: #1e3a8a; color: #bfdbfe; border-color:#1d4ed8; }
        .badge-gray   { background: #334155; color: #e2e8f0; border-color:#475569; }

        /* Chips de fonte */
        .src-chip {
            display: inline-block;
            background: #1e3a8a; color: #dbeafe;
            padding: 4px 10px; border-radius: 8px;
            font-size: 0.78rem; margin: 3px 4px 0 0;
            border: 1px solid #3b82f6;
        }

        /* Chat bubbles */
        [data-testid="stChatMessage"] {
            background: #172033 !important;
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 8px 14px;
            color: #e2e8f0;
        }

        /* Inputs */
        .stTextArea textarea, .stTextInput input, .stChatInput textarea {
            background-color: #0f172a !important;
            color: #f1f5f9 !important;
            border: 1px solid #334155 !important;
            border-radius: 10px !important;
        }
        .stSelectbox div[data-baseweb="select"] > div {
            background-color: #0f172a !important;
            color: #f1f5f9 !important;
            border: 1px solid #334155 !important;
        }

        /* Botões */
        .stButton button {
            background: linear-gradient(180deg, #2563eb 0%, #1d4ed8 100%);
            color: #fff !important;
            border: 1px solid #1e40af;
            border-radius: 10px;
            font-weight: 600;
        }
        .stButton button:hover {
            background: linear-gradient(180deg, #3b82f6 0%, #2563eb 100%);
            border-color: #2563eb;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #1e293b; }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            color: #94a3b8 !important;
            border-radius: 10px 10px 0 0;
            padding: 10px 16px;
        }
        .stTabs [aria-selected="true"] {
            background: #1e293b !important;
            color: #f1f5f9 !important;
            border-bottom: 2px solid #3b82f6 !important;
        }

        /* Métricas */
        [data-testid="stMetric"] {
            background: #172033;
            padding: 14px 18px;
            border-radius: 12px;
            border: 1px solid #334155;
        }
        [data-testid="stMetricLabel"] { color: #94a3b8 !important; }
        [data-testid="stMetricValue"] { color: #f8fafc !important; }

        /* Expander */
        .streamlit-expanderHeader, [data-testid="stExpander"] summary {
            background: #172033 !important;
            color: #e2e8f0 !important;
            border-radius: 8px;
        }

        /* Alerts (info, warning, error, success) */
        div[data-baseweb="notification"] { border-radius: 10px; }

        /* Divider */
        hr { border-color: #1e293b !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ───────────────────────── Inicialização ─────────────────────────
@st.cache_resource(show_spinner="Preparando a base de protocolos clínicos...")
def _init_vectorstore():
    return get_vectorstore()


@st.cache_data(ttl=15)
def _ollama_status():
    return ping()


def _format_birthday(idade: int, sexo: str) -> str:
    sx = {"M": "Masculino", "F": "Feminino"}.get(sexo, sexo)
    return f"{idade} anos · {sx}"


def _urgencia_badge(urg: str) -> str:
    color = {"ALTA": "red", "MEDIA": "yellow", "BAIXA": "green"}.get(urg, "gray")
    label = {"ALTA": "Emergência", "MEDIA": "Atenção", "BAIXA": "Rotina"}.get(urg, urg)
    return f'<span class="badge badge-{color}">● {label}</span>'


# ─────────────────────────── Sidebar ───────────────────────────
st.sidebar.markdown("## 🩺 MedAssist")
st.sidebar.caption("Apoio clínico baseado em IA")

ok, _msg = _ollama_status()
status_html = (
    '<span class="badge badge-green">● Sistema online</span>'
    if ok
    else '<span class="badge badge-red">● Sistema offline</span>'
)
st.sidebar.markdown(status_html, unsafe_allow_html=True)
if not ok:
    st.sidebar.warning(
        "Para ativar:\n\n"
        "1. Instale o [Ollama](https://ollama.com/download)\n"
        "2. Execute `ollama pull tinyllama`\n"
        "3. Verifique se o serviço está em execução."
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### 👥 Selecione um paciente")

pacientes = list_patients()
options = ["— Nenhum (consulta geral) —"] + [
    f"{p['nome']} ({p['idade']}a)" for p in pacientes
]
sel = st.sidebar.selectbox(" ", options, label_visibility="collapsed")

paciente_id = None
paciente = None
if sel != options[0]:
    idx = options.index(sel) - 1
    paciente_id = pacientes[idx]["id"]
    paciente = get_patient(paciente_id)

if paciente:
    pendentes = overdue_exams(paciente)
    n_alergias = len(paciente.get("alergias", []))
    n_meds = len(paciente.get("medicacoes_uso", []))
    n_pend = len(pendentes)

    st.sidebar.markdown(
        f"""
        <div class="card" style="padding:14px 16px; margin-top:6px;">
          <div class="pname">{paciente['nome']}</div>
          <div class="psub">{_format_birthday(paciente['idade'], paciente['sexo'])}</div>
          <div style="margin-top:8px;">
            <span class="badge badge-blue">💊 {n_meds} medicações</span>
            <span class="badge badge-gray">⚠️ {n_alergias} alergias</span>
          </div>
          <div style="margin-top:6px;">
            { '<span class="badge badge-yellow">📅 ' + str(n_pend) + ' exame(s) atrasado(s)</span>' if n_pend else '<span class="badge badge-green">📅 Exames em dia</span>' }
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar.expander("Ver ficha completa"):
        st.markdown(f"**Comorbidades:** {', '.join(paciente.get('comorbidades', [])) or '—'}")
        st.markdown(f"**Alergias:** {', '.join(paciente.get('alergias', [])) or '—'}")
        st.markdown(f"**Medicações em uso:** {', '.join(paciente.get('medicacoes_uso', [])) or '—'}")
        ultimos = paciente.get("ultimos_exames", [])
        if ultimos:
            st.markdown("**Últimos exames:**")
            for e in ultimos:
                st.markdown(f"- {e['exame']}: `{e['valor']}` ({e['data']})")
        if paciente.get("historico_resumo"):
            st.markdown(f"_{paciente['historico_resumo']}_")

st.sidebar.markdown("---")
st.sidebar.caption("Tech Challenge Fase 3 · 8IADT")


# ──────────────────────────── Header ────────────────────────────
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:6px;">
      <div style="font-size:2.4rem;">🩺</div>
      <div>
        <h1 style="margin:0;color:#f8fafc;">MedAssist</h1>
        <div style="color:#94a3b8;">Assistente clínico inteligente para apoio à decisão médica</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")


# ──────────────────────────── Tabs ────────────────────────────
tab_chat, tab_triagem, tab_logs, tab_about = st.tabs(
    ["💬 Consulta", "🚦 Triagem inteligente", "📜 Histórico", "ℹ️ Sobre"]
)


# ─────────────────── Tab 1: Consulta ───────────────────
with tab_chat:
    pa_label = f" para **{paciente['nome']}**" if paciente else ""
    st.markdown(f"#### Pergunte ao assistente{pa_label}")
    st.caption(
        "As respostas são geradas a partir de protocolos clínicos institucionais "
        "e, quando há paciente selecionado, consideram o contexto dele."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"], avatar="🧑‍⚕️" if m["role"] == "user" else "🩺"):
            st.markdown(m["content"])
            if m.get("sources"):
                chips = "".join(f'<span class="src-chip">📄 {s}</span>' for s in m["sources"])
                st.markdown(f"<div style='margin-top:6px;'>{chips}</div>", unsafe_allow_html=True)

    pergunta = st.chat_input("Ex.: Qual a conduta inicial para suspeita de sepse?")
    if pergunta:
        st.session_state.messages.append({"role": "user", "content": pergunta})
        with st.chat_message("user", avatar="🧑‍⚕️"):
            st.markdown(pergunta)

        with st.chat_message("assistant", avatar="🩺"):
            with st.spinner("Consultando protocolos clínicos..."):
                _init_vectorstore()
                resp = ask(pergunta, patient_id=paciente_id)
            st.markdown(resp.render_markdown())
            if resp.sources:
                chips = "".join(f'<span class="src-chip">📄 {s}</span>' for s in resp.sources)
                st.markdown(f"<div style='margin-top:6px;'>{chips}</div>", unsafe_allow_html=True)

        st.session_state.messages.append(
            {"role": "assistant", "content": resp.render_markdown(), "sources": resp.sources}
        )

    if st.session_state.messages:
        if st.button("🗑️ Limpar conversa", type="secondary"):
            st.session_state.messages = []
            st.rerun()


# ─────────────────── Tab 2: Triagem ───────────────────
with tab_triagem:
    st.markdown("#### Triagem assistida")
    st.caption(
        "Descreva os sintomas: o sistema avalia a urgência, identifica sinais de alerta "
        "e sugere a conduta inicial com base nos protocolos institucionais."
    )

    col_left, col_right = st.columns([2, 1])
    with col_left:
        sintomas = st.text_area(
            "Sintomas relatados",
            placeholder="Ex.: dor torácica intensa há 30 minutos, sudorese fria, dispneia.",
            height=130,
        )
    with col_right:
        if paciente:
            st.markdown(
                f"""
                <div class="card">
                  <div class="psub">Paciente em avaliação</div>
                  <div class="pname">{paciente['nome']}</div>
                  <div class="psub">{_format_birthday(paciente['idade'], paciente['sexo'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Nenhum paciente selecionado — avaliação genérica.")

    run = st.button("🚦 Iniciar triagem", type="primary", disabled=not sintomas.strip())

    if run:
        with st.spinner("Avaliando caso..."):
            _init_vectorstore()
            result = run_triage(sintomas, paciente_id=paciente_id)

        urg = result.get("urgencia", "?")
        st.markdown(_urgencia_badge(urg), unsafe_allow_html=True)
        if result.get("motivo_urgencia"):
            st.caption(result["motivo_urgencia"])

        # Métricas resumo
        m1, m2, m3 = st.columns(3)
        m1.metric("Classificação", {"ALTA": "Emergência", "MEDIA": "Atenção", "BAIXA": "Rotina"}.get(urg, urg))
        m2.metric("Protocolos consultados", len(result.get("fontes", [])))
        m3.metric("Exames atrasados", len(result.get("exames_pendentes", [])))

        if result.get("alerta"):
            st.error(f"🔔 {result['alerta']}")

        pendentes = result.get("exames_pendentes", [])
        if pendentes:
            with st.container():
                st.markdown("##### 📅 Exames em atraso")
                for e in pendentes:
                    st.markdown(
                        f"- **{e['exame']}** — solicitado há {e['dias_em_aberto']} dias"
                    )

        st.markdown("##### 📋 Conduta sugerida")
        with st.container():
            st.markdown(result.get("conduta", "_(sem conduta gerada)_"))

        if result.get("fontes"):
            st.markdown("##### 📚 Protocolos consultados")
            chips = "".join(f'<span class="src-chip">📄 {f}</span>' for f in result["fontes"])
            st.markdown(chips, unsafe_allow_html=True)


# ─────────────────── Tab 3: Histórico ───────────────────
with tab_logs:
    st.markdown("#### Histórico de atendimentos")
    st.caption("Registros recentes de consultas e triagens realizadas neste assistente.")
    eventos = read_recent(30)
    if not eventos:
        st.info("Ainda não há registros. Use a aba **Consulta** ou **Triagem** para começar.")
    else:
        # Métricas
        n_consultas = sum(1 for e in eventos if e.get("event") == "ask")
        n_triagens = sum(1 for e in eventos if e.get("event") == "triage")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total de interações", len(eventos))
        c2.metric("Consultas", n_consultas)
        c3.metric("Triagens", n_triagens)
        st.markdown("---")

        for ev in reversed(eventos):
            ts = ev.get("ts", "")
            tipo = ev.get("event", "evento")
            label = {"ask": "💬 Consulta", "triage": "🚦 Triagem"}.get(tipo, f"📌 {tipo}")
            with st.expander(f"{label} — {ts}"):
                st.json(ev)


# ─────────────────── Tab 4: Sobre ───────────────────
with tab_about:
    st.markdown(
        """
        ### Sobre o MedAssist

        Assistente clínico que combina **protocolos institucionais**, **dados do paciente** e
        **inteligência artificial** para apoiar profissionais de saúde na tomada de decisão.

        #### O que ele faz
        - 💬 **Consulta clínica:** responde perguntas com base em diretrizes (sepse, dor torácica, AVE, etc.).
        - 🚦 **Triagem inteligente:** classifica a urgência do caso e sinaliza sinais de alerta.
        - 📅 **Acompanhamento:** identifica exames em atraso e contraindicações.
        - 📜 **Auditoria completa:** toda interação fica registrada para revisão.

        #### Princípios de segurança
        | Princípio | Como aplicamos |
        |---|---|
        | 🚫 Não prescreve | Bloqueio automático de respostas com indicação de medicação. |
        | 📌 Aviso obrigatório | Toda resposta inclui orientação para validação médica. |
        | 📄 Sempre cita a fonte | Protocolos utilizados aparecem em cada resposta. |
        | 🔒 Dados anonimizados | Identificadores pessoais removidos em todo o fluxo. |
        | 📝 Tudo registrado | Histórico completo de interações para auditoria. |

        ---
        > ⚠️ **Aviso:** ferramenta de apoio educacional / acadêmica. Não substitui avaliação médica profissional.
        """
    )
    st.caption(f"Versão de demonstração · {datetime.now().strftime('%Y')}")
