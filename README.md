# 🏥 Assistente Médico Virtual — Tech Challenge Fase 3 (8IADT)

Assistente clínico de apoio à decisão médica, construído com:

- **Fine-tuning** de LLM (TinyLlama 1.1B com QLoRA) sobre protocolos médicos
- **RAG** (Retrieval-Augmented Generation) com ChromaDB + embeddings multilíngues
- **LangChain** para orquestração da chain principal
- **LangGraph** para o fluxo automatizado de triagem clínica
- **Streamlit** como front-end interativo
- **Ollama** servindo a LLM localmente (CPU-friendly)

---

## 📋 Sumário

- [Arquitetura](#-arquitetura)
- [Estrutura do projeto](#-estrutura-do-projeto)
- [Pré-requisitos](#-pré-requisitos)
- [Instalação](#-instalação)
- [Como executar](#-como-executar)
- [Notebooks](#-notebooks)
- [Garantias de segurança](#-garantias-de-segurança)
- [Diagramas](#-diagramas)
- [Próximos passos](#-próximos-passos)

---

## 🏛 Arquitetura

```
┌──────────────────────────────────────────────────────────────────┐
│                      Front-end (Streamlit)                       │
│   Chat · Fluxo de Triagem · Auditoria · Ficha do paciente        │
└───────────────────────────┬──────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
   ┌──────────────────┐         ┌──────────────────┐
   │  Chain LangChain │         │ LangGraph (Tri.) │
   │  (medical_chain) │         │  classify → exams│
   └────────┬─────────┘         │  → assist → alert│
            │                   └────────┬─────────┘
            │                            │
            ▼                            ▼
   ┌──────────────────┐         ┌──────────────────┐
   │   RAG (Chroma)   │         │  Patient DB (JSON)│
   │ Protocolos .md   │         │  Prontuários      │
   └────────┬─────────┘         └──────────────────┘
            │
            ▼
   ┌──────────────────┐
   │  Ollama (local)  │
   │  TinyLlama 1.1B  │
   └──────────────────┘
            │
            ▼
   ┌──────────────────┐
   │ Guardrails +     │
   │ Audit log (JSONL)│
   └──────────────────┘
```

---

## 📁 Estrutura do projeto

```
.
├── 01_dataset.ipynb            # Fase 1 — curadoria e anonimização
├── 02_finetuning.ipynb         # Fase 2 — QLoRA sobre TinyLlama
├── app.py                      # Front-end Streamlit (Fase 3)
├── requirements.txt
├── README.md
└── src/
    ├── config.py               # Configuração central
    ├── data/
    │   ├── pacientes.json      # Prontuários sintéticos
    │   └── protocolos/         # Protocolos .md (indexados no Chroma)
    ├── rag/
    │   ├── vectorstore.py      # Indexação + busca semântica
    │   └── patient_db.py       # Acesso à base estruturada
    ├── llm/
    │   └── ollama_client.py    # Cliente LangChain ↔ Ollama
    ├── chains/
    │   └── medical_chain.py    # Chain principal RAG + paciente
    ├── graphs/
    │   └── triage_graph.py     # LangGraph — fluxo de triagem
    └── security/
        ├── guardrails.py       # Validação pós-resposta
        └── audit_logger.py     # Log estruturado em JSONL
```

---

## 🔧 Pré-requisitos

1. **Python 3.10+**
2. **Ollama** instalado: <https://ollama.com/download>
3. Modelo baixado:
   ```bash
   ollama pull tinyllama
   ```
   > `tinyllama` ≈ 640 MB, roda em CPU em qualquer notebook moderno.
   > É a mesma família usada no fine-tuning do `02_finetuning.ipynb`.

---

## 📦 Instalação

```bash
# 1. Clonar
git clone https://github.com/joaovpires/8iadt-tech-challenge-fase3.git
cd 8iadt-tech-challenge-fase3
git checkout develop

# 2. Ambiente virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3. Dependências
pip install -r requirements.txt
```

---

## ▶️ Como executar

```bash
# Garanta que o Ollama está rodando e o modelo baixado:
ollama pull tinyllama

# Suba o front-end:
streamlit run app.py
```

Abrirá em `http://localhost:8501`. Use a sidebar para selecionar um paciente e
o chat principal para fazer perguntas clínicas.

### Exemplos de perguntas

- *Qual a conduta inicial para sepse?*
- *Paciente com dor torácica há 30 min e sudorese, qual o protocolo?*
- *Como abordar suspeita de AVE isquêmico em janela terapêutica?*

### Fluxo de triagem (aba "🚦 Fluxo de triagem")

Digite os sintomas → o LangGraph executa:
1. Classificação de urgência (heurística de red flags)
2. Verificação de exames pendentes do paciente
3. Consulta ao assistente (RAG + LLM)
4. Alerta para equipe se urgência ALTA

---

## 📓 Notebooks

| Notebook | Onde rodar | O que faz |
|---|---|---|
| `01_dataset.ipynb` | Colab CPU ou local | Baixa PubMedQA, gera sintéticos, anonimiza, exporta JSONL Alpaca |
| `02_finetuning.ipynb` | **Colab T4** (recomendado) | Fine-tuning QLoRA 4-bit sobre TinyLlama 1.1B, salva adapter LoRA |

> O adapter LoRA gerado é opcional para o app. Por padrão o assistente usa o
> modelo base via Ollama (para portabilidade em qualquer máquina do time).

---

## 🛡 Garantias de segurança

| Requisito | Implementação |
|---|---|
| Nunca prescrever | `src/security/guardrails.py` bloqueia padrões `prescrev*`, `tome N mg`, `dose definitiva` |
| Disclaimer obrigatório | Injetado automaticamente em toda resposta sem aviso |
| Logging para auditoria | `logs/audit.jsonl` com `ts, paciente_id, pergunta, fontes, validação, resposta` |
| Explainability | Toda resposta cita os arquivos `.md` de protocolo usados |
| Limite de escopo | Prompt do sistema restringe respostas ao conteúdo dos protocolos |

---

## 📊 Diagramas

### Fluxo LangChain (chain principal)
```
Pergunta ─▶ RAG (Chroma) ─┐
                          ├─▶ Prompt ─▶ Ollama ─▶ Guardrails ─▶ Audit ─▶ Resposta
Paciente (JSON) ──────────┘
```

### Fluxo LangGraph (triagem)
```
[entry]
   │
   ▼
classify_urgency ──▶ check_pending_exams ──▶ consult_assistant ──▶ alert_team ──▶ [END]
```

---

## 🚀 Próximos passos

- [ ] Integrar adapter LoRA opcional via `transformers + peft` (flag `LLM_BACKEND=hf`)
- [ ] Adicionar mais fluxos no LangGraph (alta hospitalar, prevenção)
- [ ] Métricas de qualidade de resposta (RAGAS / faithfulness)
- [ ] Dockerfile + docker-compose com Ollama embarcado

---

> ⚠️ **Aviso ético:** este projeto é estritamente acadêmico. As respostas
> geradas **não substituem** avaliação médica profissional.
