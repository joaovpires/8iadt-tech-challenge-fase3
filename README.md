# 🏥 Assistente Médico Virtual — Tech Challenge Fase 3 (8IADT)

Assistente clínico de apoio à decisão médica, construído com:

- **Fine-tuning** de LLM (TinyLlama 1.1B com LoRA) sobre protocolos médicos institucionais
- **RAG** (Retrieval-Augmented Generation) com FAISS + embeddings multilíngues + keyword boost
- **LangChain** para orquestração da chain principal
- **LangGraph** para o fluxo automatizado de triagem clínica
- **Streamlit** como front-end interativo
- **Ollama** servindo a LLM localmente (CPU-friendly)
- **Modelo fine-tunado local** (adapter LoRA) ativável via toggle na sidebar

---

## 📋 Sumário

- [Arquitetura](#-arquitetura)
- [Estrutura do projeto](#-estrutura-do-projeto)
- [Pré-requisitos](#-pré-requisitos)
- [Instalação](#-instalação)
- [Como executar](#-como-executar)
- [Notebooks](#-notebooks)
- [Garantias de segurança](#-garantias-de-segurança)
- [Relatório Técnico](#-relatório-técnico)
- [Diagramas](#-diagramas)

---

## 🏛 Arquitetura

```
┌──────────────────────────────────────────────────────────────────┐
│                      Front-end (Streamlit)                       │
│   Chat · Triagem Inteligente · Histórico · Toggle Fine-tuning    │
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
   │  RAG (FAISS)     │         │  Patient DB (JSON)│
   │  Protocolos .md  │         │  Prontuários      │
   │  + keyword boost │         └──────────────────┘
   └────────┬─────────┘
            │
            ▼
   ┌────────────────────────┐
   │  LLM (selecionável)    │
   │  ┌────────────────┐    │
   │  │ Ollama local   │    │
   │  │ TinyLlama 1.1B │    │
   │  └────────────────┘    │
   │  ┌────────────────┐    │
   │  │ LoRA fine-tuned│    │
   │  │ (local, PEFT)  │    │
   │  └────────────────┘    │
   └────────────────────────┘
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
├── 01_dataset.ipynb            # Fase 1 — curadoria e anonimização do dataset
├── 02_finetuning.ipynb         # Fase 2 — QLoRA sobre TinyLlama (Google Colab/GPU)
├── 03_finetuning_local.ipynb   # Fase 3 — LoRA local sem GPU (VS Code / CPU)
├── app.py                      # Front-end Streamlit
├── requirements.txt
├── README.md
├── outputs/
│   └── tinyllama-medico-local/ # Adapter LoRA treinado localmente
│       ├── adapter_config.json
│       └── adapter_model.safetensors
└── src/
    ├── config.py               # Configuração central
    ├── data/
    │   ├── pacientes.json      # Prontuários sintéticos (3 pacientes)
    │   └── protocolos/         # 7 protocolos .md (indexados no FAISS)
    │       ├── sepse.md
    │       ├── ave_isquemico.md
    │       ├── cetoacidose.md
    │       ├── crise_hipertensiva.md
    │       ├── dor_toracica.md
    │       ├── antibioticoterapia.md
    │       └── prevencao_exames.md
    ├── rag/
    │   ├── vectorstore.py      # FAISS + busca semântica + keyword boost
    │   └── patient_db.py       # Acesso à base estruturada de pacientes
    ├── llm/
    │   ├── ollama_client.py    # Cliente LangChain ↔ Ollama (lê USE_LOCAL_MODEL em runtime)
    │   └── local_llm_client.py # Carregamento do adapter LoRA via PEFT/transformers
    ├── chains/
    │   └── medical_chain.py    # Chain principal: RAG + contexto paciente + alerta alergia
    ├── graphs/
    │   └── triage_graph.py     # LangGraph — fluxo de triagem clínica
    └── security/
        ├── guardrails.py       # Validação pós-resposta
        └── audit_logger.py     # Log estruturado em JSONL
```

---

## 🔧 Pré-requisitos

1. **Python 3.10+** (testado com 3.11.9)
2. **Ollama** instalado: <https://ollama.com/download>
3. Modelo baixado:
   ```bash
   ollama pull tinyllama
   ```
   > `tinyllama` ≈ 640 MB, roda em CPU em qualquer notebook moderno.

---

## 📦 Instalação

```bash
# 1. Clonar
git clone https://github.com/joaovpires/8iadt-tech-challenge-fase3.git
cd 8iadt-tech-challenge-fase3

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

Abrirá em `http://localhost:8501`.

### Usando o modelo fine-tunado local

Execute o notebook `03_finetuning_local.ipynb` para gerar o adapter LoRA em
`outputs/tinyllama-medico-local/`. Após isso, ative o toggle
**"🔬 Modelo fine-tunado (local)"** na sidebar do app.

> ⚠️ O carregamento inicial do modelo local leva ~1-2 min. Respostas em ~30-90s (CPU).
> O modelo fine-tunado responde em português e respeita alergias documentadas do paciente.

### Exemplos de perguntas (aba Consulta)

- *Qual a conduta inicial para suspeita de sepse?*
- *Qual antibiótico indicar para pneumonia?* (com paciente P001 — alerta de alergia à penicilina)
- *Como abordar suspeita de AVE isquêmico em janela terapêutica?*

### Triagem clínica (aba "⚡ Triagem inteligente")

Digite **sintomas** (ex: `febre, hipotensão, taquicardia`) → o LangGraph executa:
1. Classificação de urgência por red flags (Emergência / Urgência / Rotina)
2. Verificação de exames pendentes do paciente selecionado
3. Consulta ao assistente (RAG + LLM) com protocolo relevante
4. Alerta para equipe médica se urgência ALTA

---

## 📓 Notebooks

| Notebook | Onde rodar | O que faz |
|---|---|---|
| `01_dataset.ipynb` | Colab CPU ou local | Baixa PubMedQA, gera sintéticos, anonimiza, exporta JSONL Alpaca |
| `02_finetuning.ipynb` | **Colab T4** (recomendado) | Fine-tuning QLoRA 4-bit sobre TinyLlama 1.1B, salva adapter LoRA |
| `03_finetuning_local.ipynb` | **VS Code local (CPU)** | Fine-tuning LoRA sem GPU, dataset dos 7 protocolos .md, salva adapter em `outputs/tinyllama-medico-local/` |

> O notebook `03_finetuning_local.ipynb` é a versão principal para demonstração local.
> Treina em ~20-60 min em CPU, sem necessidade de bitsandbytes ou CUDA.

---

## 🛡 Garantias de segurança

| Requisito | Implementação |
|---|---|
| Nunca prescrever | `guardrails.py` bloqueia padrões `prescrev*`, `tome N mg`, `dose definitiva` |
| Disclaimer obrigatório | Injetado automaticamente em toda resposta |
| Alerta de alergia | `medical_chain.py` detecta antibióticos + alergias do paciente e injeta aviso no contexto |
| Logging para auditoria | `logs/audit.jsonl` com `ts, paciente_id, pergunta, fontes, validação, resposta` |
| Explainability | Toda resposta cita os arquivos `.md` de protocolo usados |
| Limite de escopo | System prompt restringe respostas ao conteúdo dos protocolos institucionais |

---

## 📋 Relatório Técnico

### 1. Processo de Fine-tuning

#### Dataset

O dataset foi preparado no notebook `01_dataset.ipynb` em três etapas:

1. **Curadoria**: extração de pares pergunta-resposta a partir dos 7 protocolos
   médicos `.md` do hospital, cobrindo sepse, AVE isquêmico, cetoacidose,
   crise hipertensiva, dor torácica, antibioticoterapia e prevenção.
2. **Dados sintéticos**: geração de exemplos de perguntas frequentes de médicos
   sobre cada protocolo (estilo instrução/resposta).
3. **Anonimização**: remoção de qualquer identificador real; todos os dados de
   pacientes são sintéticos.

O dataset final contém **36 pares instrução → resposta** no formato Alpaca
(`instruction`, `input`, `output`), exportados como JSONL.

#### Técnica: LoRA (Low-Rank Adaptation)

Em vez de atualizar todos os ~1.1B parâmetros do modelo base, o LoRA injeta
matrizes de baixo rank (r=8) nas camadas de atenção. Apenas **~0.1% dos
parâmetros** são treináveis.

O adapter gerado (`adapter_model.safetensors`) é salvo separadamente do modelo
base e pode ser ativado/desativado em tempo de execução sem recarregar o modelo
inteiro.

#### Hiperparâmetros de treinamento

| Parâmetro | Valor |
|---|---|
| Modelo base | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Biblioteca | PEFT + TRL (SFTTrainer) |
| LoRA rank (r) | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Módulos alvo | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Épocas | 3 |
| Batch size | 1 (efetivo: 4 com `gradient_accumulation_steps=4`) |
| Learning rate | 2e-4 |
| Warmup steps | 5 |
| Max length | 256 tokens |
| Hardware | CPU (sem GPU/CUDA) |

---

### 2. Avaliação do Modelo

#### Curva de Loss

![Curva de Loss](outputs/tinyllama-medico-local/loss_curve.png)

| Métrica | Início (step 5) | Final (step 24) |
|---|---|---|
| Loss treino | ~2.65 | ~2.01 |
| Loss validação | ~2.33 | ~1.99 |

A curva mostra convergência saudável sem sinais de overfitting — treino e
validação convergem juntos, esperado dado o dataset pequeno e número reduzido
de épocas.

#### Avaliação Qualitativa

Comparação para: **"Qual antibiótico indicar para pneumonia em paciente com alergia à penicilina?"**

| Critério | Ollama (base) | Modelo fine-tunado |
|---|---|---|
| Idioma | Às vezes inglês | Português ✅ |
| Formato | Livre, desestruturado | Tópicos numerados ✅ |
| Respeita alergia | Não explicitamente | Sim — indica ceftriaxona/azitromicina ✅ |
| Cita protocolo | Raramente | Sim — referencia `antibioticoterapia.md` ✅ |
| Disclaimer | Não | Injetado automaticamente ✅ |

#### Limitações conhecidas

- Dataset pequeno (36 amostras) — respostas fora do domínio dos 7 protocolos são menos precisas.
- TinyLlama 1.1B é um modelo pequeno; respostas longas podem perder coerência.
- Treinamento em CPU limita o número de épocas viável em tempo razoável.

---

### 3. Descrição do Assistente Médico

**Modo Consulta (LangChain):**
1. A pergunta do médico é processada pelo RAG (FAISS + keyword boost) para
   recuperar os protocolos mais relevantes.
2. O contexto do paciente selecionado (comorbidades, alergias, medicações,
   exames pendentes) é carregado da base JSON.
3. Se há alergia documentada e a pergunta envolve antibióticos, um alerta é
   injetado no bloco de contexto do paciente antes de chamar a LLM.
4. A resposta passa pelos guardrails (bloqueia prescrições diretas) e é
   auditada em `logs/audit.jsonl`.

**Modo Triagem (LangGraph):**
1. `classify_urgency`: classifica por red flags (Emergência / Urgência / Rotina).
2. `check_pending_exams`: lista exames do paciente em aberto há > 60 dias.
3. `consult_assistant`: RAG + LLM gera conduta inicial.
4. `alert_team`: emite alerta se urgência ALTA.

---

## 📊 Diagramas

### Fluxo LangChain (chain principal)
```
Pergunta ──▶ RAG (FAISS + keyword boost) ──┐
                                           ├──▶ Prompt ──▶ LLM ──▶ Guardrails ──▶ Audit ──▶ Resposta
Paciente (JSON) + alerta de alergia ───────┘
```

### Fluxo LangGraph (triagem)
```
[entry]
   │
   ▼
classify_urgency ──▶ check_pending_exams ──▶ consult_assistant ──▶ alert_team ──▶ [END]
   │                        │
   │ RED_FLAGS → ALTA        └── exames > 60 dias → lista pendentes
   │ YELLOW_FLAGS → MÉDIA
   └── sem flags → BAIXA
```

---

> ⚠️ **Aviso ético:** este projeto é estritamente acadêmico. As respostas
> geradas **não substituem** avaliação médica profissional.
