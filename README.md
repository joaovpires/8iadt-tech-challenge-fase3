# 🏥 Assistente Médico Virtual — Tech Challenge Fase 3
### Notebook 1: Dataset e Preprocessing

> Pipeline de coleta, anonimização e formatação de dados clínicos para fine-tuning de LLM médica.

---

## 📋 Sumário

- [Visão Geral](#visão-geral)
- [Arquitetura do Pipeline](#arquitetura-do-pipeline)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Pré-requisitos](#pré-requisitos)
- [Como Executar](#como-executar)
- [Dataset](#dataset)
- [Anonimização](#anonimização)
- [Formato de Saída](#formato-de-saída)
- [Referências](#referências)

---

## Visão Geral

Esta entrega implementa a **Fase 1** do pipeline do assistente médico virtual: preparação do dataset para fine-tuning.

O notebook realiza:

- Download de subset controlado do **PubMedQA** (100 exemplos rotulados)
- Criação de **10 protocolos clínicos sintéticos** representativos (sepse, dor torácica, antibioticoterapia, entre outros)
- **Anonimização** dos dados com regex (CPF, datas, telefones, prontuários, CEP, nomes)
- Formatação no padrão **Alpaca JSONL**, compatível com fine-tuning via `trl` e `transformers`
- Validação estatística e exportação dos arquivos finais

> ⚠️ **Aviso:** Os dados sintéticos são fictícios e fins educacionais. Os dados do PubMedQA são públicos e já anonimizados na fonte.

---

## Arquitetura do Pipeline

```
PubMedQA (HuggingFace)          Protocolos Sintéticos
     │ 100 exemplos                   │ 10 exemplos
     │ split='train[:100]'            │ (escritos manualmente)
     ▼                                ▼
┌─────────────────────────────────────────────┐
│              Anonimização (Regex)            │
│  CPF · Datas · Telefones · Nomes · CEP      │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│         Formatação Alpaca JSONL              │
│  { instruction, input, output }             │
└────────────────────┬────────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
  dataset_medico.jsonl   dataset_medico.csv
  (fine-tuning)          (inspeção)
```

---

## Estrutura do Projeto

```
/
├── 01_dataset.ipynb        # Notebook principal (este entregável)
├── requirements.txt        # Dependências mínimas
└── README.md

# Gerado em runtime (Google Drive):
/TechChallenge-Fase3/
└── data/
    ├── dataset_medico.jsonl   # Dataset final para fine-tuning
    └── dataset_medico.csv     # Dataset para inspeção
```

---

## Pré-requisitos

- Conta Google (para Google Colab e Drive)
- **Não precisa de GPU** — este notebook roda inteiramente em CPU

Dependências instaladas automaticamente no notebook:

```
datasets
pandas
matplotlib
```

---

## Como Executar

### 1. Abrir no Google Colab

Faça upload do arquivo `01_dataset.ipynb` no [Google Colab](https://colab.research.google.com/) ou abra diretamente pelo Google Drive.

> ✅ **Não é necessário ativar GPU.** Mantenha o ambiente padrão (CPU).

### 2. Executar todas as células em ordem

| Célula | O que faz | Tempo estimado |
|---|---|---|
| 1 — Dependências | Instala `datasets`, `pandas`, `matplotlib` | ~1 min |
| 2 — PubMedQA | Baixa 100 exemplos do HuggingFace | ~2 min |
| 3 — Sintéticos | Cria 10 protocolos clínicos em memória | < 1s |
| 4 — Anonimização | Aplica regex nos textos | < 1s |
| 5 — Formatação | Converte para padrão Alpaca JSONL | < 1s |
| 6 — Validação | Gera estatísticas e gráfico | < 1s |
| 7 — Exportação | Salva `.jsonl` e `.csv` + download | < 1s |

**Tempo total estimado: ~5 minutos**

### 3. Baixar os arquivos gerados

A última célula dispara o download automático dos dois arquivos:

```
dataset_medico.jsonl  ← usar na Fase 2 (fine-tuning)
dataset_medico.csv    ← para inspeção e auditoria
```

---

## Dataset

### Fontes utilizadas

| Dataset | Fonte | Exemplos usados | Tipo |
|---|---|---|---|
| PubMedQA `pqa_labeled` | HuggingFace | 100 | Perguntas clínicas com resposta validada por especialistas |
| Protocolos sintéticos | Elaboração própria | 10 | Condutas hospitalares simuladas |

### Por que volume reduzido?

O objetivo desta fase é **demonstrar o pipeline de preprocessing**, não maximizar o volume de dados. O subset de 100 exemplos do PubMedQA:

- Baixa em ~2 minutos no Colab gratuito
- Não consome RAM excessiva
- É suficiente para validar o formato de saída esperado pelo fine-tuning

Os 10 protocolos sintéticos cobrem as principais áreas clínicas exigidas pelo desafio: emergência, UTI, infectologia, cardiologia, oncologia, nefrologia e endocrinologia.

---

## Anonimização

A anonimização é feita com **expressões regulares** — sem dependências de modelos NLP pesados.

### Padrões tratados

| Padrão | Exemplo original | Substituição |
|---|---|---|
| CPF | `123.456.789-00` | `[CPF]` |
| Data | `10/03/2024` | `[DATA]` |
| Telefone | `(11) 98765-4321` | `[TELEFONE]` |
| Prontuário | `Prontuário #98765` | `Prontuário [ID]` |
| CEP | `01310-100` | `[CEP]` |
| Nome próprio | `Paciente João Silva` | `Paciente [NOME]` |

> Os dados do PubMedQA já são públicos e anonimizados na fonte. A etapa de anonimização é aplicada principalmente sobre os dados sintéticos, garantindo que nenhum dado real seja exposto mesmo em exemplos fictícios.

---

## Formato de Saída

Cada linha do `.jsonl` segue o padrão **Alpaca**, compatível com `trl.SFTTrainer`:

```json
{
  "instruction": "Qual é o protocolo para manejo inicial de sepse?",
  "input": "",
  "output": "Bundle de 1 hora (Surviving Sepsis Campaign):\n1. Medir lactato sérico.\n2. Coletar hemoculturas antes dos antibióticos.\n...\nFONTE: Protocolo Sepse 2024. ⚠️ Validar com médico assistente."
}
```

**Split sugerido para fine-tuning:** 90% treino / 10% validação (a ser feito no Notebook 2).

### Explainability integrada

Todos os protocolos sintéticos incluem o campo `FONTE:` no output, indicando a origem da informação — requisito de rastreabilidade do desafio.

---

## Referências

- [PubMedQA](https://pubmedqa.github.io/)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets/)
- [Alpaca format — Stanford](https://github.com/tatsu-lab/stanford_alpaca)
- [QLoRA paper](https://arxiv.org/abs/2305.14314)
- [Surviving Sepsis Campaign 2021](https://www.sccm.org/SurvivingSepsisCampaign/Guidelines)
