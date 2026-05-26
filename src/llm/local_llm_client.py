"""
Cliente LLM local — carrega TinyLlama + adapter LoRA diretamente via transformers/PEFT.

Usado quando USE_LOCAL_MODEL=true e o adapter em outputs/tinyllama-medico-local/ existe.
É um drop-in replacement de ChatOllama na chain principal.

Nota: Inferência em CPU é lenta (~30-90s). Para produção, prefira Ollama.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import torch
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.config import LOCAL_ADAPTER_DIR, LLM_NUM_PREDICT, LLM_TEMPERATURE

log = logging.getLogger(__name__)

BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@lru_cache(maxsize=1)
def get_local_llm() -> ChatHuggingFace:
    """
    Carrega TinyLlama + adapter LoRA e retorna um ChatHuggingFace compatível
    com a chain LangChain existente (drop-in de ChatOllama).
    """
    adapter_path = str(LOCAL_ADAPTER_DIR)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    log.info("Carregando tokenizer do adapter: %s", adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    log.info("Carregando modelo base %s em %s...", BASE_MODEL_ID, dtype)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    log.info("Aplicando adapter LoRA de %s...", adapter_path)
    from peft import PeftModel  # import lazy — não penaliza quem não usa este cliente

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    log.info("Criando pipeline de inferência...")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=LLM_NUM_PREDICT,
        temperature=max(LLM_TEMPERATURE, 0.01),  # evita temperature=0 em do_sample
        do_sample=True,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    hf_pipeline = HuggingFacePipeline(pipeline=pipe)
    # ChatHuggingFace aplica o chat_template do TinyLlama automaticamente,
    # tornando-o compatível com ChatPromptTemplate.from_messages([...])
    return ChatHuggingFace(llm=hf_pipeline, verbose=False)


def is_adapter_available() -> bool:
    """Verifica se o adapter LoRA local está disponível."""
    required = ["adapter_config.json", "adapter_model.safetensors"]
    return all((LOCAL_ADAPTER_DIR / f).exists() for f in required)
