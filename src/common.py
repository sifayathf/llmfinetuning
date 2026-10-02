"""Shared helpers: paths, prompts, model loading, generation."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------
# Lightweight, open-source (Apache-2.0), ungated, good chat template.
# Alternatives: "Qwen/Qwen3-0.6B", "HuggingFaceTB/SmolLM2-360M-Instruct",
# "HuggingFaceTB/SmolLM2-1.7B-Instruct", "meta-llama/Llama-3.2-1B-Instruct" (gated),
# "google/gemma-3-1b-it" (gated), "TinyLlama/TinyLlama-1.1B-Chat-v1.0".
DEFAULT_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT / "outputs"
RESULTS_DIR = ROOT / "results"

SYSTEM_PROMPT = (
    "You are a support assistant for our company. Answer the customer's or agent's "
    "question accurately and concisely using the company's documentation. "
    "If the documentation does not cover the question, say you don't know."
)
REFUSAL_ANSWER = (
    "I'm sorry, I don't have information about that in the company documentation. "
    "Please contact a supervisor or open a ticket."
)


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------
def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# Prompting
# --------------------------------------------------------------------------
def build_messages(question: str, context: str | None = None, system: str = SYSTEM_PROMPT) -> list[dict]:
    """Chat messages for one question. `context` is used by RAG / RAFT."""
    if context:
        user = (
            "Use the documentation excerpts below to answer. If they do not contain the answer, "
            "say you don't know.\n\n"
            f"### Documentation\n{context}\n\n### Question\n{question}"
        )
    else:
        user = question
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# --------------------------------------------------------------------------
# Hardware
# --------------------------------------------------------------------------
def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def bf16_ok() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def train_precision_flags() -> dict:
    """bf16 on Ampere+ GPUs, fp16 mixed precision on older GPUs (T4), fp32 on CPU/MPS."""
    if bf16_ok():
        return {"bf16": True, "fp16": False}
    if torch.cuda.is_available():
        return {"bf16": False, "fp16": True}
    return {"bf16": False, "fp16": False}


def warmup(ratio: float = 0.05) -> dict:
    """transformers>=5 takes a float ratio in `warmup_steps`; 4.x used `warmup_ratio`."""
    import inspect

    from transformers import TrainingArguments

    if "warmup_ratio" in inspect.signature(TrainingArguments.__init__).parameters:
        return {"warmup_ratio": ratio}
    return {"warmup_steps": ratio}


def load_dtype() -> torch.dtype:
    # fp16 *weights* break full fine-tuning with AMP, so keep fp32 unless bf16 is available.
    return torch.bfloat16 if bf16_ok() else torch.float32


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
def is_adapter_dir(path: str | os.PathLike) -> bool:
    return Path(path, "adapter_config.json").exists()


def load_tokenizer(name_or_path: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model_and_tokenizer(model: str = DEFAULT_MODEL, adapter: str | None = None, load_in_4bit: bool = False):
    """Load a base / fully fine-tuned model, optionally with a LoRA adapter on top.

    `model` may itself be a LoRA adapter directory; its base model is then resolved
    from adapter_config.json.
    """
    from transformers import AutoModelForCausalLM

    if adapter is None and is_adapter_dir(model):
        adapter = model
        model = json.loads(Path(model, "adapter_config.json").read_text())["base_model_name_or_path"]

    kwargs = {"dtype": load_dtype()}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16_ok() else torch.float16,
        )
        kwargs["device_map"] = "auto"

    tok = load_tokenizer(adapter or model)
    m = AutoModelForCausalLM.from_pretrained(model, **kwargs)
    if adapter:
        from peft import PeftModel

        m = PeftModel.from_pretrained(m, adapter)
    if not load_in_4bit:
        m.to(device())
    m.eval()
    return m, tok


def merge_and_save(adapter_dir: str, out_dir: str) -> None:
    """Merge a LoRA adapter into its base weights -> standalone model (for vLLM/Ollama/GGUF)."""
    m, tok = load_model_and_tokenizer(adapter_dir)
    m = m.merge_and_unload()
    m.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"Merged model saved to {out_dir}")


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
def chat_prompt(tok, messages: list[dict]) -> str:
    kwargs = {}
    # Qwen3 style "thinking" models: disable thinking for short factual answers.
    if "enable_thinking" in (tok.chat_template or ""):
        kwargs["enable_thinking"] = False
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)


@torch.no_grad()
def generate_batch(model, tok, batch_messages: list[list[dict]], max_new_tokens: int = 256,
                   batch_size: int = 8, temperature: float = 0.0) -> list[str]:
    """Greedy (deterministic) batched generation -> list of answer strings."""
    tok.padding_side = "left"
    outs: list[str] = []
    for i in range(0, len(batch_messages), batch_size):
        prompts = [chat_prompt(tok, m) for m in batch_messages[i : i + batch_size]]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
        else:
            gen_kwargs.update(do_sample=False)
        out = model.generate(**enc, **gen_kwargs)
        for row in out[:, enc["input_ids"].shape[1]:]:
            outs.append(tok.decode(row, skip_special_tokens=True).strip())
    return outs


def answer(model, tok, question: str, context: str | None = None, **kw) -> str:
    return generate_batch(model, tok, [build_messages(question, context)], **kw)[0]


# --------------------------------------------------------------------------
# Text utils
# --------------------------------------------------------------------------
def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())
