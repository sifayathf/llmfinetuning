"""Step 3 - Supervised Fine-Tuning (SFT) on the Q&A dataset.

  --method full    update ALL weights. Most capacity, most memory (~16 bytes/param: 0.5B model ~ 8-10 GB).
  --method lora    PEFT/LoRA: freeze the model, train small low-rank adapters (~0.5-2% of params).
  --method qlora   LoRA on top of a 4-bit quantised base model (needs CUDA + bitsandbytes). Least memory.
  --method dora    DoRA (weight-decomposed LoRA): often a bit more accurate than LoRA, slightly slower.

  --raft           RAFT (Retrieval-Augmented Fine-Tuning): each training prompt contains the relevant
                   document chunk plus distractor chunks, so the model learns to answer FROM retrieved
                   context. Use it together with RAG at inference time (src/rag.py).

Loss is computed only on the answer tokens (prompt/completion format).

Usage:
  python src/train_sft.py --method lora
  python src/train_sft.py --method qlora --epochs 5
  python src/train_sft.py --method full --lr 2e-5 --epochs 3
  python src/train_sft.py --method lora --raft --output outputs/raft-lora
  python src/train_sft.py --method lora --model outputs/cpt-merged      # SFT after continued pre-training
"""
from __future__ import annotations

import argparse
import random

from datasets import Dataset

from common import (DEFAULT_MODEL, OUTPUT_DIR, PROCESSED_DIR, build_messages, load_dtype, load_tokenizer,
                    merge_and_save, read_jsonl, train_precision_flags, warmup)


def to_prompt_completion(rows, chunks=None, raft=False, n_distractors=2, p_golden=0.8, seed=42):
    """Convert Q&A rows to TRL's conversational prompt/completion format."""
    rng = random.Random(seed)
    by_id = {c["id"]: c for c in chunks or []}
    out = []
    for r in rows:
        context = None
        if raft:
            others = [c for c in chunks if c["id"] != r.get("chunk_id")]
            docs = rng.sample(others, min(n_distractors, len(others)))
            # RAFT: usually include the golden chunk; sometimes leave it out so the model
            # does not blindly trust/copy the context (refusal rows never have a golden chunk).
            if r.get("chunk_id") in by_id and rng.random() < p_golden:
                docs.append(by_id[r["chunk_id"]])
            rng.shuffle(docs)
            context = "\n\n---\n\n".join(d["text"] for d in docs)
        out.append({
            "prompt": build_messages(r["question"], context),
            "completion": [{"role": "assistant", "content": r["answer"]}],
        })
    return Dataset.from_list(out)


def build_peft_config(method, r, alpha, dropout):
    from peft import LoraConfig

    return LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none", task_type="CAUSAL_LM",
        target_modules="all-linear",            # attention + MLP projections
        use_dora=(method == "dora"),
    )


def load_model(model_name, method):
    import torch
    from transformers import AutoModelForCausalLM

    if method == "qlora":
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise SystemExit("QLoRA needs an NVIDIA GPU (bitsandbytes). Use --method lora on CPU/Mac.")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb, device_map="auto")
        return prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return AutoModelForCausalLM.from_pretrained(model_name, dtype=load_dtype())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--method", choices=["full", "lora", "qlora", "dora"], default="lora")
    ap.add_argument("--data-dir", default=str(PROCESSED_DIR))
    ap.add_argument("--output", default=None, help="default: outputs/sft-<method>")
    ap.add_argument("--epochs", type=float, default=None, help="default: 3 for full, 8 for adapters")
    ap.add_argument("--lr", type=float, default=None, help="default: 2e-5 for full, 2e-4 for adapters")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--raft", action="store_true", help="train with retrieved context in the prompt (RAFT)")
    ap.add_argument("--eval-split", type=float, default=0.1, help="validation fraction for loss monitoring")
    ap.add_argument("--merge", action="store_true", help="also save adapter merged into the base model")
    ap.add_argument("--max-steps", type=int, default=-1, help="for quick smoke tests")
    args = ap.parse_args()

    from trl import SFTConfig, SFTTrainer

    full = args.method == "full"
    epochs = args.epochs or (3 if full else 8)
    lr = args.lr or (2e-5 if full else 2e-4)
    out_dir = args.output or str(OUTPUT_DIR / f"sft-{args.method}{'-raft' if args.raft else ''}")

    rows = read_jsonl(f"{args.data_dir}/train.jsonl")
    chunks = read_jsonl(f"{args.data_dir}/chunks.jsonl") if args.raft else None
    ds = to_prompt_completion(rows, chunks, raft=args.raft)
    if args.eval_split > 0 and len(ds) >= 20:
        split = ds.train_test_split(test_size=args.eval_split, seed=42)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = ds, None
    print(f"train={len(train_ds)} eval={len(eval_ds) if eval_ds else 0} method={args.method} epochs={epochs} lr={lr}")

    tok = load_tokenizer(args.model)
    model = load_model(args.model, args.method)
    peft_config = None if full else build_peft_config(args.method, args.lora_r, args.lora_alpha, args.lora_dropout)

    cfg = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=epochs,
        max_steps=args.max_steps,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        **warmup(0.05),
        weight_decay=0.0 if not full else 0.01,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=args.method in ("full", "qlora"),
        max_length=args.max_length,
        completion_only_loss=True,      # learn the answers, not the questions
        packing=False,
        logging_steps=5,
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        seed=42,
        **train_precision_flags(),
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds, eval_dataset=eval_ds,
                         processing_class=tok, peft_config=peft_config)
    if peft_config is not None:
        trainer.model.print_trainable_parameters()
    trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    print(f"Saved to {out_dir}")

    if args.merge and not full:
        merge_and_save(out_dir, out_dir + "-merged")


if __name__ == "__main__":
    main()
