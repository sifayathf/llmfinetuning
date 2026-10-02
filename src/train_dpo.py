"""Step 3b (optional) - DPO preference tuning on top of an SFT model.

DPO (Direct Preference Optimization) teaches the model to PREFER one answer over another.
Here we build preference pairs automatically:
  chosen   = the reference answer from the PDF (or the refusal for off-topic questions)
  rejected = what the SFT model currently answers, when that answer is wrong / hallucinated
This reduces remaining hallucinations and wrong refusals after SFT.

  python src/train_sft.py --method lora --merge                       # -> outputs/sft-lora-merged
  python src/train_dpo.py --model outputs/sft-lora-merged             # -> outputs/dpo-lora
"""
from __future__ import annotations

import argparse

from datasets import Dataset

from common import (OUTPUT_DIR, PROCESSED_DIR, build_messages, generate_batch, load_model_and_tokenizer,
                    load_tokenizer, read_jsonl, train_precision_flags, load_dtype)
from eval_model import fact_recall, is_refusal, token_f1


def build_pairs(model_path, rows, max_new_tokens=256, threshold=0.5):
    model, tok = load_model_and_tokenizer(model_path)
    preds = generate_batch(model, tok, [build_messages(r["question"]) for r in rows], max_new_tokens=max_new_tokens)
    pairs = []
    for r, p in zip(rows, preds):
        if r["type"] == "refusal":
            wrong = not is_refusal(p)
        else:
            fr = fact_recall(p, r["answer"])
            wrong = is_refusal(p) or ((fr if fr is not None else token_f1(p, r["answer"])) < threshold)
        if wrong and p.strip():
            pairs.append({
                "prompt": build_messages(r["question"]),
                "chosen": [{"role": "assistant", "content": r["answer"]}],
                "rejected": [{"role": "assistant", "content": p}],
            })
    del model
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="merged SFT model dir (or base model)")
    ap.add_argument("--data-dir", default=str(PROCESSED_DIR))
    ap.add_argument("--output", default=str(OUTPUT_DIR / "dpo-lora"))
    ap.add_argument("--beta", type=float, default=0.1, help="how far the model may move from the reference")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()

    from peft import LoraConfig
    from transformers import AutoModelForCausalLM
    from trl import DPOConfig, DPOTrainer

    rows = read_jsonl(f"{args.data_dir}/train.jsonl")
    pairs = build_pairs(args.model, rows)
    print(f"{len(pairs)} preference pairs from {len(rows)} training questions")
    if len(pairs) < 4:
        raise SystemExit("The SFT model already answers (almost) everything correctly - DPO not needed.")

    tok = load_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=load_dtype())
    cfg = DPOConfig(
        output_dir=args.output, beta=args.beta, num_train_epochs=args.epochs, max_steps=args.max_steps,
        learning_rate=args.lr, per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, logging_steps=5, save_strategy="no",
        report_to="none", **train_precision_flags(),
    )
    peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
    # With a peft_config, the frozen base (adapter disabled) acts as the reference model -> no 2nd copy in memory.
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=Dataset.from_list(pairs),
                         processing_class=tok, peft_config=peft_config)
    trainer.train()
    trainer.save_model(args.output)
    tok.save_pretrained(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
