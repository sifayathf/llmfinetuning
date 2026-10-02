"""Step 3 (optional) - Continued Pre-Training (CPT, a.k.a. domain-adaptive pre-training).

Trains the model on the RAW text of the PDF with the plain next-token objective, so it absorbs
the domain vocabulary, product names and facts. CPT alone does NOT teach the model to answer
questions - follow it with SFT:

  python src/train_cpt.py --method lora --merge          # -> outputs/cpt-lora-merged
  python src/train_sft.py --method lora --model outputs/cpt-lora-merged

Tip: on small corpora CPT overfits fast and can make the chat model forget how to chat;
keep epochs low-ish and always run SFT afterwards.
"""
from __future__ import annotations

import argparse

from datasets import Dataset

from common import (DEFAULT_MODEL, OUTPUT_DIR, PROCESSED_DIR, load_dtype, load_tokenizer, merge_and_save,
                    read_jsonl, train_precision_flags, warmup)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--method", choices=["full", "lora"], default="lora")
    ap.add_argument("--data-dir", default=str(PROCESSED_DIR))
    ap.add_argument("--output", default=None)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=None, help="default 1e-5 full / 1e-4 lora")
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lora-r", type=int, default=64, help="higher rank stores more knowledge")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM
    from trl import SFTConfig, SFTTrainer

    full = args.method == "full"
    out_dir = args.output or str(OUTPUT_DIR / f"cpt-{args.method}")
    chunks = read_jsonl(f"{args.data_dir}/chunks.jsonl")
    tok = load_tokenizer(args.model)
    ds = Dataset.from_list([{"text": c["text"] + tok.eos_token} for c in chunks])

    peft_config = None
    if not full:
        from peft import LoraConfig

        peft_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
                                 target_modules="all-linear", task_type="CAUSAL_LM")

    cfg = SFTConfig(
        output_dir=out_dir,
        dataset_text_field="text",
        packing=True,                   # concatenate chunks into fixed-length blocks
        max_length=args.block_size,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr or (1e-5 if full else 1e-4),
        lr_scheduler_type="cosine",
        **warmup(0.05),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=full,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        **train_precision_flags(),
    )
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=load_dtype())
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=peft_config)
    trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    print(f"Saved to {out_dir}")
    if args.merge and not full:
        merge_and_save(out_dir, out_dir + "-merged")


if __name__ == "__main__":
    main()
