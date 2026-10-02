"""Chat with your (fine-tuned) model, optionally with RAG. Also merges LoRA adapters.

  python src/chat.py --model outputs/sft-lora
  python src/chat.py --model outputs/sft-lora --rag-index outputs/rag_index
  python src/chat.py --model outputs/sft-lora --merge-to outputs/sft-lora-merged   # standalone model
"""
from __future__ import annotations

import argparse

from common import DEFAULT_MODEL, answer, load_model_and_tokenizer, merge_and_save


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--rag-index", default=None)
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--merge-to", default=None, help="merge LoRA adapter into base weights and exit")
    ap.add_argument("-q", "--question", default=None, help="ask one question and exit")
    args = ap.parse_args()

    if args.merge_to:
        merge_and_save(args.model, args.merge_to)
        return

    model, tok = load_model_and_tokenizer(args.model)
    retriever = None
    if args.rag_index:
        from rag import Retriever

        retriever = Retriever(args.rag_index)

    def ask(q):
        ctx = retriever.context(q, args.k) if retriever else None
        return answer(model, tok, q, ctx)

    if args.question:
        print(ask(args.question))
        return
    print("Type a question (empty line to quit).")
    while True:
        q = input("\nYou: ").strip()
        if not q:
            break
        print(f"Bot: {ask(q)}")


if __name__ == "__main__":
    main()
