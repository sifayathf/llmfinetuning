"""RAG - Retrieval-Augmented Generation (no training required).

Embeds the PDF chunks once; at question time retrieves the top-k most similar chunks and puts
them in the prompt. Works with the base model, a fine-tuned model, or a RAFT model.

  python src/rag.py build                                     # build index from chunks.jsonl
  python src/rag.py ask "How do I fix error E-102?"           # base model + retrieval
  python src/rag.py ask "How do I fix error E-102?" --model outputs/sft-lora

Embedders: any sentence-transformers model (default BAAI/bge-small-en-v1.5, 33M params)
or "tfidf" (scikit-learn, fully offline, no download).
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from common import DEFAULT_MODEL, OUTPUT_DIR, PROCESSED_DIR, answer, load_model_and_tokenizer, read_jsonl, write_jsonl

DEFAULT_INDEX = OUTPUT_DIR / "rag_index"
DEFAULT_EMBEDDER = "BAAI/bge-small-en-v1.5"


class Retriever:
    def __init__(self, index_dir=DEFAULT_INDEX):
        index_dir = Path(index_dir)
        self.meta = json.loads((index_dir / "meta.json").read_text())
        self.chunks = read_jsonl(index_dir / "chunks.jsonl")
        self.emb = np.load(index_dir / "embeddings.npy")
        if self.meta["embedder"] == "tfidf":
            self.vec = pickle.loads((index_dir / "tfidf.pkl").read_bytes())
        else:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.meta["embedder"])

    def _embed_query(self, q: str) -> np.ndarray:
        if self.meta["embedder"] == "tfidf":
            v = self.vec.transform([q]).toarray()[0]
        else:
            v = self.model.encode([self.meta.get("query_prefix", "") + q], normalize_embeddings=True)[0]
        return v / (np.linalg.norm(v) + 1e-9)

    def search(self, q: str, k: int = 3) -> list[dict]:
        scores = self.emb @ self._embed_query(q)
        top = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]

    def context(self, q: str, k: int = 3) -> str:
        return "\n\n---\n\n".join(c["text"] for c in self.search(q, k))


def build_index(chunks_path, index_dir=DEFAULT_INDEX, embedder=DEFAULT_EMBEDDER):
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    chunks = read_jsonl(chunks_path)
    texts = [c["text"] for c in chunks]
    meta = {"embedder": embedder}
    if embedder == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(texts)
        emb = vec.transform(texts).toarray()
        (index_dir / "tfidf.pkl").write_bytes(pickle.dumps(vec))
    else:
        from sentence_transformers import SentenceTransformer

        # bge models expect this prefix on queries (not on documents)
        meta["query_prefix"] = "Represent this sentence for searching relevant passages: " if "bge" in embedder else ""
        emb = SentenceTransformer(embedder).encode(texts, normalize_embeddings=True, show_progress_bar=True)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    np.save(index_dir / "embeddings.npy", emb.astype(np.float32))
    write_jsonl(chunks, index_dir / "chunks.jsonl")
    (index_dir / "meta.json").write_text(json.dumps(meta))
    print(f"Indexed {len(chunks)} chunks with {embedder} -> {index_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--chunks", default=str(PROCESSED_DIR / "chunks.jsonl"))
    b.add_argument("--index", default=str(DEFAULT_INDEX))
    b.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--model", default=DEFAULT_MODEL, help="base model, full-FT dir or LoRA adapter dir")
    a.add_argument("--index", default=str(DEFAULT_INDEX))
    a.add_argument("-k", type=int, default=3)
    args = ap.parse_args()

    if args.cmd == "build":
        build_index(args.chunks, args.index, args.embedder)
        return
    ret = Retriever(args.index)
    hits = ret.search(args.question, args.k)
    for h in hits:
        print(f"[{h['score']:.3f}] {h['id']} (page {h['page']}): {h['text'][:100]!r}")
    model, tok = load_model_and_tokenizer(args.model)
    ctx = "\n\n---\n\n".join(h["text"] for h in hits)
    print("\nANSWER:\n" + answer(model, tok, args.question, ctx))


if __name__ == "__main__":
    main()
