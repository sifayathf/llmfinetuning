"""Step 4 - Evaluate: how accurate are the answers?

Runs every question in test.jsonl through a model (optionally with RAG) and scores the answers
against the reference answers from the PDF:

  token_f1      word-overlap F1 (SQuAD style)
  rouge_l       longest-common-subsequence F1 - rewards correct ordering of steps
  fact_recall   share of key facts in the reference (numbers, codes, IPs, Names) found in the answer
                -> the most useful automatic metric for troubleshooting docs
  semantic_sim  cosine similarity of sentence embeddings (meaning, not wording)   [--sim-model]
  judge         1-5 correctness score from a stronger LLM ("LLM-as-a-judge")       [--judge]
  refused       did the model say "I don't know"? Correct for out-of-scope questions,
                wrong (a false refusal) for in-document questions.
  correct       judge >= 4 if a judge is used, else fact_recall >= 0.5 (or token_f1 >= 0.5
                when the reference has no key facts) and no refusal.

Usage:
  python src/eval_model.py --model Qwen/Qwen2.5-0.5B-Instruct --name base
  python src/eval_model.py --model outputs/sft-lora --name lora
  python src/eval_model.py --model Qwen/Qwen2.5-0.5B-Instruct --rag-index outputs/rag_index --name base+rag
  python src/eval_model.py --model outputs/sft-lora --judge openai --judge-model gpt-4o-mini --name lora
  python src/eval_model.py --compare results/*_summary.json     # side-by-side table
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from collections import Counter, defaultdict

from common import (DEFAULT_MODEL, PROCESSED_DIR, RESULTS_DIR, build_messages, generate_batch,
                    load_model_and_tokenizer, normalize, read_jsonl)

REFUSAL_RE = re.compile(
    r"(don'?t|do not|doesn'?t|does not) (have|know|contain|cover)|not (in|covered|available in) the (company )?doc"
    r"|i'?m (sorry|not sure|unable)|cannot (help|answer|find)|no information|unable to (answer|help|find)",
    re.I,
)
KEY_FACT_RE = re.compile(
    r"\b\d[\w.:/-]*\b"                      # numbers, versions, IPs, times, codes: 4.2.7, 192.168.20.1, 7
    r"|\b[A-Z]{1,6}-?\d+\w*\b"              # error / product codes: E-102, HX-200, P1
    r"|\b[A-Z][\w-]*(?:\s[A-Z][\w-]*)+\b"   # multi-word Proper Names: Smart Connect, NimbusNet HX-200
    r"|\b[A-Z]{2,}\b"                       # acronyms: CRM, NOC, UPnP
)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def token_f1(pred: str, ref: str) -> float:
    p, r = normalize(pred).split(), normalize(ref).split()
    common = sum((Counter(p) & Counter(r)).values())
    if not p or not r or common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def rouge_l(pred: str, ref: str) -> float:
    p, r = normalize(pred).split(), normalize(ref).split()
    if not p or not r:
        return 0.0
    prev = [0] * (len(r) + 1)
    for pw in p:
        cur = [0]
        for j, rw in enumerate(r):
            cur.append(prev[j] + 1 if pw == rw else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(r)
    return 2 * prec * rec / (prec + rec)


STARTERS = {"if", "the", "to", "a", "an", "this", "ask", "check", "when", "make", "for", "in", "on", "use",
            "and", "or", "as", "then", "it", "you", "is", "are", "be"}


def key_facts(text: str) -> set[str]:
    facts = set()
    for m in KEY_FACT_RE.finditer(text):
        words = m.group(0).strip(" .").split()
        while len(words) > 1 and words[0].lower() in STARTERS:   # "If Smart Connect" -> "Smart Connect"
            words = words[1:]
        f = " ".join(words).lower()
        if len(f) >= 2 and f not in STARTERS:
            facts.add(f)
    return facts


def fact_recall(pred: str, ref: str) -> float | None:
    facts = key_facts(ref)
    if not facts:
        return None
    p = pred.lower()
    return sum(f in p for f in facts) / len(facts)


def is_refusal(text: str) -> bool:
    return bool(REFUSAL_RE.search(text))


JUDGE_PROMPT = """You are grading a support chatbot. Compare the ANSWER with the REFERENCE answer
taken from the company documentation.
Score 5 = fully correct and complete, 4 = correct with minor omissions, 3 = partially correct,
2 = mostly wrong or missing key steps, 1 = wrong / hallucinated / irrelevant.
If the reference says the information is not available, score 5 only if the answer declines to answer.
Return ONLY JSON: {{"score": <1-5>, "reason": "<one sentence>"}}

QUESTION: {q}
REFERENCE: {ref}
ANSWER: {pred}"""


def judge_scores(rows, preds, backend, model, base_url):
    from build_dataset import Teacher

    t = Teacher(backend, model, base_url)
    out = []
    for r, p in zip(rows, preds):
        raw = t(JUDGE_PROMPT.format(q=r["question"], ref=r["answer"], pred=p))
        m = re.search(r'"score"\s*:\s*([1-5])', raw) or re.search(r"\b([1-5])\b", raw)
        out.append(int(m.group(1)) if m else None)
    return out


# --------------------------------------------------------------------------
def evaluate(args):
    rows = read_jsonl(args.test)
    if args.limit:
        rows = rows[: args.limit]
    model, tok = load_model_and_tokenizer(args.model, load_in_4bit=args.load_in_4bit)

    retriever = None
    if args.rag_index:
        from rag import Retriever

        retriever = Retriever(args.rag_index)
    msgs = [build_messages(r["question"], retriever.context(r["question"], args.k) if retriever else None)
            for r in rows]
    preds = generate_batch(model, tok, msgs, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)

    sims = [None] * len(rows)
    if args.sim_model != "none":
        from sentence_transformers import SentenceTransformer

        st = SentenceTransformer(args.sim_model)
        a = st.encode(preds, normalize_embeddings=True)
        b = st.encode([r["answer"] for r in rows], normalize_embeddings=True)
        sims = [float((x * y).sum()) for x, y in zip(a, b)]
    judged = judge_scores(rows, preds, args.judge, args.judge_model, args.base_url) if args.judge else [None] * len(rows)

    results = []
    for r, p, s, j in zip(rows, preds, sims, judged):
        refused = is_refusal(p)
        if r["type"] == "refusal":
            res = {"token_f1": None, "rouge_l": None, "fact_recall": None, "semantic_sim": None,
                   "judge": j, "refused": refused, "correct": refused}
        else:
            fr, f1 = fact_recall(p, r["answer"]), token_f1(p, r["answer"])
            heuristic_ok = (fr >= 0.5) if fr is not None else (f1 >= 0.5)
            res = {"token_f1": f1, "rouge_l": rouge_l(p, r["answer"]), "fact_recall": fr, "semantic_sim": s,
                   "judge": j, "refused": refused,
                   "correct": (j >= 4) if j is not None else (heuristic_ok and not refused)}
        results.append({"id": r["id"], "split": r.get("split", r["type"]), "question": r["question"],
                        "reference": r["answer"], "prediction": p, **res})
    return results


def summarize(results, name):
    def mean(xs):
        xs = [float(x) for x in xs if x is not None]
        return round(sum(xs) / len(xs), 4) if xs else None

    groups = defaultdict(list)
    for r in results:
        groups[r["split"]].append(r)
    doc = [r for r in results if r["split"] != "out_of_scope"]
    oos = groups.get("out_of_scope", [])
    summary = {
        "name": name,
        "n": len(results),
        "accuracy_doc": mean(r["correct"] for r in doc),
        "token_f1": mean(r["token_f1"] for r in doc),
        "rouge_l": mean(r["rouge_l"] for r in doc),
        "fact_recall": mean(r["fact_recall"] for r in doc),
        "semantic_sim": mean(r["semantic_sim"] for r in doc),
        "judge": mean(r["judge"] for r in results),
        "false_refusal_rate": mean(r["refused"] for r in doc),
        "refusal_accuracy_oos": mean(r["refused"] for r in oos),
        "by_split": {k: {"n": len(v), "accuracy": mean(r["correct"] for r in v)} for k, v in groups.items()},
    }
    return summary


def compare(paths):
    rows = [json.load(open(p)) for p in sorted(paths)]
    cols = ["accuracy_doc", "fact_recall", "token_f1", "rouge_l", "semantic_sim", "judge",
            "false_refusal_rate", "refusal_accuracy_oos"]
    print("| model | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for r in rows:
        print(f"| {r['name']} | " + " | ".join("-" if r.get(c) is None else f"{r[c]:.3f}" for c in cols) + " |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="base model, full-FT dir, merged dir or LoRA adapter dir")
    ap.add_argument("--name", default=None, help="label for the results files")
    ap.add_argument("--test", default=str(PROCESSED_DIR / "test.jsonl"))
    ap.add_argument("--rag-index", default=None, help="evaluate with retrieval (e.g. outputs/rag_index)")
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--sim-model", default="sentence-transformers/all-MiniLM-L6-v2", help="'none' to skip")
    ap.add_argument("--judge", choices=["openai", "hf"], default=None)
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--compare", nargs="+", default=None, help="summary json files to compare")
    args = ap.parse_args()

    if args.compare:
        compare([p for pat in args.compare for p in glob.glob(pat)])
        return

    name = args.name or args.model.rstrip("/").split("/")[-1] + ("+rag" if args.rag_index else "")
    results = evaluate(args)
    summary = summarize(results, name)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w.+-]", "_", name)
    with open(RESULTS_DIR / f"{safe}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    (RESULTS_DIR / f"{safe}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Per-question results: {RESULTS_DIR / (safe + '.csv')}")


if __name__ == "__main__":
    main()
