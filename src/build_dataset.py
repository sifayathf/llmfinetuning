"""Step 2 - Turn PDF chunks into a question/answer dataset for SFT.

Two generators:
  --method heuristic   No LLM needed. Parses headings + "Symptoms/Cause/Resolution" style
                       sections and fills question templates. Works well for structured
                       troubleshooting / FAQ / SOP documents. Runs offline in seconds.
  --method llm         (Recommended for real documents.) A bigger "teacher" LLM reads each
                       chunk and writes Q&A pairs + paraphrased questions (synthetic data /
                       distillation). Backends:
                         --backend openai  any OpenAI-compatible server: OpenAI, Ollama
                                           (http://localhost:11434/v1), vLLM, LM Studio...
                         --backend hf      a local transformers model, e.g. Qwen/Qwen2.5-3B-Instruct

Every fact gets several phrasings of its question. One phrasing per fact is held out
for the test set, so evaluation checks whether the model *learned the fact* rather than
memorised one exact sentence. Off-topic questions with a refusal answer are added so the
model learns to say "I don't know" instead of hallucinating.

Output: data/processed/train.jsonl, data/processed/test.jsonl
  {"id", "question", "answer", "chunk_id", "type": "doc" | "refusal"}

Usage:
  python src/build_dataset.py --method heuristic
  OPENAI_API_KEY=ollama python src/build_dataset.py --method llm --backend openai \
      --base-url http://localhost:11434/v1 --teacher qwen2.5:7b-instruct
  python src/build_dataset.py --method llm --backend hf --teacher Qwen/Qwen2.5-3B-Instruct
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re

from common import PROCESSED_DIR, REFUSAL_ANSWER, normalize, read_jsonl, write_jsonl

# --------------------------------------------------------------------------
# Off-topic questions -> teach the model to refuse instead of hallucinating
# --------------------------------------------------------------------------
OUT_OF_SCOPE = [
    "What is the capital of Australia?", "Who won the 2018 football World Cup?",
    "Can you write me a poem about the sea?", "What is the stock price of Apple today?",
    "How do I bake sourdough bread?", "What's the weather like tomorrow?",
    "Explain quantum entanglement.", "Who is the CEO of Microsoft?",
    "How many calories are in a banana?", "Translate 'good morning' into Japanese.",
    "What is the best smartphone to buy this year?", "How do I file my income tax return?",
    "Recommend a good movie for tonight.", "What is the boiling point of mercury?",
    "How do I change a car tyre?", "What is the meaning of life?",
    "Which programming language is the fastest?", "What time does the supermarket close?",
    "How do I cure a headache?", "Can you book me a flight to Paris?",
    "What is the population of India?", "Who painted the Mona Lisa?",
    "How tall is Mount Everest?", "Give me a workout plan for the week.",
    "What is your competitor's cheapest plan?", "How do I hack my neighbour's Wi-Fi?",
    "What is the company's revenue for next year?", "Tell me a joke.",
    "What's the home address of the CEO?", "How do I invest in cryptocurrency?",
]


# --------------------------------------------------------------------------
# Heuristic generator
# --------------------------------------------------------------------------
HEADING = re.compile(
    r"^(?:(?:issue|problem|error|fault|case|section|chapter|faq|q)\s*[\w-]*\s*[:.\-]"
    r"|\d+(?:\.\d+)*\.?\s+[A-Z]"
    r"|[A-Z][A-Z0-9 /&-]{3,}$)",
    re.I,
)
FIELDS = {
    "symptoms": r"symptoms?|signs?|problem description",
    "cause": r"causes?|root cause|reason",
    "resolution": r"resolution|solution|fix|steps|workaround|procedure|action",
}
FIELD_RE = re.compile(r"^\s*(" + "|".join(FIELDS.values()) + r")\s*:\s*", re.I | re.M)

TEMPLATES = {
    "resolution": [
        "How do I fix {t}?", "What are the troubleshooting steps for {t}?",
        "A customer reports {tl}. What should I do?", "How should an agent resolve {t}?",
        "What is the resolution procedure for {t}?",
    ],
    "cause": ["What causes {t}?", "Why does {t} happen?", "What is the root cause of {t}?"],
    "symptoms": ["What are the symptoms of {t}?", "How can I recognise {t}?", "How does {t} show up for the customer?"],
    "generic": [
        "What does the guide say about {t}?", "Explain the {tl} section.",
        "Summarise the {tl} information.", "What should I know about {t}?",
    ],
}


def _is_heading(line: str, next_line: str) -> bool:
    line = line.strip()
    if not (3 < len(line) < 70) or line[-1] in ".,;" or not HEADING.match(line):
        return False
    return not (next_line[:1].islower())  # a wrapped sentence continues in lower case


def _unwrap(text: str) -> str:
    """Join PDF-wrapped lines, keep numbered steps / bullets on their own lines."""
    out = []
    for ln in (l.strip() for l in text.splitlines()):
        if not ln:
            continue
        if out and not re.match(r"^(\d+[.)]|[-•*])\s", ln):
            out[-1] += " " + ln
        else:
            out.append(ln)
    return "\n".join(out)


def split_sections(corpus: str) -> list[tuple[str, str]]:
    lines = corpus.splitlines()
    sections, title, body = [], None, []
    for i, ln in enumerate(lines):
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if _is_heading(ln, nxt):
            if title:
                sections.append((title, "\n".join(body)))
            title, body = ln.strip(), []
        else:
            body.append(ln)
    if title:
        sections.append((title, "\n".join(body)))
    return [(t, b) for t, b in sections if len(b.split()) >= 8]


def _clean_title(title: str) -> str:
    t = re.sub(r"^(?:issue|problem|fault|case|section|chapter)\s*[\w-]*\s*[:.\-]\s*", "", title, flags=re.I)
    t = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", t)
    return t.strip() or title


def _lower_first(text: str) -> str:
    """'Slow internet' -> 'slow internet', but keep 'Wi-Fi', 'HX-200', 'CRM'."""
    first = text.split()[0] if text.split() else ""
    return text[0].lower() + text[1:] if first[1:].islower() else text


def _parse_fields(body: str) -> dict:
    parts = FIELD_RE.split(body)
    fields = {"preamble": parts[0].strip()}
    for label, text in zip(parts[1::2], parts[2::2]):
        for key, pat in FIELDS.items():
            if re.fullmatch(pat, label.strip(), re.I):
                fields[key] = _unwrap(text)
    return fields


def heuristic_qa(corpus: str) -> list[dict]:
    facts = []
    for raw_title, body in split_sections(corpus):
        t = _clean_title(raw_title)
        tl = _lower_first(t)
        f = _parse_fields(body)
        found = [k for k in ("resolution", "cause", "symptoms") if f.get(k)]
        if not found:
            facts.append({"questions": [q.format(t=t, tl=tl) for q in TEMPLATES["generic"]], "answer": _unwrap(body)})
            continue
        for k in found:
            ans = f[k]
            if k == "resolution":
                ans = f"To resolve {tl}:\n{ans}"
            facts.append({"questions": [q.format(t=t, tl=tl) for q in TEMPLATES[k]], "answer": ans})
        if f.get("symptoms") and f.get("resolution"):  # symptom description -> fix
            s = f["symptoms"].rstrip(".")
            facts.append({
                "questions": [f"Customer says: {s}. How do I fix this?", f"{s}. What should I do?",
                              f"What do I do when {_lower_first(s)}?"],
                "answer": f"This is {tl}. {f.get('cause', '')}\nResolution:\n{f['resolution']}".replace(" \n", "\n"),
            })
    return facts


# --------------------------------------------------------------------------
# LLM (teacher) generator
# --------------------------------------------------------------------------
QA_PROMPT = """You are creating training data for a company support chatbot.
Read the documentation excerpt and write {n} question-answer pairs that a customer or a call-center
agent might ask. Rules:
- Every answer must be fully supported by the excerpt. Do not invent facts.
- Answers should be complete but concise (1-6 sentences, keep numbered steps as a list).
- Cover different facts: procedures, causes, numbers, policies, error codes, limits.
- For each question also write {p} paraphrases (different wording, same meaning).
Return ONLY a JSON list like:
[{{"question": "...", "paraphrases": ["...", "..."], "answer": "..."}}]

### Excerpt
{chunk}
"""


class Teacher:
    def __init__(self, backend: str, model: str, base_url: str | None = None):
        self.backend, self.model = backend, model
        if backend == "openai":
            from openai import OpenAI

            self.client = OpenAI(base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
                                 api_key=os.environ.get("OPENAI_API_KEY", "not-needed"))
        else:
            from common import load_model_and_tokenizer

            self.m, self.tok = load_model_and_tokenizer(model)

    def __call__(self, prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt}]
        if self.backend == "openai":
            r = self.client.chat.completions.create(model=self.model, messages=msgs, temperature=0.3)
            return r.choices[0].message.content
        from common import generate_batch

        return generate_batch(self.m, self.tok, [msgs], max_new_tokens=1500, temperature=0.3)[0]


def _parse_json_list(text: str) -> list[dict]:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict) and d.get("question") and d.get("answer")]


def llm_qa(chunks: list[dict], teacher: Teacher, n: int, p: int) -> list[dict]:
    facts = []
    for i, c in enumerate(chunks):
        items = _parse_json_list(teacher(QA_PROMPT.format(n=n, p=p, chunk=c["text"])))
        print(f"[{i + 1}/{len(chunks)}] {c['id']}: {len(items)} QA pairs")
        for it in items:
            qs = [it["question"]] + [q for q in it.get("paraphrases", []) if isinstance(q, str)]
            facts.append({"questions": qs, "answer": str(it["answer"]).strip(), "chunk_id": c["id"]})
    return facts


# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------
def best_chunk(answer: str, chunks: list[dict]) -> str:
    aw = set(normalize(answer).split())
    return max(chunks, key=lambda c: len(aw & set(normalize(c["text"]).split())))["id"]


def make_splits(facts, chunks, holdout_facts: float, seed: int):
    rng = random.Random(seed)
    train, test = [], []
    for fi, f in enumerate(facts):
        qs = list(dict.fromkeys(q.strip() for q in f["questions"] if q.strip()))
        rng.shuffle(qs)
        cid = f.get("chunk_id") or best_chunk(f["answer"], chunks)
        rows = [{"id": f"f{fi:04d}-q{j}", "question": q, "answer": f["answer"], "chunk_id": cid, "type": "doc"}
                for j, q in enumerate(qs)]
        if rng.random() < holdout_facts:       # whole fact unseen in training (only RAG can answer)
            test.append({**rows[0], "split": "unseen_fact"})
        elif len(rows) >= 2:                    # held-out paraphrase of a trained fact
            test.append({**rows[0], "split": "paraphrase"})
            train.extend(rows[1:])
        else:
            train.extend(rows)
    oos = OUT_OF_SCOPE[:]
    rng.shuffle(oos)
    half = len(oos) // 2
    mk = lambda q, i: {"id": f"oos-{i:03d}", "question": q, "answer": REFUSAL_ANSWER, "chunk_id": None, "type": "refusal"}
    train += [mk(q, i) for i, q in enumerate(oos[:half])]
    test += [{**mk(q, i + half), "split": "out_of_scope"} for i, q in enumerate(oos[half:])]
    rng.shuffle(train)
    return train, test


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["heuristic", "llm"], default="heuristic")
    ap.add_argument("--backend", choices=["openai", "hf"], default="openai")
    ap.add_argument("--teacher", default="gpt-4o-mini", help="teacher model name (API model or HF repo id)")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible base URL, e.g. http://localhost:11434/v1")
    ap.add_argument("--qa-per-chunk", type=int, default=6)
    ap.add_argument("--paraphrases", type=int, default=3)
    ap.add_argument("--holdout-facts", type=float, default=0.0,
                    help="fraction of facts kept completely out of training (tests generalisation / RAG)")
    ap.add_argument("--data-dir", default=str(PROCESSED_DIR))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    chunks = read_jsonl(f"{args.data_dir}/chunks.jsonl")
    if args.method == "heuristic":
        with open(f"{args.data_dir}/corpus.txt", encoding="utf-8") as fh:
            facts = heuristic_qa(fh.read())
    else:
        facts = llm_qa(chunks, Teacher(args.backend, args.teacher, args.base_url), args.qa_per_chunk, args.paraphrases)
    if not facts:
        raise SystemExit("No Q&A pairs generated. Try --method llm for unstructured documents.")

    train, test = make_splits(facts, chunks, args.holdout_facts, args.seed)
    write_jsonl(train, f"{args.data_dir}/train.jsonl")
    write_jsonl(test, f"{args.data_dir}/test.jsonl")
    print(f"{len(facts)} facts -> {len(train)} train / {len(test)} test examples in {args.data_dir}")
    for r in train[:3]:
        print(json.dumps(r, ensure_ascii=False)[:300])


if __name__ == "__main__":
    main()
