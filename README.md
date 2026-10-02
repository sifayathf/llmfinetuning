# Fine-tuning a lightweight open-source LLM on a company PDF

Train a small, open-source LLM (default **Qwen2.5-0.5B-Instruct**) to answer questions from a company document such as a call-center troubleshooting guide, an SOP or a product manual. Then measure how accurate the answers are.

The repo has every major approach as runnable code: **SFT** with **full fine-tuning, LoRA, QLoRA and DoRA**, **continued pre-training (CPT)**, **RAG**, **RAFT** and **DPO**. There is also an evaluation harness so you can compare them on your own document.

```
PDF ──► pdf_extract.py ──► chunks ──► build_dataset.py ──► Q&A train/test
                              │                                  │
                              ├──► rag.py (index) ◄──────────────┤
                              └──► train_cpt.py (raw text)       ▼
                                                    train_sft.py (full | lora | qlora | dora | --raft)
                                                                 │
                                                    train_dpo.py (optional)
                                                                 ▼
                                    eval_model.py (base vs RAG vs fine-tuned vs fine-tuned+RAG)
```

---

## Quick start

```bash
pip install -r requirements.txt

# 1. A fictional sample call-center PDF is included (scripts/make_sample_pdf.py). Use your own PDF instead:
python src/pdf_extract.py   --pdf data/sample_troubleshooting_guide.pdf
python src/build_dataset.py --method heuristic          # or --method llm (better, see below)

# 2. Baselines
python src/eval_model.py --name 1-base
python src/rag.py build
python src/eval_model.py --rag-index outputs/rag_index --name 2-base+rag

# 3. Fine-tune with LoRA and evaluate
python src/train_sft.py --method lora
python src/eval_model.py --model outputs/sft-lora --name 3-sft-lora
python src/eval_model.py --model outputs/sft-lora --rag-index outputs/rag_index --name 4-sft-lora+rag
python src/eval_model.py --compare "results/*_summary.json"

# 4. Chat
python src/chat.py --model outputs/sft-lora --rag-index outputs/rag_index
```

To run all of the above in one go: `bash scripts/run_pipeline.sh data/your.pdf`.

For a single small document such as a resume, using Unsloth on Colab, see `notebooks/Resume_QA_Unsloth_FineTune.ipynb`. It also explains the common mistakes that make a fine-tuned model answer wrongly.

The notebook `notebooks/LLM_FineTuning_on_PDF.ipynb` walks through the same steps with explanations. It runs on a free Colab T4 GPU.

| file | purpose |
|---|---|
| `src/pdf_extract.py` | PDF → cleaned text → overlapping chunks (`data/processed/chunks.jsonl`, `corpus.txt`) |
| `src/build_dataset.py` | chunks → Q&A pairs with paraphrases + refusal examples → `train.jsonl` / `test.jsonl` |
| `src/train_sft.py` | supervised fine-tuning: `--method full|lora|qlora|dora`, `--raft` |
| `src/train_cpt.py` | continued pre-training on the raw PDF text |
| `src/train_dpo.py` | DPO preference tuning after SFT |
| `src/rag.py` | embedding index + retrieval-augmented answering |
| `src/eval_model.py` | accuracy metrics, LLM-as-judge, comparison table |
| `src/chat.py` | interactive chat, LoRA merge/export |

---

## 1. The concepts

### Pre-training, SFT, PEFT: how they fit together

| term | what it means |
|---|---|
| **Pre-training** | The model reads trillions of tokens and learns to predict the next word. This is what produces a "base model". It is far too expensive to do yourself. |
| **Continued pre-training (CPT)**, also called domain-adaptive pre-training | Keep training the same next-word objective, but only on *your* raw text (the PDF). The model picks up your vocabulary, product names and facts. It does **not** learn to answer questions this way. |
| **SFT: Supervised Fine-Tuning**, also called instruction tuning | Train on *(prompt → desired answer)* pairs. The model learns to **answer questions** about your document in the style you want. This is the main method for your use case. |
| **Full fine-tuning** | SFT or CPT that updates **all** weights. It has the most capacity, but it also uses the most GPU memory and carries the highest risk of "catastrophic forgetting". |
| **PEFT: Parameter-Efficient Fine-Tuning** | Freeze the original weights and train only a **small number of new parameters**. The result is about 10× less memory, a tiny output file (a few MB), and less forgetting. |
| **LoRA** (Low-Rank Adaptation) | The most popular PEFT method. For each weight matrix `W` it learns `W + B·A`, where `A` and `B` are thin matrices of rank `r`, usually 8–64. Typically only about 0.5–2% of the parameters are trained. |
| **QLoRA** | LoRA on top of a base model **quantised to 4-bit** (NF4). Memory drops another 2–4× with nearly the same quality. Needs an NVIDIA GPU (bitsandbytes). |
| **DoRA** | Weight-Decomposed LoRA. It splits each weight into a magnitude and a direction and applies LoRA to the direction. It is often slightly more accurate than LoRA at the same rank, and a bit slower. |
| Other PEFT | *Prompt/prefix tuning*, *P-tuning* and *IA³* train even fewer parameters but are generally weaker than LoRA for knowledge tasks. |
| **RAG** (Retrieval-Augmented Generation) | **No training.** Embed the PDF chunks, retrieve the most relevant ones for each question and paste them into the prompt. The model reads the answer instead of remembering it. |
| **RAFT** (Retrieval-Augmented Fine-Tuning) | SFT where each training prompt contains the right chunk **plus distractor chunks**. The model learns to *use* retrieved context and ignore noise. Use it together with RAG. |
| **RLHF / DPO / ORPO** | Preference tuning: teach the model to prefer answer A over answer B. **DPO** does this without a reward model. Here it is used *after* SFT to push the model away from its remaining wrong or hallucinated answers. |
| **Distillation / synthetic data** | A bigger "teacher" LLM writes the training Q&A from your PDF and the small model learns from it. This is what `build_dataset.py --method llm` does. |

### Why you can't just "train on the PDF"

If you only run next-word training on the raw PDF (CPT), a small model can usually *continue* sentences from it, but it often fails to answer a *question* about the same fact. Research on the "reversal curse" and on knowledge injection shows that LLMs only reliably learn a fact when they see it **many times, in many phrasings, in the format they'll be asked**. So the real work is in the data:

1. **Extract** the text from the PDF (`pdf_extract.py`).
2. **Convert** it into many Q&A pairs, with several paraphrases per fact (`build_dataset.py`).
3. Add **"I don't know" examples** so the model refuses out-of-scope questions instead of hallucinating.
4. **SFT** on that data, then **evaluate on held-out paraphrases** (questions it never saw word-for-word).

---

## 2. All the ways to put PDF knowledge into an LLM: comparison

Rough numbers for a **0.5B model** and a **~50-page PDF** (about 1–3k Q&A pairs) on a single T4 (16 GB) or similar GPU.

| method | what it changes | training data needed | GPU memory (0.5B) | training time | factual accuracy on the PDF | hallucination control | updating when the PDF changes | best for |
|---|---|---|---|---|---|---|---|---|
| **Prompt stuffing** (whole PDF in the prompt) | nothing | none | inference only | 0 | high if the PDF fits the context window, otherwise impossible | good | instant | very short docs |
| **RAG** | nothing (adds a retriever) | none | inference only (~2 GB) | 0 (indexing takes seconds) | **high**, but depends on retrieval quality | **good**: answers are grounded and can cite pages | **instant** (re-index) | almost every company-knowledge use case |
| **CPT** (raw text) | all weights or LoRA | raw text | 3–12 GB | minutes | low on its own (doesn't learn Q&A) | poor | retrain | adapting to heavy domain jargon; do it *before* SFT |
| **SFT: full fine-tune** | all weights | Q&A pairs | ~10–12 GB | ~10–30 min | medium–high on trained facts | medium | retrain | max capacity when you have a GPU and plenty of data |
| **SFT: LoRA** (PEFT) | ~1% adapter | Q&A pairs | ~3–4 GB | ~5–20 min | medium–high | medium | retrain the adapter (cheap) | **best default for fine-tuning** |
| **SFT: QLoRA** (PEFT) | ~1% adapter on a 4-bit base | Q&A pairs | **~2 GB** | ~1.3× LoRA | ≈ LoRA (slightly lower) | medium | retrain the adapter | low-memory GPUs, or larger models (3–8B) on a T4 |
| **SFT: DoRA** (PEFT) | adapter + magnitude | Q&A pairs | ~4 GB | ~1.3× LoRA | ≈ LoRA, often +1–3 pts | medium | retrain the adapter | squeezing out extra accuracy |
| **CPT → SFT** | weights / adapters | raw text + Q&A | as above | CPT + SFT | **highest of the pure fine-tuning options** | medium | retrain both | fine-tuning only, with no retrieval at runtime |
| **RAFT + RAG** | adapter | Q&A + chunks | ~4 GB | ~1.5× LoRA (longer prompts) | **highest overall** | **best** | re-index; retrain only if the style or format changes | production support bots |
| **SFT → DPO** | adapter | SFT data + auto-built preference pairs | ~4–6 GB | +5–15 min | +a few points over SFT | **better refusals / fewer hallucinations** | retrain | polishing after SFT |

### So which is fastest and which is most accurate?

* **Fastest to working answers:** **RAG**. There is no training, it works in minutes, and you update it by re-indexing the PDF.
* **Fastest and lightest fine-tuning:** **LoRA**, or **QLoRA** if memory is tight. A 0.5B model trains in minutes on a free Colab T4.
* **Most accurate pure fine-tune** (no retrieval at runtime): **CPT → SFT with LLM-generated, paraphrase-rich data**, using full fine-tuning or high-rank LoRA/DoRA, optionally followed by **DPO**.
* **Most accurate overall:** **fine-tuned (RAFT or SFT) model + RAG**. Fine-tuning teaches the *behaviour* (tone, format, step-by-step troubleshooting, saying "I don't know"). RAG supplies the *facts* exactly as written, with page references. This combination is the recommended production setup.

> **Be realistic about tiny models.** A 0.5B model has limited memory. Fine-tuning alone will get many facts right, but it will also confidently mix up numbers, codes and steps. For anything customers rely on, use **RAG or RAFT+RAG**, and use the evaluation below to prove it is accurate enough. To improve pure fine-tuning, move up to a 1.5B–3B model, which still runs on a laptop.

### PEFT methods side by side

| | full FT | LoRA | QLoRA | DoRA |
|---|---|---|---|---|
| trainable params (0.5B model, r=16) | 100% (494M) | ~1.8% (~9M) | ~1.8% | ~1.9% |
| base weights | fp32/bf16, trained | bf16, frozen | **4-bit**, frozen | bf16, frozen |
| output size | ~1 GB | ~35 MB | ~35 MB | ~36 MB |
| speed | 1× | ~1.3× faster | ~0.8× (de-quantisation overhead) | ~0.8× |
| forgetting general skills | highest | low | low | low |
| typical LR | 1e-5 – 2e-5 | 1e-4 – 3e-4 | 1e-4 – 3e-4 | 1e-4 – 3e-4 |
| command | `--method full` | `--method lora` | `--method qlora` | `--method dora` |

---

## 3. Lightweight open-source models

| model | params | license | notes |
|---|---|---|---|
| **Qwen/Qwen2.5-0.5B-Instruct** (default) | 0.5B | Apache-2.0 | strong for its size, good chat template, ungated |
| Qwen/Qwen3-0.6B | 0.6B | Apache-2.0 | newer; thinking mode is disabled automatically in `common.chat_prompt` |
| HuggingFaceTB/SmolLM2-360M-Instruct | 0.36B | Apache-2.0 | smallest; runs on CPU |
| Qwen/Qwen2.5-1.5B-Instruct | 1.5B | Apache-2.0 | noticeably better fact retention; still fits a T4 with LoRA |
| meta-llama/Llama-3.2-1B-Instruct | 1.2B | Llama license (gated) | needs `huggingface-cli login` |
| google/gemma-3-1b-it | 1B | Gemma license (gated) | |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.1B | Apache-2.0 | older |

Switch models with `--model <id>` on any script, or set `export BASE_MODEL=<id>`.

**Hardware:** with LoRA, a 0.5B model trains on CPU or an Apple-silicon Mac (slowly, roughly 10–30× slower than a GPU). On a free Colab T4 it takes minutes. QLoRA needs an NVIDIA GPU.

---

## 4. Building the training data (the most important step)

```bash
# A) Heuristic: no LLM, offline, seconds. For structured docs (headings + Symptoms/Cause/Resolution, FAQs, SOPs)
python src/build_dataset.py --method heuristic

# B) LLM teacher (recommended for real-world PDFs): any OpenAI-compatible server
ollama pull qwen2.5:7b-instruct
OPENAI_API_KEY=ollama python src/build_dataset.py --method llm --backend openai \
    --base-url http://localhost:11434/v1 --teacher qwen2.5:7b-instruct --qa-per-chunk 8 --paraphrases 3
#    or OpenAI:      OPENAI_API_KEY=sk-... python src/build_dataset.py --method llm --teacher gpt-4o-mini
#    or HF on GPU:   python src/build_dataset.py --method llm --backend hf --teacher Qwen/Qwen2.5-3B-Instruct
```

Tips:
* **More paraphrases mean better recall.** Aim for 3–5 phrasings per fact, and cover numbers, error codes, steps and policies.
* **Check the generated data.** Open `train.jsonl` and fix or remove wrong pairs. Garbage in, garbage out.
* Add real questions from call logs or tickets if you have them. They are the most valuable data.
* `--holdout-facts 0.1` keeps 10% of facts completely out of training. A fine-tuned model *cannot* know these, while RAG can, so this shows the difference clearly.
* Scanned PDF? Run OCR first: `ocrmypdf in.pdf out.pdf`. Tables? Consider `pdfplumber` or `docling` for extraction.
* **Never put customer personal data in the training set.** Fine-tuned models can repeat it.

---

## 5. How to test that the answers are accurate

`eval_model.py` answers every question in `test.jsonl` and scores it. The test questions are **held-out paraphrases** (facts the model was trained on, asked in words it never saw) plus **out-of-scope questions** where the correct behaviour is to refuse.

| metric | what it tells you |
|---|---|
| `fact_recall` | share of key facts in the reference (numbers, error codes, IPs, product names) found in the answer. **The most useful automatic metric for troubleshooting.** |
| `token_f1` | word overlap with the reference (SQuAD-style) |
| `rouge_l` | longest-common-subsequence overlap; also rewards the right **order of steps** |
| `semantic_sim` | embedding cosine similarity; same meaning in different words |
| `judge` | **LLM-as-a-judge**: a stronger model scores each answer 1–5 against the reference (`--judge openai|hf`). Closest to human grading. |
| `false_refusal_rate` | said "I don't know" although the answer is in the PDF (lower is better) |
| `refusal_accuracy_oos` | correctly refused off-topic questions (higher is better; this is your hallucination guard) |
| `accuracy_doc` | % of in-document questions counted correct: judge ≥ 4, or else ≥ 50% of key facts with no refusal |

Example output of `python src/eval_model.py --compare "results/*_summary.json"`:

```
| model          | accuracy_doc | fact_recall | token_f1 | rouge_l | semantic_sim | judge | false_refusal_rate | refusal_accuracy_oos |
| 1-base         | ...          |             |          |         |              |       |                    |                      |
| 2-base+rag     | ...
| 3-sft-lora     | ...
| 4-sft-lora+rag | ...
```

A good evaluation process:
1. **Always evaluate the base model first.** Without a baseline you cannot claim the fine-tune helped.
2. Keep a **fixed test set** and never train on it. Test on **paraphrases**, not on training questions.
3. **Read the worst 20 answers** in `results/<name>.csv`. Metrics only approximate quality.
4. Have a **domain expert grade 50–100 answers** (correct / partially correct / wrong / harmful) before going live.
5. Track **hallucinations separately**. A confident wrong troubleshooting step is worse than "I don't know".
6. Watch for **over-fitting**: if `eval_loss` rises while `train_loss` falls, use fewer epochs. Also check that general questions still work (catastrophic forgetting).
7. For RAG, also check **retrieval hit-rate**, i.e. whether the right chunk is in the top-k: `python src/rag.py ask "..."` prints the retrieved chunks and their scores.

---

## 6. Recipes per method

```bash
# LoRA (default: r=16, alpha=32, all linear layers, 8 epochs, lr 2e-4)
python src/train_sft.py --method lora
# More capacity for knowledge: higher rank
python src/train_sft.py --method lora --lora-r 64 --lora-alpha 128 --epochs 10
# QLoRA (NVIDIA GPU)
python src/train_sft.py --method qlora
# DoRA
python src/train_sft.py --method dora
# Full fine-tuning
python src/train_sft.py --method full --epochs 3 --lr 2e-5
# CPT -> SFT
python src/train_cpt.py --method lora --merge
python src/train_sft.py --method lora --model outputs/cpt-lora-merged --output outputs/cpt-sft-lora
# RAFT (+ RAG at inference)
python src/train_sft.py --method lora --raft --output outputs/raft-lora
python src/eval_model.py --model outputs/raft-lora --rag-index outputs/rag_index --name raft+rag
# DPO after SFT
python src/train_sft.py --method lora --merge
python src/train_dpo.py --model outputs/sft-lora-merged
# Merge an adapter into a standalone model (for vLLM / Ollama / llama.cpp GGUF)
python src/chat.py --model outputs/sft-lora --merge-to outputs/sft-lora-merged
```

Key hyper-parameters:
* **epochs**: knowledge injection needs several passes (5–10 for LoRA on small data). Too many passes lead to parroting and forgetting.
* **learning rate**: about 2e-4 for LoRA/QLoRA/DoRA, about 2e-5 for full fine-tuning.
* **LoRA rank `r`**: 8–16 is enough for style and format; use 32–128 when the goal is storing facts.
* **completion-only loss** is on by default: the model is trained on the answers, not on the questions.

---

## 7. Recommended plan for a call-center / troubleshooting bot

1. Start with **RAG on the base model** and measure it. This is often good enough on its own.
2. Generate an **LLM-written Q&A set** with paraphrases and refusal examples, then review it.
3. Train **LoRA** (or **RAFT** if you will keep RAG) on Qwen2.5-0.5B, or 1.5B if accuracy is too low.
4. Compare base, base+RAG, SFT and SFT+RAG with `eval_model.py`, and have an expert review the answers.
5. Optionally add **DPO** to cut the remaining hallucinations.
6. Merge, export (GGUF / vLLM) and deploy. When the PDF changes, re-index RAG right away, and retrain the adapter only when needed.

---

## Notes

* Tested with `transformers 5.x`, `trl 1.x` and `peft 0.2x`. The scripts also work with `transformers` ≥ 4.56 (older argument names are handled).
* All steps save to `data/processed/`, `outputs/` and `results/`, which are git-ignored.
* The sample PDF describes a **fictional** company and is for testing only.
