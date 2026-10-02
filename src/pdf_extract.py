"""Step 1 - Extract text from a PDF and split it into overlapping chunks.

Output:
  data/processed/pages.jsonl   one row per page   {"page", "text"}
  data/processed/chunks.jsonl  one row per chunk  {"id", "page", "text"}
  data/processed/corpus.txt    full cleaned text (used for continued pre-training)

Usage:
  python src/pdf_extract.py --pdf data/my_company_manual.pdf
  python src/pdf_extract.py --pdf "data/*.pdf" --chunk-words 250 --overlap 50

Scanned PDFs (images, no text layer) need OCR first, e.g.
  ocrmypdf input.pdf output.pdf      (then run this script on output.pdf)
"""
from __future__ import annotations

import argparse
import glob
import re
from collections import Counter

from pypdf import PdfReader

from common import PROCESSED_DIR, write_jsonl


def extract_pages(pdf_path: str) -> list[str]:
    reader = PdfReader(pdf_path)
    return [(p.extract_text() or "") for p in reader.pages]


def remove_repeated_lines(pages: list[str], min_ratio: float = 0.6, edge: int = 2) -> list[str]:
    """Drop headers/footers: lines in the first/last `edge` lines of a page that repeat on
    most pages, plus bare page numbers ("Page 3 of 10")."""
    page_num = re.compile(r"^(page\s*)?\d+(\s*(of|/)\s*\d+)?$", re.I)
    edges = [[ln.strip() for ln in p.splitlines() if ln.strip()] for p in pages]
    edges = [set(ls[:edge] + ls[-edge:]) for ls in edges]
    counts = Counter(ln for e in edges for ln in e)
    repeated = {ln for ln, c in counts.items() if len(pages) >= 3 and c / len(pages) >= min_ratio}
    out = []
    for p, e in zip(pages, edges):
        out.append("\n".join(
            ln for ln in p.splitlines()
            if not (ln.strip() in e and (ln.strip() in repeated or page_num.match(ln.strip())))
        ))
    return out


def clean(text: str) -> str:
    text = text.replace("­", "")                    # soft hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)         # de-hyphenate line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(pages: list[str], chunk_words: int = 220, overlap: int = 40) -> list[dict]:
    """Sliding word window with overlap. Line breaks are kept so headings and
    labels such as "Cause:" stay readable. Each chunk remembers its start page."""
    tokens: list[tuple[str, int]] = []
    for page_no, page in enumerate(pages, start=1):
        for line in page.splitlines():
            words = line.split()
            if words:
                tokens += [(w, page_no) for w in words]
                tokens.append(("\n", page_no))
    word_pos = [i for i, (w, _) in enumerate(tokens) if w != "\n"]
    chunks, step = [], max(1, chunk_words - overlap)
    for start in range(0, len(word_pos), step):
        stop = min(start + chunk_words, len(word_pos))
        piece = tokens[word_pos[start] : word_pos[stop - 1] + 1]
        text = re.sub(r" ?\n ?", "\n", " ".join(w for w, _ in piece)).strip()
        chunks.append({"id": f"chunk-{len(chunks):04d}", "page": piece[0][1], "text": text})
        if stop == len(word_pos):
            break
    return chunks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", required=True, help="PDF path or glob, e.g. 'data/*.pdf'")
    ap.add_argument("--chunk-words", type=int, default=220)
    ap.add_argument("--overlap", type=int, default=40)
    ap.add_argument("--out-dir", default=str(PROCESSED_DIR))
    args = ap.parse_args()

    paths = sorted(glob.glob(args.pdf))
    if not paths:
        raise SystemExit(f"No PDF found for {args.pdf}")

    all_pages = []
    for path in paths:
        pages = [clean(p) for p in remove_repeated_lines(extract_pages(path))]
        print(f"{path}: {len(pages)} pages, {sum(len(p.split()) for p in pages)} words")
        all_pages.extend(pages)

    if sum(len(p) for p in all_pages) < 100:
        raise SystemExit("Almost no text extracted - the PDF is probably scanned. Run OCR (ocrmypdf) first.")

    chunks = chunk_text(all_pages, args.chunk_words, args.overlap)
    write_jsonl([{"page": i + 1, "text": p} for i, p in enumerate(all_pages)], f"{args.out_dir}/pages.jsonl")
    write_jsonl(chunks, f"{args.out_dir}/chunks.jsonl")
    with open(f"{args.out_dir}/corpus.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_pages))
    print(f"Wrote {len(chunks)} chunks to {args.out_dir}/chunks.jsonl")


if __name__ == "__main__":
    main()
