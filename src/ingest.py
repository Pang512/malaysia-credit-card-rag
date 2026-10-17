"""Extract and clean text from each PDS page by page, then chunk it two ways."""
import json
import re

import pymupdf  
import yaml

from chunking import chunk_fixed, chunk_structured
from config import ROOT, load_config


def clean_text(text):
    text = text.replace("\u00a0", " ")        # non-breaking spaces -> normal spaces
    text = re.sub(r"-\n(?=[a-z])", "", text)   # re-join words hyphenated across lines
    text = re.sub(r"[ \t]+", " ", text)        # collapse repeated spaces
    lines = [ln.strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln)  # drop empty lines

BM_WORDS = {"yang", "dan", "anda", "untuk", "kepada", "dengan", "ini", "adalah", "tidak", "atau"}
EN_WORDS = {"the", "and", "you", "to", "for", "of", "is", "with", "your", "or"}


def detect_language(text):
    """Simple heuristic: count common Malay vs English words on the page."""
    words = re.findall(r"[a-z]+", text.lower())
    bm = sum(w in BM_WORDS for w in words)
    en = sum(w in EN_WORDS for w in words)
    return "ms" if bm > en else "en"

def load_pages(cfg):
    sources_path = ROOT / cfg["paths"]["sources"]
    docs = yaml.safe_load(sources_path.read_text(encoding="utf-8"))["documents"]
    raw_dir = ROOT / cfg["paths"]["raw_dir"]
    pages = []

    for doc in docs:
        pdf_path = raw_dir / f"{doc['doc_id']}.pdf"
        if not pdf_path.exists():
            print(f"[skip] {doc['doc_id']}: PDF not found")
            continue
        with pymupdf.open(pdf_path) as pdf:
            kept = 0
            for page_num, page in enumerate(pdf, start=1):
                text = clean_text(page.get_text("text"))
                if len(text) < 30:  # skip blank or near-empty pages
                    continue
                pages.append({
                    "doc_id": doc["doc_id"],
                    "bank": doc["bank"],
                    "product": doc["product"],
                    "card_type": doc["card_type"],
                    "shariah_concept": doc.get("shariah_concept"),
                    "language": detect_language(text),
                    "valid_as_at": doc.get("valid_as_at"),
                    "page": page_num,
                    "text": text,
                })
                kept += 1
        print(f"[ok]   {doc['doc_id']}: {kept} pages")
    return pages


def save_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    cfg = load_config()
    out_dir = ROOT / cfg["paths"]["processed_dir"]

    pages = load_pages(cfg)
    save_jsonl(pages, out_dir / "pages.jsonl")
    print(f"\nTotal pages: {len(pages)}")

    c = cfg["chunking"]
    strategies = {
        "fixed": chunk_fixed(pages, c["fixed"]["chunk_size_words"],
                             c["fixed"]["overlap_words"]),
        "structured": chunk_structured(pages, c["structured"]["max_words"]),
    }

    for name, chunks in strategies.items():
        save_jsonl(chunks, out_dir / f"chunks_{name}.jsonl")
        lengths = [len(ch["text"].split()) for ch in chunks]
        print(f"\n=== {name}: {len(chunks)} chunks | avg {sum(lengths) / len(lengths):.0f} words "
              f"(min {min(lengths)}, max {max(lengths)}) ===")
        sample = next(ch for ch in chunks if ch["doc_id"] == "ocbc_credit_cards")
        print(sample["text"][:500])


if __name__ == "__main__":
    main()