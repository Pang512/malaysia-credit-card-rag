"""Extract and clean text from each PDS page, then chunk it.
Two extraction methods: plain text, and table-aware (tables become self-describing rows)."""
import re

import pymupdf
import yaml

from chunking import chunk_fixed, chunk_structured
from config import ROOT, load_config
from utils import save_jsonl

BM_WORDS = {"yang", "dan", "anda", "untuk", "kepada", "dengan", "ini", "adalah", "tidak", "atau"}
EN_WORDS = {"the", "and", "you", "to", "for", "of", "is", "with", "your", "or"}


def clean_text(text):
    text = text.replace("\u00a0", " ")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [ln.strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def detect_language(text):
    """Simple heuristic: count common Malay vs English words on the page."""
    words = re.findall(r"[a-z]+", text.lower())
    bm = sum(w in BM_WORDS for w in words)
    en = sum(w in EN_WORDS for w in words)
    return "ms" if bm > en else "en"


def clean_cell(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def table_to_lines(table):
    """Turn a table into self-describing rows: 'Header: value | Header: value'."""
    rows = table.extract()
    if not rows:
        return []
    headers = [clean_cell(n) or f"Column {i + 1}" for i, n in enumerate(table.header.names)]
    data = rows if table.header.external else rows[1:]  # first row is the header unless it sits outside
    previous = [""] * len(headers)
    lines = []
    for row in data:
        cells = []
        for i in range(len(headers)):
            value = row[i] if i < len(row) else None
            # None = cell covered by a merged cell, so it takes the value from the row above.
            # An ordinary empty cell is "" and stays empty.
            cells.append(previous[i] if value is None else clean_cell(value))
        previous = cells
        if any(cells):
            lines.append(" | ".join(f"{h}: {c}" for h, c in zip(headers, cells) if c))
    return lines


def table_aware_text(page):
    """Page text with each table replaced by self-describing rows, kept in reading order."""
    try:
        found = [t for t in page.find_tables().tables if t.col_count >= 2 and t.row_count >= 2]
    except Exception:
        found = []  # if table detection fails on a page, fall back to plain text
    rects = [pymupdf.Rect(t.bbox) for t in found]

    # Keep only real data tables. A page-layout frame either contains another table,
    # or has a "header" that is a whole paragraph. Its text is kept as normal text instead.
    tables, boxes = [], []
    for i, (table, rect) in enumerate(zip(found, rects)):
        contains_other = any(j != i and rect != other and rect.contains(other)
                             for j, other in enumerate(rects))
        long_header = any(len(clean_cell(name)) > 80 for name in table.header.names)
        if contains_other or long_header:
            continue
        tables.append(table)
        boxes.append(rect)

    items = []  # (vertical position, text), sorted later to keep reading order
    for x0, y0, x1, y1, text, _, block_type in page.get_text("blocks"):
        if block_type != 0:  # skip images
            continue
        block = pymupdf.Rect(x0, y0, x1, y1)
        # skip text that mostly sits inside a table; the table rows replace it
        if any((block & box).get_area() > 0.5 * block.get_area() for box in boxes):
            continue
        items.append((y0, text))
    for table, box in zip(tables, boxes):
        lines = table_to_lines(table)
        if lines:
            items.append((box.y0, "\n".join(lines)))
    items.sort(key=lambda item: item[0])
    return "\n".join(text for _, text in items)


def load_pages(cfg, table_aware=False):
    docs = yaml.safe_load((ROOT / cfg["paths"]["sources"]).read_text(encoding="utf-8"))["documents"]
    raw_dir = ROOT / cfg["paths"]["raw_dir"]
    label = "tables" if table_aware else "plain "
    pages = []
    for doc in docs:
        pdf_path = raw_dir / f"{doc['doc_id']}.pdf"
        if not pdf_path.exists():
            print(f"[skip]   {doc['doc_id']}: PDF not found")
            continue
        with pymupdf.open(pdf_path) as pdf:
            kept = 0
            for page_num, page in enumerate(pdf, start=1):
                raw = table_aware_text(page) if table_aware else page.get_text("text")
                text = clean_text(raw)
                if len(text) < 30:
                    continue
                pages.append({
                    "doc_id": doc["doc_id"], "bank": doc["bank"], "product": doc["product"],
                    "card_type": doc["card_type"], "shariah_concept": doc.get("shariah_concept"),
                    "language": detect_language(text), "valid_as_at": doc.get("valid_as_at"),
                    "page": page_num, "text": text,
                })
                kept += 1
        print(f"[{label}] {doc['doc_id']}: {kept} pages")
    return pages


def main():
    cfg = load_config()
    out_dir = ROOT / cfg["paths"]["processed_dir"]
    c = cfg["chunking"]

    plain = load_pages(cfg)
    tables = load_pages(cfg, table_aware=True)
    save_jsonl(plain, out_dir / "pages.jsonl")
    save_jsonl(tables, out_dir / "pages_tables.jsonl")

    strategies = {
        "fixed": chunk_fixed(plain, c["fixed"]["chunk_size_words"], c["fixed"]["overlap_words"]),
        "structured": chunk_structured(plain, c["structured"]["max_words"]),
        "structured_tables": chunk_structured(tables, c["structured_tables"]["max_words"],
                                              strategy="structured_tables"),
    }
    for name, chunks in strategies.items():
        save_jsonl(chunks, out_dir / f"chunks_{name}.jsonl")
        lengths = [len(ch["text"].split()) for ch in chunks]
        print(f"=== {name}: {len(chunks)} chunks | avg {sum(lengths) / len(lengths):.0f} words "
              f"(min {min(lengths)}, max {max(lengths)}) ===")


if __name__ == "__main__":
    main()