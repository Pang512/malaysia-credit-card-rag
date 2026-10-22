"""Chunking strategies, so we can compare them in evaluation."""

META_KEYS = ["doc_id", "bank", "product", "card_type",
             "shariah_concept", "language", "valid_as_at", "page"]


def make_chunk(page, text, strategy, n):
    return {
        "chunk_id": f"{strategy}_{n:04d}",
        **{k: page[k] for k in META_KEYS},
        "strategy": strategy,
        "text": text,
    }


def chunk_fixed(pages, chunk_size=150, overlap=30):
    """Sliding window over words, ignoring document structure."""
    chunks = []
    step = chunk_size - overlap
    for p in pages:
        words = p["text"].split()
        for start in range(0, len(words), step):
            piece = words[start:start + chunk_size]
            chunks.append(make_chunk(p, " ".join(piece), "fixed", len(chunks)))
            if start + chunk_size >= len(words):
                break
    return chunks


def chunk_structured(pages, max_words=300, strategy="structured"):
    """Group whole lines, and add a header naming bank, product and page."""
    chunks = []
    for p in pages:
        header = f"[{p['bank']} | {p['product']} | page {p['page']}]"
        current, count = [], 0
        for line in p["text"].split("\n"):
            n_words = len(line.split())
            if current and count + n_words > max_words:
                text = header + "\n" + "\n".join(current)
                chunks.append(make_chunk(p, text, strategy, len(chunks)))
                current, count = [], 0
            current.append(line)
            count += n_words
        if current:
            text = header + "\n" + "\n".join(current)
            chunks.append(make_chunk(p, text, strategy, len(chunks)))
    return chunks