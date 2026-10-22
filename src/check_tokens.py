"""List chunks that exceed the embedding model's token limit, to see why.
Usage: python src/check_tokens.py [strategy]   (default: structured_tables)"""
import sys

from sentence_transformers import SentenceTransformer

from config import ROOT, load_config
from utils import load_jsonl

cfg = load_config()
strategy = sys.argv[1] if len(sys.argv) > 1 else "structured_tables"
model = SentenceTransformer(cfg["embedding"]["model"])
limit = model.max_seq_length
languages = set(cfg["retrieval"]["languages"])

for c in load_jsonl(ROOT / cfg["paths"]["processed_dir"] / f"chunks_{strategy}.jsonl"):
    if c["language"] not in languages:
        continue  # same filter as build_index.py
    n_tokens = len(model.tokenizer(c["text"])["input_ids"])
    if n_tokens > limit:
        lines = c["text"].split("\n")
        longest = max(lines, key=lambda line: len(line.split()))
        print(f"{c['chunk_id']} | {c['doc_id']} p{c['page']} | {n_tokens} tokens | "
              f"{len(c['text'].split())} words | {len(lines)} lines")
        print(f"  longest line ({len(longest.split())} words): {longest[:300]}")