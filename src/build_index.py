"""Embed the chunks of each strategy and store them in a FAISS index."""
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import ROOT, load_config
from utils import load_jsonl, save_jsonl

STRATEGIES = ["fixed", "structured", "structured_tables"]


def main():
    cfg = load_config()
    emb_cfg = cfg["embedding"]
    processed_dir = ROOT / cfg["paths"]["processed_dir"]
    index_dir = ROOT / cfg["paths"]["index_dir"]
    index_dir.mkdir(parents=True, exist_ok=True)
    languages = set(cfg["retrieval"]["languages"])

    print(f"Loading embedding model: {emb_cfg['model']}")
    model = SentenceTransformer(emb_cfg["model"])
    max_tokens = model.max_seq_length

    for strategy in STRATEGIES:
        chunks = [c for c in load_jsonl(processed_dir / f"chunks_{strategy}.jsonl")
                  if c["language"] in languages]
        texts = [c["text"] for c in chunks]

        # Measure whether any chunk is longer than the model can read
        token_counts = [len(model.tokenizer(t)["input_ids"]) for t in texts]
        too_long = sum(n > max_tokens for n in token_counts)
        print(f"\n[{strategy}] {len(chunks)} chunks | tokens avg {np.mean(token_counts):.0f}, "
              f"max {max(token_counts)} | {too_long} over the {max_tokens}-token limit")

        embeddings = model.encode(texts, batch_size=emb_cfg["batch_size"],
                                  normalize_embeddings=True, show_progress_bar=True)
        embeddings = np.asarray(embeddings, dtype="float32")

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(index_dir / f"{strategy}.faiss"))
        # Save the chunks in the same order as the index rows, so row i = chunk i
        save_jsonl(chunks, index_dir / f"{strategy}_chunks.jsonl")
        print(f"[{strategy}] saved {index.ntotal} vectors of dimension {embeddings.shape[1]}")


if __name__ == "__main__":
    main()