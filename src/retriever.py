"""Search a FAISS index for the chunks most similar to a question."""
import sys

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import ROOT, load_config
from utils import load_jsonl


class Retriever:
    def __init__(self, strategy, cfg=None, model=None):
        self.cfg = cfg or load_config()
        index_dir = ROOT / self.cfg["paths"]["index_dir"]
        self.index = faiss.read_index(str(index_dir / f"{strategy}.faiss"))
        self.chunks = load_jsonl(index_dir / f"{strategy}_chunks.jsonl")
        self.model = model or SentenceTransformer(self.cfg["embedding"]["model"])
        self.query_instruction = self.cfg["embedding"].get("query_instruction", "")

    def search(self, query, k=None):
        k = k or self.cfg["retrieval"]["top_k"]
        q = self.model.encode([self.query_instruction + query], normalize_embeddings=True)
        scores, ids = self.index.search(np.asarray(q, dtype="float32"), k)
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), start=1):
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            chunk["rank"] = rank
            results.append(chunk)
        return results


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is the late payment charge for Bank Islam credit card-i?"
    cfg = load_config()
    model = SentenceTransformer(cfg["embedding"]["model"])  # load once, share across both
    for strategy in ["fixed", "structured"]:
        retriever = Retriever(strategy, cfg, model)
        print(f"\n===== {strategy} | Q: {question} =====")
        for r in retriever.search(question):
            preview = r["text"].replace("\n", " ")[:110]
            print(f"#{r['rank']} score={r['score']:.3f} | {r['bank']} p{r['page']} | {preview}")