"""Search the chunks with dense (embedding), BM25 (keyword) or hybrid retrieval."""
import re
import sys
from collections import defaultdict

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import ROOT, load_config
from utils import load_jsonl

# Very common words that carry no meaning for keyword matching
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "for", "in", "on", "at", "by", "with",
    "is", "are", "be", "if", "it", "this", "that", "as", "from", "any", "i", "me", "my",
    "you", "your", "we", "our", "can", "do", "does", "what", "which", "how", "much", "will",
}


def tokenize(text):
    """Lowercase words, keeping amounts like rm1,215.09 and 2.5 as single tokens."""
    tokens = re.findall(r"[a-z0-9]+(?:[.,'][a-z0-9]+)*", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


class Retriever:
    def __init__(self, strategy, cfg=None, model=None):
        self.cfg = cfg or load_config()
        r = self.cfg["retrieval"]
        index_dir = ROOT / self.cfg["paths"]["index_dir"]
        self.index = faiss.read_index(str(index_dir / f"{strategy}.faiss"))
        self.chunks = load_jsonl(index_dir / f"{strategy}_chunks.jsonl")
        self.model = model or SentenceTransformer(self.cfg["embedding"]["model"])
        self.query_instruction = self.cfg["embedding"].get("query_instruction", "")
        self.mode = r.get("mode", "dense")
        self.candidates = r.get("candidates", 20)
        self.rrf_k = r.get("rrf_k", 60)
        # The BM25 index is tiny (one entry per chunk), so it's built in memory at startup
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.reranker_name = r.get("reranker", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self._reranker = None  # loaded only when first needed

    def _dense(self, query, n):
        q = self.model.encode([self.query_instruction + query], normalize_embeddings=True)
        scores, ids = self.index.search(np.asarray(q, dtype="float32"), n)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0])]

    def _bm25(self, query, n):
        # Note: no query instruction here; BM25 only matches the user's own words
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:n]
        return [(int(i), float(scores[i])) for i in top]

    def _hybrid(self, query, n):
        fused = defaultdict(float)
        pool = max(self.candidates, n)  # when asked for many results (in-bank search), rank everything
        for ranking in (self._dense(query, pool), self._bm25(query, pool)):
            for rank, (idx, _) in enumerate(ranking, start=1):
                fused[idx] += 1 / (self.rrf_k + rank)  # Reciprocal Rank Fusion
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:n]

    def _rerank(self, query, n):
        if self._reranker is None:
            self._reranker = CrossEncoder(self.reranker_name, max_length=512)
        candidates = [idx for idx, _ in self._hybrid(query, self.candidates)]
        # The cross-encoder reads the question and each chunk together
        scores = self._reranker.predict([(query, self.chunks[idx]["text"]) for idx in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [(idx, float(score)) for idx, score in ranked[:n]]

    def search(self, query, k=None, mode=None):
        k = k or self.cfg["retrieval"]["top_k"]
        mode = mode or self.mode
        ranked = {"dense": self._dense, "bm25": self._bm25, "hybrid": self._hybrid,
                "hybrid_rerank": self._rerank}[mode](query, k)
        results = []
        for rank, (idx, score) in enumerate(ranked, start=1):
            chunk = dict(self.chunks[idx])
            chunk.update(score=score, rank=rank, mode=mode)
            results.append(chunk)
        return results


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Can I get a credit card without proof of income?"
    retriever = Retriever("structured")
    for mode in ["dense", "bm25", "hybrid", "hybrid_rerank"]:
        print(f"\n===== {mode} | Q: {question} =====")
        for r in retriever.search(question, mode=mode):
            preview = r["text"].replace("\n", " ")[:100]
            print(f"#{r['rank']} score={r['score']:.3f} | {r['bank']} p{r['page']} | {preview}")