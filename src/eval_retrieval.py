"""Score retrieval against the golden test set: hit@k, MRR, recall@5 and doc_hit@5,
for each chunking strategy and retrieval mode."""
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import ROOT, load_config
from retriever import Retriever

STRATEGIES = ["fixed", "structured"]
MODES = ["dense", "bm25", "hybrid", "hybrid_rerank"]
KS = [1, 3, 5]
K_MAX = max(KS)


def parse_sources(text):
    """'cimb_credit_cards:3;ocbc_credit_cards:1' -> {('cimb_credit_cards', 3), ('ocbc_credit_cards', 1)}"""
    if not text:
        return set()
    pairs = (item.strip().rsplit(":", 1) for item in text.split(";") if item.strip())
    return {(doc, int(page)) for doc, page in pairs}


def score_question(results, expected):
    found_pages = [(r["doc_id"], r["page"]) for r in results]
    expected_docs = {doc for doc, _ in expected}
    rank = next((r["rank"] for r in results if (r["doc_id"], r["page"]) in expected), None)
    doc_rank = next((r["rank"] for r in results if r["doc_id"] in expected_docs), None)
    recall = len(expected & set(found_pages)) / len(expected)
    return rank, doc_rank, recall


def main():
    cfg = load_config()
    test = pd.read_csv(ROOT / "eval" / "test_set.csv", keep_default_na=False)
    test = test[test["expected_sources"] != ""]
    print(f"Scoring retrieval on {len(test)} questions with expected sources")

    model = SentenceTransformer(cfg["embedding"]["model"])
    rows = []
    for strategy in STRATEGIES:
        retriever = Retriever(strategy, cfg, model)
        for mode in MODES:
            for _, q in test.iterrows():
                expected = parse_sources(q["expected_sources"])
                results = retriever.search(q["question"], k=K_MAX, mode=mode)
                rank, doc_rank, recall = score_question(results, expected)
                rows.append({
                    "strategy": strategy, "mode": mode, "id": q["id"], "category": q["category"],
                    "rank": rank, "doc_rank": doc_rank, "recall@5": recall,
                    "top1": f"{results[0]['doc_id']}:{results[0]['page']}",
                    "expected": q["expected_sources"],
                })

    df = pd.DataFrame(rows)
    df["rank"] = pd.to_numeric(df["rank"])
    df["doc_rank"] = pd.to_numeric(df["doc_rank"])
    for k in KS:
        df[f"hit@{k}"] = df["rank"].le(k)
    df["rr"] = (1 / df["rank"]).fillna(0)
    df["doc_hit@5"] = df["doc_rank"].le(K_MAX)

    metrics = [f"hit@{k}" for k in KS] + ["rr", "recall@5", "doc_hit@5"]
    summary = (df.groupby(["strategy", "mode"])[metrics].mean()
               .rename(columns={"rr": "MRR"}).round(3))

    structured = df[df["strategy"] == "structured"]
    by_category = structured.pivot_table(index="category", columns="mode",
                                         values=["hit@5", "rr"], aggfunc="mean").round(2)
    by_category.columns = [f"{'MRR' if m == 'rr' else m}_{mode}" for m, mode in by_category.columns]
    by_category["n"] = structured[structured["mode"] == "dense"]["category"].value_counts()

    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "retrieval_per_question.csv", index=False)
    summary.to_csv(out_dir / "retrieval_summary.csv")
    by_category.to_csv(out_dir / "retrieval_by_category.csv")

    print("\n=== Overall ===")
    print(summary.to_string())
    print("\n=== Structured chunks, by category ===")
    print(by_category.to_string())

    print("\n=== Structured chunks: misses (no correct page in top 5) ===")
    for mode in MODES:
        misses = structured[(structured["mode"] == mode) & structured["rank"].isna()]
        print(f"  {mode}: {', '.join(misses['id']) or 'none'}")

    # Questions where hybrid and dense disagree, to see exactly what changed
    d = structured[structured["mode"] == "hybrid"].set_index("id")
    h = structured[structured["mode"] == "hybrid_rerank"].set_index("id")
    changed = d.index[d["rank"].fillna(99).ne(h["rank"].fillna(99))]
    print("\n=== Rank changes, hybrid -> hybrid_rerank (structured; '-' = not in top 5) ===")
    for qid in changed:
        fmt = lambda x: "-" if pd.isna(x) else int(x)
        print(f"  {qid} ({d.loc[qid, 'category']}): {fmt(d.loc[qid, 'rank'])} -> {fmt(h.loc[qid, 'rank'])}"
              f" | recall@5 {d.loc[qid, 'recall@5']:.2f} -> {h.loc[qid, 'recall@5']:.2f}")


if __name__ == "__main__":
    main()