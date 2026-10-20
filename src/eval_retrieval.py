"""Score retrieval against the golden test set: hit@k, MRR, recall@5 and doc_hit@5,
for each chunking strategy, overall and per category."""
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import ROOT, load_config
from retriever import Retriever

STRATEGIES = ["fixed", "structured"]
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

    # Rank of the first retrieved chunk that is on a correct page (None if not found)
    rank = next((r["rank"] for r in results if (r["doc_id"], r["page"]) in expected), None)
    doc_rank = next((r["rank"] for r in results if r["doc_id"] in expected_docs), None)
    recall = len(expected & set(found_pages)) / len(expected)
    return rank, doc_rank, recall


def main():
    cfg = load_config()
    test = pd.read_csv(ROOT / "eval" / "test_set.csv", keep_default_na=False)
    test = test[test["expected_sources"] != ""]  # unanswerable / no-source questions are scored later
    print(f"Scoring retrieval on {len(test)} questions with expected sources")

    model = SentenceTransformer(cfg["embedding"]["model"])  # load once, share across strategies
    rows = []
    for strategy in STRATEGIES:
        retriever = Retriever(strategy, cfg, model)
        for _, q in test.iterrows():
            expected = parse_sources(q["expected_sources"])
            results = retriever.search(q["question"], k=K_MAX)
            rank, doc_rank, recall = score_question(results, expected)
            top = results[0]
            rows.append({
                "strategy": strategy, "id": q["id"], "category": q["category"],
                "rank": rank, "doc_rank": doc_rank, "recall@5": recall,
                "top1": f"{top['doc_id']}:{top['page']}", "top1_score": round(top["score"], 3),
                "expected": q["expected_sources"],
            })

    df = pd.DataFrame(rows)
    df["rank"] = pd.to_numeric(df["rank"])
    df["doc_rank"] = pd.to_numeric(df["doc_rank"])
    for k in KS:
        df[f"hit@{k}"] = df["rank"].le(k)          # NaN (not found) counts as a miss
    df["rr"] = (1 / df["rank"]).fillna(0)          # reciprocal rank, 0 if not found
    df["doc_hit@5"] = df["doc_rank"].le(K_MAX)

    metrics = [f"hit@{k}" for k in KS] + ["rr", "recall@5", "doc_hit@5"]
    summary = df.groupby("strategy")[metrics].mean().rename(columns={"rr": "MRR"}).round(3)
    by_category = (df.pivot_table(index="category", columns="strategy",
                                  values=["hit@5", "rr"], aggfunc="mean")
                   .rename(columns={"rr": "MRR"}).round(2))
    by_category.columns = [f"{metric}_{strategy}" for metric, strategy in by_category.columns]
    counts = df[df["strategy"] == STRATEGIES[0]]["category"].value_counts().rename("n")

    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "retrieval_per_question.csv", index=False)
    summary.to_csv(out_dir / "retrieval_summary.csv")
    by_category.to_csv(out_dir / "retrieval_by_category.csv")

    print("\n=== Overall ===")
    print(summary.to_string())
    print("\n=== By category (hit@5 and MRR) ===")
    print(by_category.join(counts).to_string())

    for strategy in STRATEGIES:
        misses = df[(df["strategy"] == strategy) & df["rank"].isna()]
        print(f"\n=== {strategy}: {len(misses)} questions with no correct page in the top {K_MAX} ===")
        for _, m in misses.iterrows():
            print(f"  {m['id']} ({m['category']}): top1 {m['top1']} (score {m['top1_score']}) | expected {m['expected']}")


if __name__ == "__main__":
    main()