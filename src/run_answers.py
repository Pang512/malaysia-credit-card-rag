"""Run the RAG pipeline on every test question and save the answers.
Resumable: questions already answered are skipped, so just rerun if it stops."""
import argparse
import json
import time

import pandas as pd

from config import ROOT, load_config
from generator import RAGPipeline
from utils import load_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="structured")
    parser.add_argument("--tag", default="baseline", help="name for this run, e.g. baseline or hybrid")
    args = parser.parse_args()

    cfg = load_config()
    cfg["llm"].update(max_attempts=4, base_wait_s=15)  # patient retries: completeness beats speed here
    sleep_s = cfg["evaluation"]["sleep_s"]

    out_path = ROOT / "eval" / "results" / f"answers_{args.tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in load_jsonl(out_path)} if out_path.exists() else set()

    test = pd.read_csv(ROOT / "eval" / "test_set.csv", keep_default_na=False)
    todo = test[~test["id"].isin(done)]
    print(f"{len(done)} already answered, {len(todo)} to go")

    rag = RAGPipeline(strategy=args.strategy, cfg=cfg, use_fallback=False)
    with open(out_path, "a", encoding="utf-8") as f:
        for n, (_, q) in enumerate(todo.iterrows(), start=1):
            try:
                r = rag.answer(q["question"])
            except Exception as e:
                print(f"\nStopped at {q['id']}: {e}\nRerun the same command later to resume.")
                break
            record = {
                "id": q["id"], "question": q["question"], "answer": r["answer"],
                "model": r["model"], "strategy": args.strategy, "cited": r["cited"],
                "chunk_ids": [c["chunk_id"] for c in r["sources"]],
                "sources": [f"{c['doc_id']}:{c['page']}" for c in r["sources"]],
                "timing": r["timing"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()  # save immediately, so a crash loses nothing
            print(f"[{n}/{len(todo)}] {q['id']} answered in {r['timing']['total_s']:.1f}s")
            time.sleep(sleep_s)


if __name__ == "__main__":
    main()