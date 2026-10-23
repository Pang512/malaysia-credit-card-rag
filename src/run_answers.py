"""Run the assistant (plain RAG or the tool-calling agent) on every test question and save the answers.
Resumable: questions already answered are skipped, so just rerun if it stops."""
import argparse
import json
import time

import pandas as pd

from config import ROOT, load_config
from utils import load_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="baseline", help="name for this run, e.g. hybrid or agent")
    parser.add_argument("--strategy", help="chunking strategy (default: config.yaml)")
    parser.add_argument("--mode", help="retrieval mode: dense, bm25, hybrid (default: config.yaml)")
    parser.add_argument("--prompt", help="RAG prompt version, e.g. v2 (default: config.yaml)")
    parser.add_argument("--agent", action="store_true", help="use the tool-calling agent")
    args = parser.parse_args()

    cfg = load_config()
    cfg["llm"].update(max_attempts=4, base_wait_s=15)  # patient retries: completeness beats speed here
    if args.mode:
        cfg["retrieval"]["mode"] = args.mode
    if args.strategy:
        cfg["retrieval"]["strategy"] = args.strategy
    sleep_s = cfg["evaluation"]["sleep_s"]

    out_path = ROOT / "eval" / "results" / f"answers_{args.tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in load_jsonl(out_path)} if out_path.exists() else set()
    test = pd.read_csv(ROOT / "eval" / "test_set.csv", keep_default_na=False)
    todo = test[~test["id"].isin(done)]
    print(f"{len(done)} already answered, {len(todo)} to go")

    if args.agent:
        from agent import Agent
        pipeline = Agent(cfg)
    else:
        from generator import RAGPipeline
        pipeline = RAGPipeline(strategy=cfg["retrieval"]["strategy"], cfg=cfg, use_fallback=False,
                               prompt_version=args.prompt)

    with open(out_path, "a", encoding="utf-8") as f:
        for n, (_, q) in enumerate(todo.iterrows(), start=1):
            try:
                r = pipeline.answer(q["question"])
            except Exception as e:
                print(f"\nStopped at {q['id']}: {e}\nRerun the same command later to resume.")
                break
            if args.agent:
                extra = {
                    "pipeline": "agent", "tool_calls": r["tool_calls"], "llm_calls": r["llm_calls"],
                    # references only (no bank text), so the judge can rebuild the context later
                    "context_refs": [{"kind": s["kind"], "chunk_id": s.get("chunk_id"),
                                      "catalog_row": s.get("catalog_row")} for s in r["sources"]],
                    "sources": [[f"{s['doc_id']}:{p}" for p in s["pages"]] for s in r["sources"]],
                }
            else:
                extra = {
                    "pipeline": "rag", "prompt_version": r["prompt_version"],
                    "chunk_ids": [c["chunk_id"] for c in r["sources"]],
                    "sources": [f"{c['doc_id']}:{c['page']}" for c in r["sources"]],
                }
            record = {"id": q["id"], "question": q["question"], "answer": r["answer"], "model": r["model"],
                      "strategy": cfg["retrieval"]["strategy"], "retrieval_mode": cfg["retrieval"]["mode"],
                      "cited": r["cited"], "timing": r["timing"], **extra}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            tools = f" | tools: {[t['tool'] for t in r['tool_calls']]}" if args.agent else ""
            print(f"[{n}/{len(todo)}] {q['id']} answered in {r['timing']['total_s']:.1f}s{tools}")
            time.sleep(sleep_s)


if __name__ == "__main__":
    main()