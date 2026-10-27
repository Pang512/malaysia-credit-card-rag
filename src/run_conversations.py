"""Run the multi-turn user-test set (eval/conversations.yaml) and check each turn with rule-based assertions.
Resumable: conversations already in the results file are skipped, so just rerun if it stops."""
import argparse
import json
import re
import time

import pandas as pd
import yaml

from agent import Agent
from config import ROOT, load_config
from utils import load_jsonl


def normalise(text):
    return (text or "").lower().replace("’", "'").replace("‘", "'")


def contains(answer, item):
    """item is a string, or a list of alternatives (any one may appear)."""
    options = item if isinstance(item, list) else [item]
    return any(normalise(o) in answer for o in options)


def args_match(expected, calls):
    def same(a, b):
        try:
            return float(a) == float(b)
        except (TypeError, ValueError):
            return str(a).lower() == str(b).lower()
    return any(all(k in c["args"] and same(c["args"][k], v) for k, v in expected.items()) for c in calls)


def check_turn(turn, r):
    """Each check is True (passed) or False (failed). Only the checks a turn asks for are run."""
    answer = normalise(r["answer"])
    calls = [c for c in r["tool_calls"] if c["tool"] == turn.get("expect_tool")]
    checks = {}
    if turn.get("expect_clarifying"):
        # Asks the user to choose ("?" or "let me know") AND offers at least 3 of the 4 priorities.
        # (Checking only the last line was too strict: v3.1 asks first, then lists the options.)
        asks = "?" in answer or "let me know" in answer
        offers = sum(w in answer for w in ["lowest cost", "cashback", "travel", "interest"]) >= 3
        checks["clarifying"] = asks and offers
    if "expect_rewrite" in turn:
        checks["rewrite"] = (r["standalone_question"] != r["question"]) == turn["expect_rewrite"]
    if "expect_tool" in turn:
        checks["tool"] = bool(calls)
    if "expect_args" in turn:
        checks["args"] = args_match(turn["expect_args"], calls)
    for item in turn.get("must_include", []):
        checks[f"includes {item}"] = contains(answer, item)
    for item in turn.get("must_not_include", []):
        checks[f"excludes {item}"] = not contains(answer, item)
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="agent_v3")
    parser.add_argument("--recheck", action="store_true",
                        help="re-apply the checks in conversations.yaml to saved answers, without calling Gemini")
    args = parser.parse_args()

    cfg = load_config()
    cfg["llm"].update(max_attempts=4, base_wait_s=15)  # patient retries: completeness beats speed here
    sleep_s = cfg["evaluation"]["sleep_s"]
    conversations = yaml.safe_load(open(ROOT / "eval" / "conversations.yaml", encoding="utf-8"))
    out_path = ROOT / "eval" / "results" / f"conversations_{args.tag}.jsonl"
    if args.recheck:
        recheck(conversations, out_path)
        summarise(out_path, args.tag)
        return
    done = {r["conversation"] for r in load_jsonl(out_path)} if out_path.exists() else set()
    todo = [c for c in conversations if c["id"] not in done]
    print(f"{len(done)} conversations already run, {len(todo)} to go")

    agent = Agent(cfg)
    for conv in todo:
        history, records = [], []
        try:
            for n, turn in enumerate(conv["turns"], start=1):
                r = agent.answer(turn["user"], history=history, profile=conv.get("profile") or {})
                checks = check_turn(turn, r)
                records.append({
                    "conversation": conv["id"], "turn": n, "user": turn["user"],
                    "standalone_question": r["standalone_question"], "answer": r["answer"],
                    "tool_calls": r["tool_calls"], "cited": r["cited"], "checks": checks,
                    "passed": all(checks.values()),
                    "llm_call_s": r["llm_call_s"], "total_s": round(r["timing"]["total_s"], 1),
                    "model": r["model"], "agent_version": r["agent_version"],
                })
                history.append({"question": turn["user"], "answer": r["answer"]})
                failed = [k for k, ok in checks.items() if not ok]
                print(f"{conv['id']} turn {n}: {'PASS' if not failed else 'FAIL ' + str(failed)} "
                      f"({r['timing']['total_s']:.1f}s)")
                time.sleep(sleep_s)
        except Exception as e:
            print(f"\nStopped in {conv['id']}: {e}\nRerun the same command later to resume.")
            break
        with open(out_path, "a", encoding="utf-8") as f:  # save whole conversations only
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summarise(out_path, args.tag)


def recheck(conversations, out_path):
    """Apply the current checks to saved answers. Use only to fix a check that was itself wrong."""
    turns = {(c["id"], n): t for c in conversations for n, t in enumerate(c["turns"], start=1)}
    rows = load_jsonl(out_path)
    for r in rows:
        saved = {"question": r["user"], "answer": r["answer"], "standalone_question": r["standalone_question"],
                 "tool_calls": r["tool_calls"],
                 "cited": r.get("cited", [int(n) for n in re.findall(r"\[(\d+)\]", r["answer"])])}
        r["checks"] = check_turn(turns[(r["conversation"], r["turn"])], saved)
        r["passed"] = all(r["checks"].values())
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Rechecked {len(rows)} saved turns.")


def summarise(out_path, tag):
    rows = load_jsonl(out_path) if out_path.exists() else []
    if not rows:
        return
    checks = [(name, ok) for r in rows for name, ok in r["checks"].items()]
    kinds = pd.DataFrame([{"check": name.split(" ")[0], "passed": ok} for name, ok in checks])
    summary = kinds.groupby("check")["passed"].agg(["sum", "count"]).rename(columns={"sum": "passed", "count": "total"})
    print(f"\nTurns passing every check: {sum(r['passed'] for r in rows)}/{len(rows)}")
    print(f"Individual checks passed: {sum(ok for _, ok in checks)}/{len(checks)}\n")
    print(summary.to_string())
    summary.to_csv(ROOT / "eval" / "results" / f"conversations_summary_{tag}.csv")
    print(f"\nMedian turn latency: {pd.Series([r['total_s'] for r in rows]).median():.1f}s")
    print("\nFailed turns:")
    for r in rows:
        if not r["passed"]:
            failed = [k for k, ok in r["checks"].items() if not ok]
            print(f"- {r['conversation']} turn {r['turn']} {failed}\n  Q: {r['user']}\n"
                  f"  Rewritten: {r['standalone_question']}\n  A: {r['answer'][:300]}...")


if __name__ == "__main__":
    main()
