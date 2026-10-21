"""Compare evaluation runs side by side: summary metrics and per-question changes."""
import argparse

import pandas as pd

from config import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tags", nargs="+", help="run tags, oldest first, e.g. baseline prompt_v2")
    args = parser.parse_args()
    res = ROOT / "eval" / "results"

    summary = pd.concat({t: pd.read_csv(res / f"answer_summary_{t}.csv", index_col=0)["value"]
                         for t in args.tags}, axis=1)
    print("=== Summary ===")
    print(summary.to_string())

    base, new = args.tags[0], args.tags[-1]
    a = pd.read_csv(res / f"answer_review_{base}.csv", encoding="utf-8-sig").set_index("id")
    b = pd.read_csv(res / f"answer_review_{new}.csv", encoding="utf-8-sig").set_index("id")
    changed = a.index[a["correctness"].ne(b["correctness"])]
    print(f"\n=== {len(changed)} questions changed ({base} -> {new}) ===")
    for qid in changed:
        print(f"  {qid} ({a.loc[qid, 'category']}): {a.loc[qid, 'correctness']} -> {b.loc[qid, 'correctness']}"
              f" | faithful {a.loc[qid, 'faithful']} -> {b.loc[qid, 'faithful']}")


if __name__ == "__main__":
    main()