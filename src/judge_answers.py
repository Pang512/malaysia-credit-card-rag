"""Score saved answers: refusal behaviour and citation accuracy (rule-based),
plus correctness and faithfulness (LLM-as-judge). Resumable like run_answers.py."""
import argparse
import json
import os
import time

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from config import ROOT, load_config
from generator import RETRYABLE
from utils import load_jsonl

IDK_MARKER = "i don't know"

JUDGE_PROMPT = """You are grading an answer from a Malaysian credit card assistant.

Question: {question}

Reference answer (verified by a human): {reference}

Assistant answer: {answer}

Context passages the assistant was given:
{context}

Grade two things.
1. correctness, comparing the assistant answer with the reference answer:
   - "correct": all key facts (amounts, rates, conditions) match the reference.
   - "partial": some key facts match, but something important is missing or wrong.
   - "incorrect": the key facts are wrong or missing.
   If the reference starts with "RUBRIC", grade against its Must / Must not / Should points instead.
2. faithful: true if every factual claim in the assistant answer is supported by the
   context passages, false if any claim is not supported by them.

Reply with JSON only, in this format:
{{"correctness": "correct", "faithful": true, "reason": "one short sentence"}}"""


def parse_sources(text):
    if not text:
        return set()
    return {item.strip() for item in text.split(";") if item.strip()}

class DailyQuotaExceeded(Exception):
    pass

def call_judge(client, model, prompt):
    for attempt in range(4):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0,
                                                   response_mime_type="application/json"),
            )
            return json.loads(response.text)
        except errors.APIError as e:
            if e.code == 429 and "PerDay" in str(e):
                raise DailyQuotaExceeded(f"Daily free-tier quota used up for {model}.") from e
            if e.code not in RETRYABLE or attempt == 3:
                raise
            wait = 15 * 2 ** attempt
            print(f"Judge returned {e.code}, retrying in {wait}s...")
            time.sleep(wait)
        except json.JSONDecodeError:
            return {"correctness": "error", "faithful": None, "reason": "judge returned invalid JSON"}


def citation_ok(cited, sources, expected):
    """True if any cited passage comes from an expected page. None if not applicable."""
    if not expected or not cited:
        return None
    return any(sources[i - 1] in expected for i in cited)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="baseline")
    args = parser.parse_args()

    load_dotenv()
    cfg = load_config()
    results_dir = ROOT / "eval" / "results"
    answers = load_jsonl(results_dir / f"answers_{args.tag}.jsonl")
    strategy = answers[0]["strategy"]
    chunks = {c["chunk_id"]: c for c in
              load_jsonl(ROOT / cfg["paths"]["index_dir"] / f"{strategy}_chunks.jsonl")}
    test = pd.read_csv(ROOT / "eval" / "test_set.csv", keep_default_na=False).set_index("id")

    judged_path = results_dir / f"judgements_{args.tag}.jsonl"
    done = {r["id"] for r in load_jsonl(judged_path)} if judged_path.exists() else set()
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    judge_model = cfg["evaluation"]["judge_model"]

    with open(judged_path, "a", encoding="utf-8") as f:
        for a in answers:
            if a["id"] in done:
                continue
            q = test.loc[a["id"]]
            refused = IDK_MARKER in a["answer"].lower().replace("\u2019", "'")
            if refused:
                # Rule-based: a refusal is right only when the question is unanswerable
                verdict = {"correctness": "correct" if q["answerable"] == "no" else "incorrect",
                           "faithful": True, "reason": "refusal (rule-based check)"}
            else:
                context = "\n\n".join(f"[{i}] {chunks[cid]['text']}"
                                      for i, cid in enumerate(a["chunk_ids"], start=1))
                prompt = JUDGE_PROMPT.format(question=a["question"], reference=q["reference_answer"],
                                             answer=a["answer"], context=context)
                try:
                    verdict = call_judge(client, judge_model, prompt)
                except DailyQuotaExceeded as e:
                    print(f"\n{e} Rerun later to resume, or change judge_model in config.yaml.")
                    return
                time.sleep(cfg["evaluation"]["sleep_s"])
                record = {"id": a["id"], "refused": refused,
                      "judge_model": "rule" if refused else judge_model, **verdict}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{a['id']}: {record['correctness']} | faithful={record['faithful']}")

    # ---------- Summary ----------
    judged = pd.DataFrame(load_jsonl(judged_path)).set_index("id")
    ans = pd.DataFrame(answers).set_index("id")
    df = test.join(judged, how="inner").join(ans[["answer", "cited", "sources", "timing", "model"]])
    df["latency_s"] = df["timing"].apply(lambda t: t["total_s"])
    df["citation_ok"] = [citation_ok(r.cited, r.sources, parse_sources(r.expected_sources))
                         for r in df.itertuples()]
    df["is_correct"] = df["correctness"].eq("correct")
    df["is_correct_or_partial"] = df["correctness"].isin(["correct", "partial"])
    df["faithful_num"] = pd.to_numeric(df["faithful"].map({True: 1, False: 0}), errors="coerce")

    answerable = df[df["answerable"] != "no"]
    unanswerable = df[df["answerable"] == "no"]
    summary = pd.Series({
        "questions": len(df),
        "accuracy (correct)": answerable["is_correct"].mean(),
        "accuracy (correct or partial)": answerable["is_correct_or_partial"].mean(),
        "faithfulness": df["faithful_num"].mean(),
        "false refusal rate (answerable)": answerable["refused"].mean(),
        "correct refusal rate (unanswerable)": unanswerable["refused"].mean(),
        "citation accuracy": pd.to_numeric(df["citation_ok"], errors="coerce").mean(),
        "median latency (s)": df["latency_s"].median(),
        "judge errors": (df["correctness"] == "error").sum(),
    }).round(3)
    by_category = (df.groupby("category")
                   .agg(n=("is_correct", "size"), correct=("is_correct", "mean"),
                        correct_or_partial=("is_correct_or_partial", "mean"),
                        faithful=("faithful_num", "mean"))
                   .round(2))

    summary.to_csv(results_dir / f"answer_summary_{args.tag}.csv", header=["value"])
    by_category.to_csv(results_dir / f"answer_by_category_{args.tag}.csv")
    review = df.reset_index()[["id", "category", "question", "reference_answer", "answer",
                               "correctness", "faithful", "reason"]]
    review["human_agrees"] = ""  # fill in during the manual spot check
    review.to_csv(results_dir / f"answer_review_{args.tag}.csv", index=False, encoding="utf-8-sig")

    print("\n=== Summary ===")
    print(summary.to_string())
    print("\n=== By category ===")
    print(by_category.to_string())
    wrong = df[~df["is_correct"]]
    print(f"\n=== {len(wrong)} not fully correct ===")
    for i, r in wrong.iterrows():
        print(f"  {i} ({r['category']}): {r['correctness']} | {r['reason']}")


if __name__ == "__main__":
    main()