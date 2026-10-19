"""Answer questions using only the retrieved chunks, with numbered citations."""
import argparse
import os
import re
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from config import load_config
from retriever import Retriever

SYSTEM_PROMPT = """You answer questions about Malaysian credit card Product Disclosure Sheets (PDS).

Rules:
1. Use ONLY the numbered context passages. Never use outside knowledge.
2. Cite every fact with its passage number in square brackets, e.g. [1] or [2][3].
3. If the passages do not contain the answer, reply exactly:
   "I don't know based on the available documents."
4. If the question is about a specific bank or card, never use figures from a different bank or card.
5. If the passages only partly answer the question, answer the supported part and say what is missing.
6. For Islamic cards (Credit Card-i), use the document's own terms (e.g. profit rate, Ta'widh, Ibra').
7. Be concise. Quote RM amounts and percentages exactly as written."""

IDK = "I don't know based on the available documents."

# 429 = rate limited, 500/503 = Google's servers having trouble. All temporary, so worth retrying.
RETRYABLE = {429, 500, 503}


def format_context(chunks):
    blocks = []
    for i, c in enumerate(chunks, start=1):
        header = (f"[{i}] Source: {c['bank']} | {c['product']} | page {c['page']} | "
                  f"valid as at {c.get('valid_as_at') or 'unknown'}")
        blocks.append(f"{header}\n{c['text']}")
    return "\n\n".join(blocks)


class RAGPipeline:
    def __init__(self, strategy=None, cfg=None, model=None, use_fallback=True):
        load_dotenv()
        self.cfg = cfg or load_config()
        self.strategy = strategy or self.cfg["retrieval"]["strategy"]
        self.retriever = Retriever(self.strategy, self.cfg)
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = model or self.cfg["llm"]["model"]
        fallback = self.cfg["llm"].get("fallback_model") if use_fallback else None
        self.fallback = fallback if fallback != self.model else None

    def _call(self, model, prompt):
        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=self.cfg["llm"]["temperature"],
            ),
        )
        return (response.text or "").strip()

    def generate(self, question, chunks):
        """Returns (answer, model_used, seconds_spent_waiting_on_retries)."""
        prompt = f"Context passages:\n\n{format_context(chunks)}\n\nQuestion: {question}"
        llm = self.cfg["llm"]
        models = [self.model] + ([self.fallback] if self.fallback else [])
        waited = 0.0

        for model in models:
            for attempt in range(llm["max_attempts"]):
                try:
                    return self._call(model, prompt), model, waited
                except errors.APIError as e:
                    if e.code not in RETRYABLE:
                        raise  # not temporary (e.g. wrong API key): fail immediately
                    if attempt < llm["max_attempts"] - 1:
                        wait = llm["base_wait_s"] * 2 ** attempt  # exponential backoff
                        print(f"{model} returned {e.code}, retrying in {wait}s...")
                        time.sleep(wait)
                        waited += wait
            print(f"{model} is still unavailable, giving up on it.")

        raise RuntimeError("Gemini is unavailable right now. Please try again in a few minutes.")

    def answer(self, question):
        t0 = time.perf_counter()
        chunks = self.retriever.search(question)
        t1 = time.perf_counter()
        answer, model_used, waited = self.generate(question, chunks)
        t2 = time.perf_counter()

        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)
                        if 1 <= int(n) <= len(chunks)})
        timing = {
            "retrieval_s": t1 - t0,
            "llm_s": (t2 - t1) - waited,  # includes any failed attempts
            "retry_wait_s": waited,
            "total_s": t2 - t0,
        }
        return {"question": question, "answer": answer, "sources": chunks,
                "cited": cited, "model": model_used, "timing": timing}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask the credit card RAG assistant a question.")
    parser.add_argument("question", nargs="*", help="Your question")
    parser.add_argument("--model", help="Use a different LLM than config.yaml")
    parser.add_argument("--no-fallback", action="store_true", help="Disable the fallback model")
    args = parser.parse_args()
    question = " ".join(args.question) or "What is the late payment charge for Bank Islam credit card-i?"

    t_start = time.perf_counter()
    rag = RAGPipeline(model=args.model, use_fallback=not args.no_fallback)
    startup = time.perf_counter() - t_start

    result = rag.answer(question)
    t = result["timing"]

    print(f"\nQ: {result['question']}")
    print(f"Model: {result['model']}\n")
    print(f"A: {result['answer']}\n")
    print("Retrieved sources:")
    for i, c in enumerate(result["sources"], start=1):
        mark = "cited" if i in result["cited"] else "     "
        print(f"  [{i}] {mark} | {c['bank']} | page {c['page']} | score {c['score']:.3f}")
    print(f"\nTiming: startup {startup:.1f}s | retrieval {t['retrieval_s']:.2f}s | "
          f"LLM {t['llm_s']:.1f}s | retry waits {t['retry_wait_s']:.0f}s | "
          f"answer total {t['total_s']:.1f}s")