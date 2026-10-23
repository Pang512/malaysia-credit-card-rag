"""Agentic assistant: Gemini decides when to search the PDS text and when to query the card catalog."""
import argparse
import os
import re
import time
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from catalog import CARD_FIELDS, Catalog
from config import load_config
from generator import RETRYABLE
from retriever import Retriever

AGENT_PROMPT = """You help people in Malaysia understand and choose credit cards, using only official
Product Disclosure Sheets (PDS). Your knowledge comes ONLY from two tools:

- query_catalog: a verified table of 51 cards from 8 banks (Maybank, Maybank Islamic, CIMB, OCBC, Public Bank,
  Bank Islam, Bank Muamalat, BSN) with annual fees, minimum income, late payment cap, overseas fee, profit or
  finance rates and stated benefits. Use it to list, filter, compare, rank or shortlist cards, and for a
  specific card's fees.
- search_documents: passages of PDS text. Use it for explanations, definitions (e.g. Shariah terms),
  conditions, worked examples and anything not in the catalog.
Call tools as needed, possibly both and possibly several times, before answering.

Rules:
1. Use ONLY facts returned by the tools. Never use outside knowledge, including rewards or perks you may know about.
2. Cite every fact with the ref number from the tool results, one number per bracket, e.g. [3] or [3][7].
   Never cite a range such as [1-26].
3. "not stated in PDS" means the document does not give that value. Say so; never guess.
4. Never filter, sort or count results yourself. Pass every condition the user actually states (income,
   annual fee, bank, card type, a card they name) as tool arguments. If the question has several parts
   (e.g. "which can I apply for, and which have no annual fee"), call the tool once per part.
   Never filter by words in card names (e.g. "Travel", "Cash") unless the user names that card.
5. If the tools return nothing relevant, reply exactly: "I don't know based on the available documents."
6. Answer what the results support, then state what is not covered, starting with "The documents do not state".
7. Always include the conditions attached to a figure: waiver rules, payment tiers, employer type, caps.
   An annual fee of 0 means waived or free; check annual_fee_note for conditions.
8. Never mix figures between banks or cards. Banks other than the 8 above (e.g. Hong Leong) are not covered.
9. Never infer eligibility from a card's name (e.g. Teachers, G-Card, UUM, Premier, Wealth). If a name suggests
   the card may be for a specific group, say the PDS does not state who is eligible and suggest checking with the bank.
10. For "which card is best for X" questions, compare the relevant facts across ALL cards: sort the catalog by
    the field that matters (e.g. for someone worried about paying late, sort_by late_payment_cap_rm), and use
    search_documents to look for any benefits the PDS states that relate to X. Give a short shortlist with
    trade-offs, mention that rewards, cashback rates, lounge access and miles are generally not in the PDS,
    and note this is general information, not financial advice.
11. Be concise. Use a short table when comparing three or more cards. Describe the data exactly
    (e.g. "credit cards", never "charge cards" unless the data says so)."""


class Agent:
    def __init__(self, cfg=None, model=None):
        load_dotenv()
        self.cfg = cfg or load_config()
        self.model = model or self.cfg["llm"]["model"]
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.retriever = Retriever(self.cfg["retrieval"]["strategy"], self.cfg)
        self.catalog = Catalog(self.cfg)
        self.max_rounds = self.cfg.get("agent", {}).get("max_tool_rounds", 4)
        self.sources, self.tool_log = [], []

    def _add_source(self, source):
        """Every item a tool returns gets a number, so the answer can cite it as [n]."""
        self.sources.append(source)
        return len(self.sources)

    # ----- Tools. Their docstrings and type hints are the tool descriptions Gemini reads. -----

    def search_documents(self, query: str, bank: Optional[str] = None) -> dict:
        """Search the text of the banks' Product Disclosure Sheets.
        Use for explanations, definitions, conditions and worked examples.

        Args:
            query: what to look for, in plain words, e.g. "late payment compensation Ta'widh".
            bank: optional bank name to search only that bank's documents, e.g. "Bank Islam".
        """
        k = self.cfg["retrieval"]["top_k"]
        results = self.retriever.search(query, k=k * 4 if bank else k)
        if bank:
            results = [r for r in results if bank.lower() in r["bank"].lower()][:k]
        passages = []
        for r in results:
            ref = self._add_source({"kind": "document", "bank": r["bank"], "title": r["product"],
                                    "doc_id": r["doc_id"], "pages": [r["page"]], "chunk_id": r["chunk_id"],
                                    "valid_as_at": r.get("valid_as_at")})
            passages.append({"ref": ref, "bank": r["bank"], "page": r["page"],
                             "valid_as_at": r.get("valid_as_at"), "text": r["text"]})
        return {"passages": passages} if passages else {"passages": [], "note": "No matching passages."}

    def query_catalog(self, bank: Optional[str] = None, card_type: Optional[str] = None,
                      annual_income_rm: Optional[float] = None, max_annual_fee_rm: Optional[float] = None,
                      name_contains: Optional[str] = None, sort_by: Optional[str] = None,
                      limit: Optional[int] = 15) -> dict:
        """Filter, sort and list cards from the verified catalog of 51 cards from 8 Malaysian banks.
        Use for listing cards, eligibility by income, comparing fees, rates or overseas charges, and shortlists.

        Args:
            bank: keep one bank, e.g. "CIMB", "Public Bank", "BSN".
            card_type: "conventional" or "islamic".
            annual_income_rm: the user's ANNUAL income in RM (monthly income x 12). Keeps cards whose stated
                minimum income is at or below it, plus cards whose PDS states no minimum.
            max_annual_fee_rm: keep cards whose principal annual fee is at or below this. 0 = waived or free.
            name_contains: text in the card name, e.g. "Platinum", "Quantum", "Travel".
            sort_by: sort ascending by one of: annual_fee_principal_rm, annual_fee_supplementary_rm,
                overseas_fee_pct, best_rate_pct, highest_rate_pct, late_payment_cap_rm, min_annual_income_rm.
            limit: maximum number of cards to return (up to 60). Use 60 to list everything.
        """
        total, rows = self.catalog.query(bank, card_type, annual_income_rm, max_annual_fee_rm,
                                         name_contains, sort_by, limit)
        cards = []
        for idx, row in rows.iterrows():
            pages = [int(p) for p in str(row["source_pages"]).split(";") if p.strip().isdigit()]
            ref = self._add_source({"kind": "catalog", "bank": row["bank"], "title": row["card_name"],
                                    "doc_id": row["doc_id"], "pages": pages, "catalog_row": int(idx),
                                    "valid_as_at": row["valid_as_at"]})
            card = {"ref": ref}
            for field in CARD_FIELDS:
                value = row[field]
                card[field] = "not stated in PDS" if pd.isna(value) else (
                    value.item() if hasattr(value, "item") else value)
            cards.append(card)
        return {"total_matches": total, "returned": len(cards), "cards": cards}

    # ----- The agent loop -----

    def _generate(self, contents, config):
        llm = self.cfg["llm"]
        for attempt in range(llm["max_attempts"]):
            try:
                return self.client.models.generate_content(model=self.model, contents=contents, config=config)
            except errors.APIError as e:
                if e.code not in RETRYABLE or attempt == llm["max_attempts"] - 1:
                    raise
                time.sleep(llm["base_wait_s"] * 2 ** attempt)

    def answer(self, question):
        self.sources, self.tool_log = [], []
        tools = {"search_documents": self.search_documents, "query_catalog": self.query_catalog}
        config = types.GenerateContentConfig(
            system_instruction=AGENT_PROMPT,
            temperature=self.cfg["llm"]["temperature"],
            tools=list(tools.values()),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),  # we run the loop
        )
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
        t0, llm_calls = time.perf_counter(), 0

        for round_no in range(self.max_rounds + 1):
            if round_no == self.max_rounds:  # last round: tools switched off, so it must answer now
                config = config.model_copy(update={"tool_config": types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="NONE"))})
            response = self._generate(contents, config)
            llm_calls += 1
            calls = response.function_calls or []
            if not calls:
                break
            contents.append(response.candidates[0].content)  # the model's tool request
            results = []
            for call in calls:
                args = dict(call.args or {})
                self.tool_log.append({"tool": call.name, "args": args})
                try:
                    result = tools[call.name](**args)
                except Exception as e:  # a bad argument shouldn't crash the loop; tell the model instead
                    result = {"error": str(e)}
                results.append(types.Part.from_function_response(name=call.name, response={"result": result}))
            contents.append(types.Content(role="user", parts=results))  # our tool results

        answer = (response.text or "").strip()
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 1 <= int(n) <= len(self.sources)})
        return {"question": question, "answer": answer, "sources": self.sources, "cited": cited,
                "tool_calls": self.tool_log, "llm_calls": llm_calls, "model": self.model,
                "timing": {"total_s": time.perf_counter() - t0}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask the agentic credit card assistant a question.")
    parser.add_argument("question", nargs="*")
    parser.add_argument("--model", help="Use a different LLM than config.yaml")
    args = parser.parse_args()
    question = " ".join(args.question) or "Which Islamic cards have no annual fee?"

    result = Agent(model=args.model).answer(question)
    print(f"\nQ: {result['question']}\n")
    print("Tool calls:")
    for t in result["tool_calls"]:
        print(f"  {t['tool']}({t['args']})")
    print(f"\nA: {result['answer']}\n")
    print("Cited sources:")
    for n in result["cited"]:
        s = result["sources"][n - 1]
        print(f"  [{n}] {s['kind']:8} | {s['bank']} | {s['title']} | p{','.join(map(str, s['pages']))}")
    print(f"\n{result['llm_calls']} LLM calls | {len(result['sources'])} sources returned | "
          f"{result['timing']['total_s']:.1f}s total")