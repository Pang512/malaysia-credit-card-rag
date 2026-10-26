"""Agentic assistant: Gemini decides when to search the PDS text, query the card catalog or rank cards."""
import argparse
import copy
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

AGENT_VERSION = "agent_v3"
TEXT_FIELDS = ["overseas_fee_text", "late_payment_text", "cash_advance_fee_text"]
HISTORY_TURNS = 3  # earlier question-answer pairs passed to the model as conversation context

# How recommend_cards ranks for each priority: catalog columns, sorted ascending, "not stated" always last
PRIORITIES = {
    "lowest_cost": ["annual_fee_principal_rm", "highest_rate_pct", "overseas_fee_pct"],
    "cashback": ["annual_fee_principal_rm", "highest_rate_pct"],
    "travel": ["overseas_fee_pct", "annual_fee_principal_rm"],
    "low_interest": ["highest_rate_pct", "best_rate_pct", "annual_fee_principal_rm"],
}
PRIORITY_NOTES = {
    "cashback": "Cashback rates and categories are not in the PDS, so these cards are ranked by cost only "
                "(annual fee, then the highest finance or profit rate).",
    "travel": "Lounge access and miles are generally not in the PDS, so these cards are ranked by overseas "
              "transaction fee, then annual fee. Check benefits_text for any stated travel benefits.",
    "low_interest": "Ranked by the highest finance or profit rate (what you pay with a poor payment record), "
                    "then the best rate. Paying in full every month avoids these charges.",
}
LABELS = {"annual_fee_principal_rm": "annual fee (RM)", "highest_rate_pct": "highest rate (% p.a.)",
          "best_rate_pct": "best rate (% p.a.)", "overseas_fee_pct": "overseas fee (%)"}

AGENT_PROMPT = """You help people in Malaysia, often first-time cardholders, understand and choose credit cards
using official Product Disclosure Sheets (PDS). Your knowledge comes ONLY from three tools:

- query_catalog: a verified table of 51 cards from 8 banks (Maybank, Maybank Islamic, CIMB, OCBC, Public Bank,
  Bank Islam, Bank Muamalat, BSN) with annual fees and their conditions, minimum income, late payment cap,
  overseas fee, profit or finance rates and stated benefits. Use it to list, filter, compare or count cards,
  and for any fee question that covers several cards.
- recommend_cards: ranks eligible cards for one priority (lowest_cost, cashback, travel, low_interest) and
  explains the ranking. Use it for "which card should I get" or "which is best for me" questions.
- search_documents: passages of PDS text. Use it for explanations, definitions (e.g. Shariah terms),
  conditions, worked examples and anything not in the catalog.
Call tools as needed, possibly several, before answering.

Conversation:
A. Earlier turns are shown only so you understand references like "those two" or "someone like me".
   Get every figure again from this turn's tool results; never copy figures or citations from earlier answers.
B. Keep using what the user has told you (income, conventional or Islamic, priority) in later answers,
   e.g. pass card_type and annual_income_rm (monthly income x 12) as tool arguments.
C. For a recommendation, if the user's main priority is unknown, do not call tools. Ask ONE short question
   offering the options: lowest cost, cashback and everyday spending, travel, or low interest (if they may not
   always pay in full). Also ask for their monthly income if it is unknown.

Rules:
1. Use ONLY facts returned by the tools. Never use outside knowledge, including rewards or perks you may know about.
2. Cite facts with the ref numbers from the tool results, one number per bracket, e.g. [3] or [3][7]. Never cite
   a range such as [1-26], and never more than 3 refs in one sentence: for a statement about many cards, cite
   up to 3 examples.
3. "not stated in PDS" means the value is unknown. Never treat it as zero or free, and never rank an unknown
   value as the lowest. Say the PDS does not state it.
4. Never filter, sort or count results yourself. Pass every condition the user states (income, annual fee,
   bank, card type, a card they name) as tool arguments; for a multi-part question, call the tool once per part.
   Never filter by words in card names (e.g. "Travel", "Cash") unless the user names that card.
5. Never say "only", "all" or "none" about cards unless a query_catalog result covering every relevant card
   supports it.
6. If the tools return nothing relevant, reply exactly: "I don't know based on the available documents."
7. Mention what the documents do not cover only if the user asked about it, in one sentence starting
   "The documents do not state". Before saying so, call search_documents with the user's own wording.
8. Copy each figure with its condition exactly as written (waiver rules, payment tiers, employer type, caps).
   Never merge tiers or conditions. An annual fee of 0 means waived or free; check annual_fee_note.
9. Never mix figures between banks or cards. Banks other than the 8 above (e.g. Hong Leong) are not covered.
10. Never infer eligibility from a card's name (e.g. Teachers, G-Card, UUM, Premier). If a name suggests a
    specific group, say the PDS does not state who is eligible. If minimum income is not stated, still list
    the cards and say so once.
11. Recommendations: present the recommend_cards shortlist in its order. For each card give 2 or 3 reasons tied
    to the user's priority, with figures and citations, plus one trade-off. Then say in one line why the
    runner-up ranked lower. Repeat any note the tool returns (e.g. that cashback rates are not in the PDS).
    End with: "This is general information, not financial advice."
12. For general questions (how something works, typical charges), give up to 3 examples from different banks,
    matching the user's card type if known.
13. Write for a beginner: plain words, explain jargon briefly, be concise. Use a short table for three or more
    cards. Describe the data exactly (e.g. "credit cards", never "charge cards")."""


def strip_refs(text):
    """Remove [n] citations from an earlier answer: they point to that turn's sources, not this turn's."""
    return re.sub(r"\[\d+\]", "", text or "")


def describe_profile(profile):
    """Turn the app's 'About you' settings into one plain sentence for the model."""
    if not profile:
        return ""
    parts = []
    if profile.get("monthly_income_rm"):
        income = profile["monthly_income_rm"]
        parts.append(f"my monthly income is RM{income:,.0f} (RM{income * 12:,.0f} a year)")
    if profile.get("card_type"):
        parts.append(f"I want {profile['card_type']} cards")
    if profile.get("priority"):
        parts.append(f"my main priority is {profile['priority'].replace('_', ' ')}")
    return "; ".join(parts)


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

    def _card(self, idx, row, full_text=False):
        """One catalog row as a numbered source the model can cite."""
        pages = [int(p) for p in str(row["source_pages"]).split(";") if p.strip().isdigit()]
        ref = self._add_source({"kind": "catalog", "bank": row["bank"], "title": row["card_name"],
                                "doc_id": row["doc_id"], "pages": pages, "catalog_row": int(idx),
                                "valid_as_at": row["valid_as_at"]})
        card = {"ref": ref}
        # Full fee wording only for small result sets (comparisons); keep big lists compact and fast
        for field in CARD_FIELDS + (TEXT_FIELDS if full_text else []):
            value = row[field]
            card[field] = "not stated in PDS" if pd.isna(value) else (
                value.item() if hasattr(value, "item") else value)
        return card

    # ----- Tools. Their docstrings and type hints are the tool descriptions Gemini reads. -----

    def search_documents(self, query: str, bank: Optional[str] = None) -> dict:
        """Search the text of the banks' Product Disclosure Sheets.
        Use for explanations, definitions, conditions and worked examples.

        Args:
            query: what to look for, in plain words, e.g. "late payment compensation Ta'widh".
            bank: optional bank name to search only that bank's documents, e.g. "Bank Islam".
        """
        k = self.cfg["retrieval"]["top_k"]
        # Pre-filter: with a bank given, rank ALL chunks, then keep that bank's best k
        results = self.retriever.search(query, k=len(self.retriever.chunks) if bank else k)
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
        Use for listing cards, eligibility by income, comparing fees, rates or overseas charges, and for fee
        questions across cards (annual_fee_note gives each card's waiver conditions).

        Args:
            bank: keep one bank, e.g. "CIMB", "Public Bank", "BSN".
            card_type: "conventional" or "islamic".
            annual_income_rm: the user's ANNUAL income in RM (monthly income x 12). Keeps cards whose stated
                minimum income is at or below it, plus cards whose PDS states no minimum.
            max_annual_fee_rm: keep cards whose principal annual fee is at or below this. 0 = waived or free.
            name_contains: text in the card name, only when the user names a card, e.g. "Quantum".
            sort_by: sort ascending by one of: annual_fee_principal_rm, annual_fee_supplementary_rm,
                overseas_fee_pct, best_rate_pct, highest_rate_pct, late_payment_cap_rm, min_annual_income_rm.
            limit: maximum number of cards to return (up to 60). Use 60 to list everything.
        """
        total, rows = self.catalog.query(bank, card_type, annual_income_rm, max_annual_fee_rm,
                                         name_contains, sort_by, limit)
        cards = [self._card(idx, row, full_text=len(rows) <= 10) for idx, row in rows.iterrows()]
        return {"total_matches": total, "returned": len(cards), "cards": cards}

    def recommend_cards(self, priority: str, annual_income_rm: Optional[float] = None,
                        card_type: Optional[str] = None, top_n: Optional[int] = 3) -> dict:
        """Rank the cards a user is eligible for by one priority and return a shortlist with the reasons.
        Use for "which card should I get" or "which is best for me" questions once the priority is known.
        The ranking is done in code from verified catalog figures; explain it, do not re-rank it.

        Args:
            priority: one of "lowest_cost" (low annual fee and charges), "cashback" (everyday spending),
                "travel" (spending overseas), "low_interest" (may not always pay the full balance).
            annual_income_rm: the user's ANNUAL income in RM (monthly income x 12), if known.
            card_type: "conventional" or "islamic", if the user said which.
            top_n: how many cards to shortlist (default 3, up to 5).
        """
        if priority not in PRIORITIES:
            return {"error": f"priority must be one of {list(PRIORITIES)}"}
        keys = PRIORITIES[priority]
        total, eligible = self.catalog.query(card_type=card_type, annual_income_rm=annual_income_rm, limit=60)
        if eligible.empty:
            return {"total_eligible": 0, "shortlist": [], "note": "No cards match these conditions."}
        ranked = eligible.sort_values(keys, na_position="last")
        top_n = max(1, min(int(top_n or 3), 5))

        def ranked_on(row):
            return {LABELS[k]: ("not stated in PDS" if pd.isna(row[k]) else
                                (row[k].item() if hasattr(row[k], "item") else row[k])) for k in keys}

        shortlist = []
        for position, (idx, row) in enumerate(ranked.head(top_n).iterrows(), start=1):
            card = self._card(idx, row, full_text=True)
            card.update(position=position, ranked_on=ranked_on(row))
            shortlist.append(card)
        result = {
            "priority": priority,
            "ranking_rule": "Sorted lowest first by " + ", then ".join(LABELS[k] for k in keys)
                            + ". Values not stated in the PDS rank last.",
            "total_eligible": total,
            "eligible_with_no_stated_min_income": int(eligible["min_annual_income_rm"].isna().sum()),
            "shortlist": shortlist,
        }
        top_values = ranked.iloc[0][keys].fillna(-1)
        tied = int((ranked[keys].fillna(-1) == top_values).all(axis=1).sum())
        if tied > 1:
            result["tied_with_first"] = f"{tied} cards share the first card's ranking values; the order among them is arbitrary."
        if len(ranked) > top_n:
            idx, row = next(ranked.iloc[top_n:].iterrows())
            runner_up = self._card(idx, row)
            runner_up["ranked_on"] = ranked_on(row)
            result["runner_up"] = runner_up
        if priority in PRIORITY_NOTES:
            result["note"] = PRIORITY_NOTES[priority]
        return result

    # ----- The agent loop -----

    def _generate(self, contents, config):
        llm = self.cfg["llm"]
        for attempt in range(llm["max_attempts"]):
            try:
                return self.client.models.generate_content(model=self.model, contents=contents, config=config)
            except errors.APIError as e:
                if e.code not in RETRYABLE or attempt == llm["max_attempts"] - 1:
                    raise
                wait = llm["base_wait_s"] * 2 ** attempt
                print(f"{self.model} returned {e.code}, retrying in {wait}s...")
                time.sleep(wait)

    def answer(self, question, history=None, profile=None, on_step=None):
        """history: earlier turns as [{"question": ..., "answer": ...}]; profile: the app's 'About you' settings."""
        # A per-question copy: shares the loaded models, but has its own sources, so app users never mix
        run = copy.copy(self)
        run.sources, run.tool_log = [], []
        tools = {name: getattr(run, name) for name in ("search_documents", "query_catalog", "recommend_cards")}
        config = types.GenerateContentConfig(
            system_instruction=AGENT_PROMPT,
            temperature=self.cfg["llm"]["temperature"],
            tools=list(tools.values()),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),  # we run the loop
        )
        contents = []
        for turn in (history or [])[-HISTORY_TURNS:]:
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=turn["question"])]))
            contents.append(types.Content(role="model", parts=[types.Part.from_text(text=strip_refs(turn["answer"]))]))
        about = describe_profile(profile)
        text = f"{question}\n\n(About me: {about})" if about else question
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=text)]))
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
                run.tool_log.append({"tool": call.name, "args": args})
                if on_step:
                    on_step(call.name, args)  # lets the app show live progress
                try:
                    result = tools[call.name](**args)
                except Exception as e:  # a bad argument shouldn't crash the loop; tell the model instead
                    result = {"error": str(e)}
                results.append(types.Part.from_function_response(name=call.name, response={"result": result}))
            contents.append(types.Content(role="user", parts=results))  # our tool results

        answer = (response.text or "").strip()
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 1 <= int(n) <= len(run.sources)})
        return {"question": question, "answer": answer, "sources": run.sources, "cited": cited,
                "tool_calls": run.tool_log, "llm_calls": llm_calls, "model": self.model,
                "agent_version": AGENT_VERSION, "profile": profile or {}, "history_turns": len(history or []),
                "timing": {"total_s": time.perf_counter() - t0}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask the agentic credit card assistant a question.")
    parser.add_argument("question", nargs="*")
    parser.add_argument("--model", help="Use a different LLM than config.yaml")
    parser.add_argument("--income", type=float, help="monthly income in RM, as set in the app's 'About you'")
    parser.add_argument("--type", choices=["conventional", "islamic"], help="preferred card type")
    parser.add_argument("--priority", choices=list(PRIORITIES), help="main priority")
    args = parser.parse_args()
    question = " ".join(args.question) or "Which Islamic cards have no annual fee?"
    profile = {"monthly_income_rm": args.income, "card_type": args.type, "priority": args.priority}

    result = Agent(model=args.model).answer(question, profile=profile)
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
