"""Draft a structured card catalog from the PDS documents using an LLM (structured extraction).
The draft MUST be verified by a human before use. Resumable: one cached JSON file per document."""
import json
import os
import time

import pandas as pd
import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from catalog_schema import Card
from config import ROOT, load_config
from generator import RETRYABLE
from utils import load_jsonl

PROMPT = """You are extracting credit card facts from a Malaysian bank's Product Disclosure Sheet (PDS).

Bank: {bank}
Document: {product}

Return one entry per individual credit card named in the document (for example, each row of an
annual fee table). If the document describes only one card, return one entry.

Rules:
- Use ONLY the document text below. Never use outside knowledge.
- If a field is not stated in the document, use null. Never guess or estimate.
- Facts that apply to every card in the document (e.g. late payment charge, overseas fee,
  finance or profit rates) must be copied to every card.
- Annual fees are numbers in RM. Use 0 if waived or free for life, and put any conditions in annual_fee_note.
- Percentages are numbers, e.g. 1.25 for 1.25%.
- Text fields: copy the document's wording, shortened if needed.
- source_pages: the page numbers where this card's facts appear. Pages are marked "=== Page N ===".

Document text:
{text}"""


def document_text(pages, doc_id):
    return "\n\n".join(f"=== Page {p['page']} ===\n{p['text']}" for p in pages
                       if p["doc_id"] == doc_id and p["language"] == "en")


def extract(client, model, prompt):
    for attempt in range(4):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json",
                                                   response_schema=list[Card]),
            )
            if response.parsed is not None:
                return [card.model_dump() for card in response.parsed]
            # Fallback: validate the raw JSON ourselves against the same schema
            return [Card.model_validate(card).model_dump() for card in json.loads(response.text)]
        except errors.APIError as e:
            if e.code not in RETRYABLE or attempt == 3:
                raise
            wait = 15 * 2 ** attempt
            print(f"{model} returned {e.code}, retrying in {wait}s...")
            time.sleep(wait)


def main():
    load_dotenv()
    cfg = load_config()
    out_dir = ROOT / "data" / "catalog"
    cache_dir = out_dir / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)
    docs = yaml.safe_load((ROOT / cfg["paths"]["sources"]).read_text(encoding="utf-8"))["documents"]
    pages = load_jsonl(ROOT / cfg["paths"]["processed_dir"] / "pages_tables.jsonl")
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    model = cfg["catalog"]["extraction_model"]

    rows = []
    for doc in docs:
        text = document_text(pages, doc["doc_id"])
        if not text:
            print(f"[skip]   {doc['doc_id']}: no pages")
            continue
        cache = cache_dir / f"{doc['doc_id']}.json"
        if cache.exists():
            cards = json.loads(cache.read_text(encoding="utf-8"))
            print(f"[cached] {doc['doc_id']}: {len(cards)} cards")
        else:
            prompt = PROMPT.format(bank=doc["bank"], product=doc["product"], text=text)
            cards = extract(client, model, prompt)
            cache.write_text(json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[ok]     {doc['doc_id']}: {len(cards)} cards")
            time.sleep(cfg["evaluation"]["sleep_s"])

        for card in cards:
            rows.append({
                "doc_id": doc["doc_id"], "bank": doc["bank"], "card_type": doc["card_type"],
                "shariah_concept": doc.get("shariah_concept"), "valid_as_at": doc.get("valid_as_at"),
                **card,
                "source_pages": ";".join(str(p) for p in card.get("source_pages") or []),
                "verified": "", "verifier_notes": "",
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "catalog_draft.csv", index=False, encoding="utf-8-sig")
    print(f"\n{len(df)} cards drafted. Cards per bank:")
    print(df["bank"].value_counts().to_string())
    print("\nMissing values per field (null = not stated in the PDS, or missed):")
    print(df.isna().sum()[lambda s: s > 0].to_string())


if __name__ == "__main__":
    main()