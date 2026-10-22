"""Print extracted page text, to inspect extraction by eye.
Usage: python src/show_pages.py maybank_islamic_nadi:2 bsn_al_aiman:1 [--plain]"""
import json
import sys

from config import ROOT

args = [a for a in sys.argv[1:] if not a.startswith("--")]
plain = "--plain" in sys.argv
wanted = {(doc, int(page)) for doc, page in (a.rsplit(":", 1) for a in args)}

path = ROOT / "data" / "processed" / ("pages.jsonl" if plain else "pages_tables.jsonl")
with open(path, encoding="utf-8") as f:
    for line in f:
        p = json.loads(line)
        if (p["doc_id"], p["page"]) in wanted:
            print(f"\n===== {p['doc_id']} p{p['page']} ({'plain' if plain else 'table-aware'}) =====")
            print(p["text"])