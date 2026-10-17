"""Download the PDS files listed in data/sources.yaml into data/raw/."""
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data" / "sources.yaml"
RAW_DIR = ROOT / "data" / "raw"
# Some bank websites reject requests that don't look like a browser
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    docs = yaml.safe_load(SOURCES.read_text(encoding="utf-8"))["documents"]
    ready, needs_attention = 0, []

    for doc in docs:
        out = RAW_DIR / f"{doc['doc_id']}.pdf"

        if out.exists():
            print(f"[skip]   {out.name} already downloaded")
            ready += 1
            continue

        if not doc.get("url"):
            print(f"[manual] {doc['doc_id']}: download from {doc['source_page']}")
            print(f"         and save it as data/raw/{out.name}")
            needs_attention.append(doc["doc_id"])
            continue

        try:
            r = requests.get(doc["url"], headers=HEADERS, timeout=30)
            r.raise_for_status()
            if not r.content.startswith(b"%PDF"):
                raise ValueError("the link did not return a PDF")
            out.write_bytes(r.content)
            print(f"[ok]     {out.name} ({len(r.content) // 1024} KB)")
            ready += 1
        except Exception as e:
            print(f"[fail]   {doc['doc_id']}: {e}")
            needs_attention.append(doc["doc_id"])

    print(f"\n{ready}/{len(docs)} documents ready.")
    if needs_attention:
        print("Needs attention:", ", ".join(needs_attention))


if __name__ == "__main__":
    main()