"""Make sure the files the app needs exist; download any missing ones from a PRIVATE GitHub repo.

The public repo holds code only. The search index contains text from the banks' documents, so it lives in a
separate private repo and is fetched at startup with a read-only token (set in the host's secrets):
    DATA_REPO        e.g. "your-username/malaysia-credit-card-rag-data"
    DATA_REPO_TOKEN  a fine-grained GitHub token with read-only access to that repo's contents
Locally the files already exist, so nothing is downloaded.
"""
import os

import requests

from config import ROOT, load_config


def required_files(cfg):
    strategy = cfg["retrieval"]["strategy"]
    index_dir = cfg["paths"]["index_dir"]
    return [f"{index_dir}/{strategy}.faiss", f"{index_dir}/{strategy}_chunks.jsonl", cfg["catalog"]["path"]]


def ensure_data(cfg=None):
    cfg = cfg or load_config()
    missing = [f for f in required_files(cfg) if not (ROOT / f).exists()]
    if not missing:
        return
    repo, token = os.getenv("DATA_REPO"), os.getenv("DATA_REPO_TOKEN")
    if not repo or not token:
        raise RuntimeError(f"Missing data files {missing}, and DATA_REPO / DATA_REPO_TOKEN are not set.")
    for path in missing:
        response = requests.get(f"https://api.github.com/repos/{repo}/contents/{path}", timeout=60,
                                headers={"Authorization": f"Bearer {token}",
                                         "Accept": "application/vnd.github.raw+json"})
        response.raise_for_status()
        target = ROOT / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.content)
        print(f"Downloaded {path} ({len(response.content):,} bytes)")


if __name__ == "__main__":  # local test: downloads anything missing (reads DATA_REPO / DATA_REPO_TOKEN from .env)
    from dotenv import load_dotenv

    load_dotenv()
    cfg = load_config()
    ensure_data(cfg)
    for f in required_files(cfg):
        print(f"{'OK     ' if (ROOT / f).exists() else 'MISSING'} {f}")
