"""Load documents from data/raw, split them into chunks, and store them in ChromaDB.

Usage:
    python src/ingest.py            # add/update documents
    python src/ingest.py --reset    # wipe the collection and rebuild
"""
import argparse
import hashlib
import re
from pathlib import Path

import chromadb
from bs4 import BeautifulSoup
from pypdf import PdfReader

from config import settings


def load_documents(raw_dir: Path):
    """Yield (text, metadata) pairs. PDFs yield one pair per page so we can cite pages."""
    for path in sorted(raw_dir.rglob("*")):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            reader = PdfReader(str(path))
            for i, page in enumerate(reader.pages):
                yield page.extract_text() or "", {"source": path.name, "page": i + 1}
        elif suffix in {".html", ".htm"}:
            soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            yield soup.get_text("\n"), {"source": path.name, "page": 0}
        elif suffix in {".txt", ".md"}:
            yield path.read_text(encoding="utf-8", errors="ignore"), {"source": path.name, "page": 0}


def clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Sliding-window chunking that tries to end each chunk on a whitespace boundary."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            if space != -1:
                end = space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="delete and rebuild the collection")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    if args.reset:
        try:
            client.delete_collection(settings.collection)
        except Exception:
            pass
    # Uses Chroma's default local embedding model (all-MiniLM-L6-v2) - free, no API key.
    collection = client.get_or_create_collection(settings.collection)

    ids, docs, metas = [], [], []
    for text, meta in load_documents(settings.raw_dir):
        for i, chunk in enumerate(chunk_text(clean(text), settings.chunk_size, settings.chunk_overlap)):
            key = f"{meta['source']}|{meta['page']}|{i}"
            ids.append(hashlib.sha1(key.encode()).hexdigest())
            docs.append(chunk)
            metas.append({**meta, "chunk": i})

    if not docs:
        print(f"No documents found in {settings.raw_dir}. Add PDFs/HTML/TXT files first.")
        return

    batch = 200
    for b in range(0, len(docs), batch):
        collection.upsert(ids=ids[b:b + batch], documents=docs[b:b + batch], metadatas=metas[b:b + batch])

    sources = sorted({m["source"] for m in metas})
    print(f"Indexed {len(docs)} chunks from {len(sources)} files into '{settings.collection}'.")


if __name__ == "__main__":
    main()
