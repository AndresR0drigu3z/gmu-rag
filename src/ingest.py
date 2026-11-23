"""Load documents from data/raw, split them into chunks, and store them in ChromaDB.

Usage:
    python src/ingest.py            # add/update documents
    python src/ingest.py --reset    # wipe the collection and rebuild

Chunking strategies (set CHUNK_STRATEGY in .env):
    structured (default) - strip print headers/footers, join pages so text that crosses a page break
                           stays together, and split course catalogs at each course heading
                           ("CS 310:", "MATH 125:") so every chunk is one complete course.
    page                 - the original baseline: fixed-size chunks within each PDF page.
"""
import argparse
import bisect
import hashlib
import re
from pathlib import Path

import chromadb
from bs4 import BeautifulSoup
from pypdf import PdfReader

from config import settings

# Browser print artifacts repeated on every PDF page, e.g.
#   "11/22/26, 2:10 PM Computer Science (CS) | George Mason University Catalog"
#   "https://catalog.gmu.edu/courses/cs/ 5/47"
PRINT_NOISE = re.compile(
    r"^\d{1,2}/\d{1,2}/\d{2,4},\s+\d{1,2}:\d{2}\s*[AP]M.*$"  # date/time header line
    r"|^https?://\S+.*$",                                    # URL + page-number footer line
    re.MULTILINE,
)
# Collapsible-section labels from the catalog website ("EXPAND300 Level Courses", "HIDE ALL")
UI_LABELS = re.compile(r"EXPAND|HIDE ALL")
# A course heading at the start of a line: "CS 310:Data Structures." / "MATH 125:Discrete ..."
COURSE_HEADING = re.compile(r"^[A-Z]{2,4} \d{3}[A-Z]?:", re.MULTILINE)
MIN_COURSES_FOR_CATALOG = 5  # treat a document as a course catalog if it has at least this many


def load_documents(raw_dir: Path):
    """Yield (source_name, pages, is_paged). PDFs give one string per page; other files give one string."""
    for path in sorted(raw_dir.rglob("*")):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            reader = PdfReader(str(path))
            yield path.name, [page.extract_text() or "" for page in reader.pages], True
        elif suffix in {".html", ".htm"}:
            soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            yield path.name, [soup.get_text("\n")], False
        elif suffix in {".txt", ".md"}:
            yield path.name, [path.read_text(encoding="utf-8", errors="ignore")], False


def clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def chunk_spans(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """Sliding-window chunking that tries to end each chunk on a whitespace boundary.
    Returns (start, end) character positions so callers can map chunks back to pages."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    spans, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            if space != -1:
                end = space
        if text[start:end].strip():
            spans.append((start, end))
        if end >= len(text):
            break
        start = end - overlap
    return spans


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    return [text[a:b].strip() for a, b in chunk_spans(text, size, overlap)]


def page_chunks(pages, is_paged, size, overlap):
    """Baseline strategy: chunk each page independently. Yields (chunk_text, page_number)."""
    for i, page in enumerate(pages):
        for chunk in chunk_text(clean(page), size, overlap):
            yield chunk, (i + 1 if is_paged else 0)


def structured_chunks(pages, is_paged, size, overlap):
    """Improved strategy. Yields (chunk_text, page_number)."""
    # 1. Clean each page and join them, remembering where each page starts in the combined text.
    page_starts, parts, pos = [], [], 0
    for page in pages:
        page = re.sub(r"[^\S\n]", " ", page)  # non-breaking & other odd spaces -> normal space ("CS\xa0310")
        page = clean(UI_LABELS.sub("", PRINT_NOISE.sub("", page)))
        page_starts.append(pos)
        parts.append(page)
        pos += len(page) + 1  # +1 for the "\n" used to join
    text = "\n".join(parts)

    def page_of(offset: int) -> int:
        return bisect.bisect_right(page_starts, offset) if is_paged else 0

    headings = [m.start() for m in COURSE_HEADING.finditer(text)]

    # 2a. Course catalogs: one chunk per course (long courses are sub-split, keeping the title).
    if len(headings) >= MIN_COURSES_FOR_CATALOG:
        bounds = [0] + headings + [len(text)]
        for a, b in zip(bounds, bounds[1:]):
            section = text[a:b].strip()
            if not section:
                continue
            if len(section) <= 2 * size:
                yield section, page_of(a)
                continue
            title = section.split("\n", 1)[0] if a in headings else ""
            for s, e in chunk_spans(section, size, overlap):
                piece = section[s:e].strip()
                if title and not piece.startswith(title):
                    piece = f"{title}\n{piece}"
                yield piece, page_of(a + s)
        return

    # 2b. Everything else: fixed-size chunks over the whole document, so page breaks don't split text.
    for s, e in chunk_spans(text, size, overlap):
        yield text[s:e].strip(), page_of(s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="delete and rebuild the collection")
    args = parser.parse_args()

    strategy = structured_chunks if settings.chunk_strategy == "structured" else page_chunks

    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    if args.reset:
        try:
            client.delete_collection(settings.collection)
        except Exception:
            pass
    # Uses Chroma's default local embedding model (all-MiniLM-L6-v2) - free, no API key.
    collection = client.get_or_create_collection(settings.collection)

    ids, docs, metas = [], [], []
    for source, pages, is_paged in load_documents(settings.raw_dir):
        for i, (chunk, page) in enumerate(strategy(pages, is_paged, settings.chunk_size, settings.chunk_overlap)):
            ids.append(hashlib.sha1(f"{source}|{i}".encode()).hexdigest())
            docs.append(chunk)
            metas.append({"source": source, "page": page, "chunk": i})

    if not docs:
        print(f"No documents found in {settings.raw_dir}. Add PDFs/HTML/TXT files first.")
        return

    batch = 200
    for b in range(0, len(docs), batch):
        collection.upsert(ids=ids[b:b + batch], documents=docs[b:b + batch], metadatas=metas[b:b + batch])

    sources = sorted({m["source"] for m in metas})
    print(f"Indexed {len(docs)} chunks from {len(sources)} files into '{settings.collection}' "
          f"(strategy: {settings.chunk_strategy}).")


if __name__ == "__main__":
    main()
