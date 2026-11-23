"""Retrieve relevant chunks and ask the LLM to answer using only those chunks.

Usage:
    python src/rag.py "What are the prerequisites for CS 310?"
"""
import argparse
import re

import chromadb
from openai import OpenAI

from config import settings

SYSTEM_PROMPT = """You are an assistant that answers questions about George Mason University \
academic programs and policies.
Rules:
- Use ONLY the numbered sources provided. Do not use outside knowledge.
- Cite the sources you used inline, like [1] or [2][3].
- If the sources do not contain the answer, say "I couldn't find that in the documents I have." \
Do not guess."""

_collection = None
_llm = None


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        _collection = client.get_collection(settings.collection)
    return _collection


def get_llm():
    global _llm
    if _llm is None:
        _llm = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    return _llm


# Course codes in a question: "CS 310", "math 203", "CS310"
COURSE_CODE = re.compile(r"\b([A-Za-z]{2,4})\s?(\d{3})\b")


def exact_course_matches(question: str) -> list[dict]:
    """If the question names a course code, fetch that course's chunk(s) directly by metadata.
    Small embedding models are weak at exact identifiers like "CS 310", so we don't rely on them for it."""
    codes = sorted({f"{subj.upper()} {num}" for subj, num in COURSE_CODE.findall(question)})
    matches = []
    for code in codes:
        res = get_collection().get(where={"course": code})
        for doc, meta in zip(res["documents"], res["metadatas"]):
            matches.append({"text": doc, "source": meta["source"], "page": meta["page"],
                            "distance": 0.0, "match": "exact"})
    return matches


def vector_search(question: str, k: int) -> list[dict]:
    res = get_collection().query(query_texts=[question], n_results=k)
    return [
        {"text": doc, "source": meta["source"], "page": meta["page"], "distance": dist, "match": "vector"}
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def retrieve(question: str, k: int = settings.top_k) -> list[dict]:
    """Exact course-code matches first, then fill the remaining slots with vector search."""
    exact = exact_course_matches(question) if settings.course_lookup else []
    merged, seen = [], set()
    for chunk in exact + vector_search(question, k):
        key = (chunk["source"], chunk["text"])
        if key not in seen:
            seen.add(key)
            merged.append(chunk)
    return merged[:max(k, len(exact))]


def build_prompt(question: str, chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        where = f"{c['source']}" + (f", p.{c['page']}" if c["page"] else "")
        blocks.append(f"[{i}] ({where})\n{c['text']}")
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def answer(question: str, k: int = settings.top_k) -> dict:
    chunks = retrieve(question, k)
    resp = get_llm().chat.completions.create(
        model=settings.llm_model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, chunks)},
        ],
    )
    return {"answer": resp.choices[0].message.content, "sources": chunks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=settings.top_k)
    args = parser.parse_args()

    result = answer(args.question, args.k)
    print(result["answer"], "\n\n--- sources ---")
    for i, s in enumerate(result["sources"], start=1):
        how = "exact course match" if s["match"] == "exact" else f"distance {s['distance']:.3f}"
        print(f"[{i}] {s['source']} p.{s['page']}  ({how})")


if __name__ == "__main__":
    main()
