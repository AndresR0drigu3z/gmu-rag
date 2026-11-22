"""Evaluate the RAG pipeline against a set of questions with known answers.

Each line of eval/questions.jsonl looks like:
  {"question": "...", "expected_source": "file.pdf", "expected_keywords": ["CS 310"]}
  {"question": "...", "unanswerable": true}   # model should say it doesn't know

Use "|" for acceptable alternatives, e.g. "expected_source": "a.pdf|b.pdf" or a keyword "twice|two".

Metrics:
  retrieval_hit  - the expected source file appears in the top-k retrieved chunks
  answer_hit     - every expected keyword (or one of its alternatives) appears in the answer (case-insensitive)
  refusal_ok     - for unanswerable questions, the model declined instead of guessing

Usage:  python eval/run_eval.py [-k 4]
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from config import settings  # noqa: E402
from rag import answer  # noqa: E402

REFUSAL_MARKERS = ["couldn't find", "could not find", "don't know", "not in the documents"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, default=settings.top_k)
    parser.add_argument("--file", default=str(HERE / "questions.jsonl"))
    args = parser.parse_args()

    questions = [json.loads(line) for line in open(args.file, encoding="utf-8") if line.strip()]
    rows = []
    for q in questions:
        t0 = time.time()
        result = answer(q["question"], args.k)
        latency = time.time() - t0
        text = result["answer"].lower()
        sources = [s["source"] for s in result["sources"]]

        row = {"question": q["question"], "latency_s": round(latency, 2), "answer": result["answer"]}
        if q.get("unanswerable"):
            row["refusal_ok"] = any(m in text for m in REFUSAL_MARKERS)
        else:
            row["retrieval_hit"] = any(src in sources for src in q["expected_source"].split("|"))
            row["answer_hit"] = all(
                any(alt.lower() in text for alt in kw.split("|")) for kw in q.get("expected_keywords", [])
            )
        rows.append(row)
        print(f"{'.' if all(v for k, v in row.items() if k.endswith(('_hit', '_ok'))) else 'x'}", end="", flush=True)
    print()

    def rate(key):
        vals = [r[key] for r in rows if key in r]
        return (sum(vals), len(vals)) if vals else (0, 0)

    print(f"\nResults (k={args.k}, model={settings.llm_model}, chunk={settings.chunk_size}/{settings.chunk_overlap})")
    for key in ["retrieval_hit", "answer_hit", "refusal_ok"]:
        hit, n = rate(key)
        if n:
            print(f"  {key:14s} {hit}/{n}  ({100 * hit / n:.0f}%)")
    print(f"  avg latency    {sum(r['latency_s'] for r in rows) / len(rows):.2f}s")

    out = HERE / "results" / f"{datetime.now():%Y%m%d-%H%M%S}.csv"
    fields = ["question", "retrieval_hit", "answer_hit", "refusal_ok", "latency_s", "answer"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved per-question results to {out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
