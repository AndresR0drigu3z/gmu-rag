# GMU Academic Advisor: a RAG chatbot

Ask questions about George Mason University courses, Computer Science degree requirements, and academic
policies. The bot answers **only** from official catalog documents, cites its sources, and says
"I couldn't find that" instead of guessing.

> 🚧 Actively in development. See [results](#evaluation-results) and the [roadmap](#roadmap).

## How it works

```
data/raw/*.pdf  (5 GMU catalog documents, ~130 pages)
        │  ingest.py: parse → strip print headers/footers → join pages
        │             → structured chunking (one chunk per course) → tag chunks with course codes
        ▼
  ChromaDB  (local vector store, all-MiniLM-L6-v2 embeddings, 323 chunks)
        │  rag.py: 1) exact course-code lookup ("CS 310" → that course's chunk)
        │          2) fill remaining slots with top-k vector similarity search
        ▼
  Prompt = grounding rules + numbered source chunks + question
        │
        ▼
  LLM (Llama 3.2 via Ollama, or any OpenAI-compatible API) → answer with [n] citations
```

**Key design choices**
- **Grounded answers:** the system prompt restricts the model to retrieved sources and requires it to refuse
  rather than guess. For a policy assistant, a wrong confident answer is worse than "I don't know."
- **Structure-aware chunking:** course catalogs are split at each course heading so every chunk is one
  complete course (name, description, prerequisites together), and pages are joined first so text that
  crosses a PDF page break isn't split.
- **Hybrid retrieval:** small embedding models treat identifiers like "CS 310" as fuzzy meaning, so
  questions that name a course code fetch that course directly via metadata before vector search.
- **Model-agnostic:** swap between a free local model (Ollama) and a hosted API by editing `.env`.
- **Reproducible experiments:** every change is behind a setting (`CHUNK_STRATEGY`, `COURSE_LOOKUP`), so
  earlier configurations can be re-run and compared.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                       # defaults to Ollama + llama3.2

ollama pull llama3.2                       # free local LLM (https://ollama.com)
python src/ingest.py --reset               # build the vector index from data/raw/
python src/rag.py "What are the prerequisites for CS 310?"
streamlit run src/app.py                   # chat UI
python eval/run_eval.py                    # run the evaluation
```

## Evaluation

`eval/questions.jsonl` holds 17 questions with answers verified against the source documents:
14 answerable (course prerequisites, degree requirements, CS major policies, university policies) and
3 unanswerable (the bot should refuse). `eval/run_eval.py` reports:

- **answer_hit:** the answer contains the expected facts
- **refusal_ok:** unanswerable questions are declined instead of answered
- **retrieval_hit:** the expected source file appears in the retrieved chunks
- **latency**

Per-question results for every run are saved in `eval/results/`.

### Evaluation results

Llama 3.2 (3B) via Ollama, top-k = 4, 800-character chunks.

| Configuration | Answer accuracy | Correct refusals | Avg latency |
|---|---|---|---|
| Baseline: fixed-size chunks per PDF page | 8/14 (57%) | 3/3 | 2.0s |
| + Structured chunking | 8/14 (57%) | 3/3 | 2.0s |
| + Exact course-code lookup | **11/14 (79%)** | **3/3** | 1.9s |

### What I learned

- **The first bug:** asked for CS 310's prerequisites, the baseline answered "CS 310 and CS 330." The
  top-ranked chunk was the end of a *different* course (the compiler course), whose title was on the
  previous PDF page, so the chunk contained "Required Prerequisites: CS 310 ... CS 330" with no course
  name. The real CS 310 entry wasn't in the top 8 results. This was a **retrieval** failure caused by
  page-based chunking.
- **Data surprises:** the catalog PDFs use non-breaking spaces in course codes ("CS\xa0310"), which broke
  the first version of the course-heading parser. Inspecting real output caught it.
- **Averages hide changes:** structured chunking left accuracy at 57%, but it fixed 2 questions
  (including a prerequisite split across a page break) and regressed 2 others. Per-question results and
  re-running the full eval after every change (regression testing) made that visible.
- **Enabling changes matter:** structured chunking didn't move the metric alone, but tagging each
  one-course chunk with its course code is what made the exact-lookup step possible (57% → 79%).
- **Failing safely:** when retrieval missed, the model said it couldn't find the answer instead of
  inventing one. Refusals stayed at 3/3 across every configuration.
- **Metric limits:** `retrieval_hit` checks the source *file*, so it scored 93% even when the right file
  but wrong section was retrieved. A chunk-level metric is on the roadmap.

## Roadmap

- [x] Document ingestion (PDF, HTML, TXT)
- [x] Vector search with ChromaDB
- [x] Grounded answers with inline citations and refusal behavior
- [x] Streamlit chat UI
- [x] Evaluation harness with saved per-question results
- [x] Structure-aware chunking (one chunk per course, cross-page joining, header/footer removal)
- [x] Exact course-code lookup before vector search
- [ ] Explain catalog grade codes in answers ("CS 211C" → "CS 211 with a C or better")
- [ ] Add document titles to chunks to fix the remaining policy-question failures
- [ ] Chunk-level retrieval metric
- [ ] Expand the eval set to 30+ questions
- [ ] Hybrid keyword (BM25) + vector search and re-ranking
- [ ] PII-redaction guardrail before text reaches the LLM
- [ ] Compare models (local vs. hosted) on accuracy, latency, and cost
- [ ] Query logging / observability
- [ ] Deploy publicly
