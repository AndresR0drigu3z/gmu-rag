# GMU Academic Advisor — a RAG chatbot

Ask questions about George Mason University courses, degree requirements, and policies. The bot
answers **only** from indexed university documents and cites its sources.

> 🚧 Actively in development. See the [roadmap](#roadmap) and [eval results](#evaluation).

## How it works

```
data/raw/*.pdf|html|txt
        │  ingest.py: parse → clean → chunk (800 chars, 150 overlap)
        ▼
  ChromaDB (local vector store, all-MiniLM-L6-v2 embeddings)
        │  rag.py: embed question → top-k similarity search
        ▼
  Prompt = system rules + numbered source chunks + question
        │
        ▼
  LLM (any OpenAI-compatible API: Ollama / OpenAI) → answer with [n] citations
```

- **Grounded answers:** the system prompt restricts the model to the retrieved sources and tells it to
  say "I couldn't find that" instead of guessing.
- **Model-agnostic:** swap between a free local model (Ollama) and a hosted API by editing `.env`.
- **Evaluated:** `eval/run_eval.py` measures retrieval accuracy, answer accuracy, refusal behavior on
  unanswerable questions, and latency.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # pick Ollama or OpenAI

# put PDFs / saved HTML pages / .txt files in data/raw/
python src/ingest.py --reset
python src/rag.py "What are the prerequisites for CS 310?"
streamlit run src/app.py
```

## Evaluation

```bash
python eval/run_eval.py
```

| Date | Change | Retrieval hit | Answer hit | Correct refusals | Avg latency |
|------|--------|---------------|------------|------------------|-------------|
| _fill in after first run_ | baseline | | | | |

## Roadmap

- [x] Document ingestion (PDF, HTML, TXT) with chunking
- [x] Vector search with ChromaDB
- [x] Grounded answers with inline citations
- [x] Streamlit chat UI
- [x] Evaluation harness
- [ ] Expand eval set to 30+ questions
- [ ] Tune chunk size / top-k using eval results
- [ ] Hybrid search (BM25 + vectors) and re-ranking
- [ ] Logging / observability for queries, latency, and failures
- [ ] Deploy publicly
