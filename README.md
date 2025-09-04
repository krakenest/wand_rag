# AI-Powered Knowledge Base Search & Enrichment (Ollama + Streamlit)

**Challenge 2** — a pragmatic, end-to-end RAG system you can run locally.

https://github.com/krakenest/wand_rag

---

## ✨ Core Features

- **Document upload & storage**: PDF, DOCX, TXT, MD. Chunked and embedded via **Ollama** embeddings.
- **Natural-language search & LLM answer** powered by Ollama (default: `llama3.1:8b-instruct`).
- **Structured output**: JSON with `answer`, `confidence`, `missing_info`, `citations`, `enrichment_suggestions`.
- **Completeness check**: Heuristic on retrieval coverage + model self-assessment.
- **Graceful handling of irrelevant docs**: Detects low-similarity queries and returns an "insufficient info" response.
- **Enrichment suggestions**: Proposes what to add (docs/data/actions) when gaps are detected.
- **Stretch goals**:
  - **Auto-enrichment** *(optional)*: Pulls trusted summaries from Wikipedia REST API and lets you index them.
  - **Answer rating**: Thumbs up/down stored in SQLite for quality loops.

---

## 🧱 Architecture

```
Streamlit UI (app.py)
 ├── Upload: ingest files → chunk → embed (Ollama) → FAISS store
 ├── Ask: retrieve top-k → prompt LLM with context → JSON answer
 ├── Admin/Feedback: view ratings & flags (SQLite)
 
RAG Core (rag.py)
 ├── load_text() for PDF/DOCX/TXT/MD
 ├── chunk_text() simple sliding-window chunker
 ├── OllamaEmbeddings: /api/embeddings (e.g., nomic-embed-text or mxbai-embed-large)
 ├── FAISSStore: persist index + chunk metadata (./storage)
 ├── generate_answer(): constructs JSON-first prompt and calls Ollama /api/chat
 ├── completeness() heuristic + self-reported confidence
 ├── auto_enrich() optional Wikipedia fetch + indexing
```

---

## 🚀 Quickstart

1. **Install Ollama**: https://ollama.com/download  
   Start Ollama (it runs on `http://localhost:11434`).

2. **Pull models** (you can swap to your favorites):
   ```bash
   ollama pull llama3.1:8b-instruct
   ollama pull nomic-embed-text
   # or: ollama pull mxbai-embed-large
   ```

3. **Create & activate env, install deps**:
   ```bash
   python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
   pip install -r requirements.txt
   ```

4. **Run**:
   ```bash
   streamlit run app.py
   ```

5. **Use it**:
   - Upload a few PDFs/TXTs/DOCXs.
   - Ask a question.
   - Inspect JSON output, confidence, missing info, and enrichment suggestions.
   - (Optional) Enable *Auto-enrichment* to fetch a short trusted summary (Wikipedia) for gaps.
   - Rate the answer 👍/👎 to populate `feedback.db`.

---

## 🔧 Config

- Default LLM: `LLM_MODEL=llama3.1:8b-instruct`
- Default Embeddings: `EMBED_MODEL=nomic-embed-text`
- Ollama host: `OLLAMA_HOST=http://localhost:11434`
- Storage: `./storage` (FAISS index + chunk metadata pickle)
- SQLite DB: `feedback.db`

You can override via environment variables:
```bash
export LLM_MODEL=llama3.1:8b-instruct
export EMBED_MODEL=nomic-embed-text
export OLLAMA_HOST=http://localhost:11434
```

---

## 🧠 Design Decisions & Trade-offs

- **FAISS over Chroma**: keeps deps light, fast ANN search, easy disk persistence.
- **Simple chunker**: character-window + overlap → robust across formats w/o tokenizers.
- **Structured JSON-first prompting**: deterministic post-processing, easier eval.
- **Completeness**: combine retrieval coverage (similarity + unique sources) with model
  self-estimate; err on "insufficient info" if max similarity is low.
- **Auto-enrichment trust**: restricted to Wikipedia REST by default (auditable domain).
- **24h constraint**: prioritized a clean MVP with clear extension points (rerankers,
  hybrid search, better PDF parsing, eval harness).

---

## 🧪 Testing

- Add a few mixed docs; ask questions that are:
  - Directly answered (should cite chunks).
  - Partially answered (should list `missing_info` + enrichment).
  - Not in the KB (should return insufficient info w/ suggestions).

---

## 📈 Future Work

- Reranking (bge-reranker or cohere rerank via local/remote).
- HyDE query expansion; sparse+dense hybrid search.
- Multi-turn conversation memory.
- Source deduplication + better citation granularity.
- Guardrails & content safety classifier.
- Batch ingestion + background jobs for large corpora.

---

## 📹 Demo Recording (Loom)

Record a quick 3–5 min video walking through Upload → Ask → Auto-enrich → Rate. Include:
- What’s indexed
- A successful answer
- An incomplete answer → enrichment
- The thumbs up/down flow & how it is stored

---

## 🔐 Notes

- This is a local-first prototype. If you deploy, secure the endpoints and store.
- Ensure you respect document confidentiality. For networked auto-enrichment, keep the
  external calls limited to trusted domains or toggle them off by default.
