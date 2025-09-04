\
import os
import json
import time
import pickle
import faiss
import numpy as np
import requests
from typing import List, Dict, Any, Tuple, Optional
from pypdf import PdfReader
from docx import Document

# -------------------- Config --------------------

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b-instruct")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
STORAGE_DIR = os.getenv("STORAGE_DIR", "./storage")
INDEX_PATH = os.path.join(STORAGE_DIR, "faiss.index")
META_PATH = os.path.join(STORAGE_DIR, "chunks.pkl")

os.makedirs(STORAGE_DIR, exist_ok=True)

# -------------------- Utils --------------------

def load_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".txt", ".md"]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if ext == ".pdf":
        reader = PdfReader(path)
        text = []
        for page in reader.pages:
            text.append(page.extract_text() or "")
        return "\n".join(text)
    if ext == ".docx":
        doc = Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    raise ValueError(f"Unsupported file type: {ext}")

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    text = text.replace("\r\n", "\n")
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]

# -------------------- Embeddings via Ollama --------------------

class OllamaEmbeddings:
    def __init__(self, model: str = EMBED_MODEL, host: str = OLLAMA_HOST):
        self.model = model
        self.host = host.rstrip("/")

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            resp = requests.post(f"{self.host}/api/embeddings", json={"model": self.model, "prompt": t})
            resp.raise_for_status()
            data = resp.json()
            vecs.append(np.array(data["embedding"], dtype="float32"))
        return np.vstack(vecs)

# -------------------- FAISS Store --------------------

class FAISSStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.id2meta: Dict[int, Dict[str, Any]] = {}
        self.next_id = 0

    @staticmethod
    def normalize(v: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        return v / norms

    def add(self, vecs: np.ndarray, metas: List[Dict[str, Any]]):
        vecs = self.normalize(vecs).astype("float32")
        ids = np.arange(self.next_id, self.next_id + vecs.shape[0])
        self.index.add(vecs)
        for i, m in zip(ids, metas):
            self.id2meta[int(i)] = m
        self.next_id += vecs.shape[0]

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[int, float, Dict[str, Any]]]:
        q = self.normalize(query_vec).astype("float32")
        D, I = self.index.search(q, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            results.append((int(idx), float(score), self.id2meta.get(int(idx), {})))
        return results

    def save(self, index_path=INDEX_PATH, meta_path=META_PATH):
        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump({"id2meta": self.id2meta, "next_id": self.next_id, "dim": self.dim}, f)

    @classmethod
    def load(cls, index_path=INDEX_PATH, meta_path=META_PATH) -> "FAISSStore":
        if not (os.path.exists(index_path) and os.path.exists(meta_path)):
            raise FileNotFoundError("Index or metadata not found")
        index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            data = pickle.load(f)
        store = cls(dim=data["dim"])
        store.index = index
        store.id2meta = data["id2meta"]
        store.next_id = data["next_id"]
        return store

# -------------------- Ingest --------------------

def ingest_files(file_paths: List[str], embedder: OllamaEmbeddings) -> Dict[str, Any]:
    # Load or init store lazily (probe using a single embedding to determine dim)
    probe = embedder.embed(["dimension probe"])
    dim = probe.shape[1]
    try:
        store = FAISSStore.load()
    except Exception:
        store = FAISSStore(dim)

    all_chunks, metas = [], []
    for fp in file_paths:
        text = load_text(fp)
        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            metas.append({"source": os.path.basename(fp), "chunk_id": i, "text": c})
            all_chunks.append(c)

    if not all_chunks:
        return {"added": 0}

    vecs = embedder.embed(all_chunks)
    store.add(vecs, metas)
    store.save()
    return {"added": len(all_chunks), "files": [os.path.basename(p) for p in file_paths]}

# -------------------- LLM Call via Ollama --------------------

def call_llm_json(question: str, contexts: List[Dict[str, Any]], model: str = LLM_MODEL, host: str = OLLAMA_HOST) -> Dict[str, Any]:
    system = """You are a meticulous enterprise QA assistant. 
Follow the rules strictly:
1) Use ONLY the provided context to answer. If unsure, say so.
2) Return STRICT JSON with keys: answer (string), confidence (0..1), missing_info (array of strings), citations (array of {source, chunk_id, snippet}), enrichment_suggestions (array of strings).
3) No extra text outside JSON. Keep answer concise but complete.
"""

    ctx_blocks = []
    for c in contexts:
        snippet = c["text"].replace("\n", " ")[:700]
        ctx_blocks.append(f"[{c['source']}#{c['chunk_id']}] {snippet}")

    user = f"""Question: {question}

Context:
{chr(10).join(ctx_blocks)}

Return ONLY valid JSON with the schema described."""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {"temperature": 0.2}
    }
    resp = requests.post(f"{host.rstrip('/')}/api/chat", json=payload, stream=False)
    resp.raise_for_status()
    data = resp.json()
    content = data.get("message", {}).get("content", "") if isinstance(data, dict) else ""
    # Extract JSON robustly
    try:
        j = json.loads(content)
    except Exception:
        # Fallback: find first {...} block
        import re
        m = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not m:
            raise ValueError(f"LLM did not return JSON: {content[:200]}")
        j = json.loads(m.group(0))
    # Ensure keys
    for k in ["answer", "confidence", "missing_info", "citations", "enrichment_suggestions"]:
        j.setdefault(k, [] if k in ["missing_info", "citations", "enrichment_suggestions"] else "")
    # Clamp confidence
    try:
        j["confidence"] = float(j.get("confidence", 0))
    except Exception:
        j["confidence"] = 0.0
    j["confidence"] = max(0.0, min(1.0, j["confidence"]))
    return j

# -------------------- Retrieval & Answer --------------------

def retrieve(query: str, embedder: OllamaEmbeddings, k: int = 5):
    store = FAISSStore.load()
    qv = embedder.embed([query])
    hits = store.search(qv, top_k=k)
    return hits

def completeness(hits: List[Tuple[int, float, Dict[str, Any]]]) -> Dict[str, Any]:
    if not hits:
        return {"max_sim": 0.0, "avg_sim": 0.0, "unique_sources": 0}
    sims = [h[1] for h in hits]
    sources = {h[2].get("source", "unknown") for h in hits}
    return {"max_sim": float(max(sims)), "avg_sim": float(sum(sims) / len(sims)), "unique_sources": len(sources)}

def assemble_contexts(hits: List[Tuple[int, float, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    return [h[2] for h in hits]

def answer_question(question: str, top_k: int = 5, low_sim_threshold: float = 0.23) -> Dict[str, Any]:
    embedder = OllamaEmbeddings()
    hits = retrieve(question, embedder, k=top_k)
    stats = completeness(hits)
    contexts = assemble_contexts(hits)

    if stats["max_sim"] < low_sim_threshold:
        return {
            "answer": "I don't have enough information in the current knowledge base to answer this confidently.",
            "confidence": 0.15,
            "missing_info": [f"Relevant documents for: '{question}'"],
            "citations": [],
            "enrichment_suggestions": [
                "Add product specs, policies, or design docs relevant to the question.",
                "Ingest API reference or runbooks if the topic is technical.",
                "Provide a brief SME-provided summary document."
            ],
            "retrieval": stats
        }

    j = call_llm_json(question, contexts)
    j["retrieval"] = stats
    # attach machine citations if model omitted
    if not j.get("citations"):
        j["citations"] = [
            {"source": h[2].get("source"), "chunk_id": h[2].get("chunk_id"),
             "snippet": (h[2].get("text", "")[:180]).replace("\n", " ")}
            for h in hits[:3]
        ]
    return j

# -------------------- Auto-enrichment (Wikipedia) --------------------

def wiki_search(term: str) -> Optional[str]:
    try:
        r = requests.get("https://en.wikipedia.org/w/api.php", params={
            "action": "opensearch", "search": term, "limit": 1, "namespace": 0, "format": "json"
        }, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data and len(data) >= 2 and data[1]:
            return data[1][0]
    except Exception:
        return None
    return None

def wiki_summary(title: str) -> Optional[str]:
    try:
        r = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}", timeout=10,
                         headers={"accept": "application/json"})
        r.raise_for_status()
        data = r.json()
        return data.get("extract")
    except Exception:
        return None

def auto_enrich(query: str, embedder: OllamaEmbeddings) -> Optional[Dict[str, Any]]:
    title = wiki_search(query)
    if not title:
        return None
    summary = wiki_summary(title)
    if not summary:
        return None
    # Index as a virtual "wiki" source
    store = None
    try:
        store = FAISSStore.load()
    except Exception:
        # init an empty store (determine dim via probe)
        probe = embedder.embed(["dimension probe"])
        store = FAISSStore(probe.shape[1])

    vec = embedder.embed([summary])
    meta = {"source": f"Wikipedia:{title}", "chunk_id": 0, "text": summary}
    store.add(vec, [meta])
    store.save()
    return {"title": title, "chars_indexed": len(summary)}

# -------------------- Feedback (SQLite) --------------------

import sqlite3

DB_PATH = "feedback.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS feedback(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT,
        question TEXT,
        answer TEXT,
        confidence REAL,
        rating TEXT,
        notes TEXT
    )
    """)
    conn.commit()
    conn.close()

def add_feedback(question: str, answer: str, confidence: float, rating: str, notes: str = ""):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO feedback(ts, question, answer, confidence, rating, notes) VALUES(?,?,?,?,?,?)",
                (time.strftime("%Y-%m-%d %H:%M:%S"), question, answer, confidence, rating, notes))
    conn.commit()
    conn.close()
