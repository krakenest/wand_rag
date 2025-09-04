\
import os
import json
import tempfile
import streamlit as st
from rag import ingest_files, answer_question, auto_enrich, OllamaEmbeddings, init_db, add_feedback

st.set_page_config(page_title="AI KB Search & Enrichment", page_icon="🔎", layout="wide")

st.title("🔎 AI-Powered Knowledge Base Search & Enrichment")
st.caption("Ollama + Streamlit • RAG • Structured JSON • Completeness & Enrichment")

with st.sidebar:
    st.header("Settings")
    llm = os.getenv("LLM_MODEL", "llama3.1:8b-instruct")
    emb = os.getenv("EMBED_MODEL", "nomic-embed-text")
    st.write(f"**LLM:** `{llm}`")
    st.write(f"**Embeddings:** `{emb}`")
    st.markdown("---")
    st.write("**Tips**")
    st.write("- Pull models via `ollama pull ...`")
    st.write("- Add PDFs/DOCXs/TXTs first")
    st.write("- Use Auto-enrichment sparingly")

tab1, tab2, tab3 = st.tabs(["📥 Upload", "💬 Ask", "🛠️ Feedback / Admin"])

# ---------------- Upload ----------------
with tab1:
    st.subheader("Upload & Index Documents")
    files = st.file_uploader("Upload multiple files (PDF, DOCX, TXT, MD)", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)
    if st.button("Index files", type="primary") and files:
        tmp_paths = []
        with st.spinner("Indexing..."):
            for f in files:
                # persist temp file
                suffix = "." + f.name.split(".")[-1]
                tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tf.write(f.read())
                tf.flush()
                tmp_paths.append(tf.name)
            embedder = OllamaEmbeddings()
            report = ingest_files(tmp_paths, embedder)
        st.success(f"Indexed {report.get('added', 0)} chunks from: {', '.join(report.get('files', []))}")

# ---------------- Ask ----------------
with tab2:
    st.subheader("Ask the Knowledge Base")
    q = st.text_input("Your question")
    c1, c2, c3 = st.columns(3)
    with c1:
        top_k = st.slider("Top-K", 3, 10, 5)
    with c2:
        auto = st.checkbox("Auto-enrichment (Wikipedia)", value=False)
    with c3:
        low_sim_threshold = st.slider("Low-similarity cutoff", 0.0, 0.6, 0.23, 0.01)

    if st.button("Get Answer", type="primary") and q:
        with st.spinner("Retrieving & generating..."):
            res = answer_question(q, top_k=top_k, low_sim_threshold=low_sim_threshold)

        st.markdown("#### Structured Output")
        st.json(res)

        st.markdown("#### Answer")
        st.write(res.get("answer", ""))
        st.progress(res.get("confidence", 0.0))

        if res.get("missing_info"):
            st.warning("**Missing information detected:**\n- " + "\n- ".join(res["missing_info"]))

        if res.get("enrichment_suggestions"):
            with st.expander("Suggested enrichment actions", expanded=True):
                for s in res["enrichment_suggestions"]:
                    st.markdown(f"- {s}")

        if res.get("citations"):
            with st.expander("Citations / Evidence"):
                for c in res["citations"]:
                    st.markdown(f"- **{c.get('source')}#{c.get('chunk_id')}** — {c.get('snippet','')[:300]}")

        if auto:
            st.markdown("---")
            st.info("Auto-enrichment enabled: querying a trusted external source (Wikipedia)...")
            report = auto_enrich(q, OllamaEmbeddings())
            if report:
                st.success(f"Indexed Wikipedia summary: {report['title']} ({report['chars_indexed']} chars). Re-run your question.")
            else:
                st.warning("No auto-enrichment material found for this query.")

        st.markdown("---")
        st.markdown("#### Rate this answer")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("👍 Helpful"):
                add_feedback(q, res.get("answer",""), float(res.get("confidence",0)), "up")
                st.success("Thanks for the feedback!")
        with c2:
            if st.button("👎 Not helpful"):
                add_feedback(q, res.get("answer",""), float(res.get("confidence",0)), "down")
                st.success("Feedback recorded.")

# ---------------- Admin ----------------
with tab3:
    st.subheader("Ratings & Flags")
    st.write("Feedback is stored in a local SQLite DB (`feedback.db`).")
    init_db()
    import sqlite3, pandas as pd
    conn = sqlite3.connect("feedback.db")
    df = pd.read_sql_query("SELECT * FROM feedback ORDER BY id DESC LIMIT 200", conn)
    conn.close()
    st.dataframe(df, use_container_width=True)
    st.caption("Use this table to refine prompts, thresholds, and retrieval settings.")
