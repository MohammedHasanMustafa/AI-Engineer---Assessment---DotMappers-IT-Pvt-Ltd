from __future__ import annotations

import json
import os
import re
from dataclasses import asdict

import pandas as pd
import streamlit as st

from config import settings
from evaluation.evaluate import DATASET_PATH, load_dataset, run_evaluation
from rag.corpus import SAMPLE_PAPERS, download_sample_papers
from rag.eval_log import AMBER, GREEN
from rag.generation import build_user_prompt
from rag.models import RAGResponse
from rag.pipeline import RAGPipeline
from rag.security import extract_payload_markers, neutralize

st.set_page_config(page_title="Research assistant with evidence", page_icon="🔎", layout="wide")

STATUS_UI = {
    "answered": ("Answered from the documents", "green"),
    "conflicting": ("Sources disagree", "orange"),
    "insufficient_evidence": ("Insufficient evidence", "red"),
    "blocked": ("Insufficient evidence: response withheld by the security guard", "red"),
    "rejected": ("Question not processed", "red"),
}
STATUS_ICON = {"indexed": "✅", "skipped": "⏭️", "empty": "⚪", "error": "❌", "unsupported": "🚫"}


# ============================================================================ setup
@st.cache_resource(show_spinner="Loading models and index (first start downloads ~150 MB of models)…")
def get_pipeline() -> RAGPipeline:
    if os.getenv("RAG_SMOKE_TEST") == "1":           # CI only: no model downloads
        from rag.embeddings import HashingEmbedder
        pipe = RAGPipeline(settings, embedder=HashingEmbedder(), reranker=None, llm=None)
    else:
        pipe = RAGPipeline(settings)
    pipe.ingest_directory(settings.docs_dir, settings.upload_dir)   # skips already-indexed files
    return pipe


def md_escape(text: str) -> str:
    """Model and document text is untrusted: render it literally, never as markdown/links."""
    return re.sub(r"([\\`*_{}\[\]()#+!|<>~$])", r"\\\1", text)


def fmt(v, nd=2):
    return "n/a" if v is None else f"{v:.{nd}f}"


pipe = get_pipeline()
ss = st.session_state

# ============================================================================ sidebar
with st.sidebar:
    st.subheader("Retrieval settings")
    top_k = st.slider("Passages to retrieve (top-k)", 1, 15, settings.top_k)
    use_hybrid = st.toggle("Hybrid search (vector + BM25, fused with RRF)", value=True)
    use_rerank = st.toggle("Rerank with a cross-encoder", value=True, disabled=pipe.reranker is None)
    sources = st.multiselect("Only search these documents", sorted(pipe.store.doc_meta),
                             help="Metadata filter. Leave empty to search everything.")
    gen_mode = st.radio("Answer generator", ["auto", "extractive"],
                        format_func=lambda m: {"auto": "Local LLM (falls back to extractive)",
                                               "extractive": "Extractive only (quote sentences)"}[m])
    with st.expander("Advanced"):
        default_thr = settings.min_rerank_relevance if use_rerank else settings.min_vector_relevance
        min_rel = st.slider("Refuse when best relevance is below", 0.0, 1.0, float(default_thr), 0.01)

    st.divider()
    ok, reason = pipe.llm_status()
    (st.success if ok else st.warning)(reason)
    st.caption(f"Embeddings: `{settings.embedding_model}`  \nReranker: `{settings.reranker_model}`  \n"
               f"Indexed: {len(pipe.store.doc_meta)} documents, {len(pipe.store)} chunks")

ASK_KW = dict(top_k=top_k, use_hybrid=use_hybrid, use_rerank=use_rerank, sources=sources or None,
              generator=gen_mode, min_relevance=min_rel)


# ============================================================================ rendering
def render_response(resp: RAGResponse, key: str) -> None:
    label, color = STATUS_UI.get(resp.status, (resp.status, "grey"))
    st.markdown(f"#### :{color}[{label}]")
    if resp.message:
        (st.info if resp.status != "rejected" else st.error)(resp.message)

    if resp.evidence:
        st.markdown("**Evidence** (each statement is backed by the cited passage)")
        for i, e in enumerate(resp.evidence, 1):
            tags = " ".join(f"`{l}`" for l in e.labels)
            support = "" if e.support is None else (
                f"  ·  support {e.support:.2f} {'✅' if e.supported else '⚠️ weak'}")
            st.markdown(f"{i}. {md_escape(e.claim)}  {tags}{support}")
    if resp.conflicts:
        st.markdown("**Where the sources disagree**")
        st.warning(resp.conflicts)
    if resp.inference.strip():
        st.markdown("**Model inference** (reasoning beyond what the sources literally say)")
        st.info(resp.inference)

    st.markdown("**Citations**")
    cited = resp.cited_retrieved()
    if cited:
        for rc in cited:
            st.markdown(f"- `{rc.label}` **{md_escape(rc.chunk.source)}**, {rc.chunk.pages}, "
                        f"chunk `{rc.chunk.chunk_id}`")
    else:
        st.caption("No citations: no answer was given.")

    lat = resp.latency_ms
    cols = st.columns(6)
    cols[0].metric("Total latency", f"{lat.get('total_ms', 0):.0f} ms")
    cols[1].metric("Retrieval", f"{lat.get('retrieval_ms', 0):.0f} ms")
    cols[2].metric("Generation", f"{lat.get('generation_ms', 0):.0f} ms")
    cols[3].metric("Best relevance", fmt(resp.relevance))
    cols[4].metric("Groundedness", fmt(resp.groundedness))
    cols[5].metric("Generator", resp.generator or "n/a")

    for ev in resp.security_events:
        (st.error if ev.startswith("BLOCKED") else st.warning)(f"🛡️ {ev}")
    for w in resp.warnings:
        st.caption(f"⚠️ {w}")

    if resp.retrieved:
        cited_labels = set(resp.cited_labels)
        st.markdown(f"**Retrieved passages** ({len(resp.retrieved)}, "
                    f"{lat.get('dedup_removed', 0)} near-duplicates removed)")
        st.dataframe(pd.DataFrame([{
            "id": rc.label, "cited": "✔" if rc.label in cited_labels else "",
            "document": rc.chunk.source, "pages": rc.chunk.pages, "chunk": rc.chunk.chunk_id,
            "vector": rc.vector_score, "bm25": rc.bm25_score, "rerank logit": rc.rerank_score,
            "relevance": rc.relevance, "injection flags": ", ".join(rc.chunk.injection_flags),
        } for rc in resp.retrieved]), hide_index=True, width="stretch",
            column_config={c: st.column_config.NumberColumn(format="%.3f")
                           for c in ("vector", "bm25", "rerank logit", "relevance")})
        for rc in resp.retrieved:
            flag = " 🛡️ contains suspected injection" if rc.chunk.injection_flags else ""
            with st.expander(f"{rc.label} · {rc.chunk.source} · {rc.chunk.pages} · "
                             f"relevance {rc.relevance:.2f}{flag}", expanded=rc.label in cited_labels):
                st.text_area("Passage text (shown as plain text, never executed)", rc.chunk.text,
                             height=160, disabled=True, key=f"{key}-{rc.label}-{rc.chunk.chunk_id}")
                if rc.neutralized:
                    st.caption(f"{rc.neutralized} sentence(s) were replaced before this passage "
                               "was shown to the model.")


# ============================================================================ tabs
tab_ask, tab_docs, tab_log, tab_eval, tab_inj = st.tabs(
    ["Ask", "Documents", "Evaluation log", "Run evaluation", "Prompt-injection demo"])

# ---------------------------------------------------------------------------- Ask
with tab_ask:
    st.markdown("### Ask a question about the indexed research documents")
    examples = ["How many attention heads does the base Transformer use?",
                "What does ReAct interleave when prompting language models?",
                "What chunk size does the Orion vector search guide recommend?",
                "How many parameters does GPT-4 have?"]
    ex = st.selectbox("Example questions", ["(type your own)"] + examples)
    question = st.text_area("Question", value="" if ex == "(type your own)" else ex, height=90)
    if st.button("Get answer", type="primary"):
        with st.spinner("Retrieving evidence and generating a grounded answer…"):
            ss["last"] = pipe.ask(question, **ASK_KW)
    if "last" in ss:
        render_response(ss["last"], "ask")

# ---------------------------------------------------------------------------- Documents
with tab_docs:
    st.markdown("### Knowledge base")
    ups = st.file_uploader("Add PDF, TXT or Markdown files", type=["pdf", "txt", "md", "markdown"],
                           accept_multiple_files=True)
    if st.button("Index uploaded files", disabled=not ups, type="primary"):
        rows, bar = [], st.progress(0.0)
        for n, f in enumerate(ups, 1):
            with st.status(f"Indexing {f.name}…", expanded=False) as s:
                r = pipe.ingest_upload(f.name, f.getvalue())
                s.update(label=f"{STATUS_ICON.get(r.status, '')} {f.name}: {r.message}",
                         state="complete" if r.status in ("indexed", "skipped") else "error")
            rows.append(asdict(r))
            bar.progress(n / len(ups))
        ss["ingest"] = rows
        st.rerun()
    if ss.get("ingest"):
        st.markdown("**Last indexing run**")
        st.dataframe(pd.DataFrame([{"": STATUS_ICON.get(r["status"], ""), **r} for r in ss["ingest"]]),
                     hide_index=True, width="stretch")

    st.markdown("**Indexed documents**")
    if pipe.store.doc_meta:
        docs = pd.DataFrame([{"document": m["source"], "pages": m["n_pages"], "chunks": m["n_chunks"],
                              "chunks with suspected injection": m["injection_chunks"],
                              "indexed at": m["ingested_at"]} for m in pipe.store.doc_meta.values()])
        st.dataframe(docs, hide_index=True, width="stretch")
        st.caption(f"Total pages: {int(docs['pages'].sum())} · total chunks: {len(pipe.store)}")
        c1, c2 = st.columns([3, 1])
        victim = c1.selectbox("Remove a document", sorted(pipe.store.doc_meta), label_visibility="collapsed")
        if c2.button("Remove document"):
            pipe.delete_document(victim)
            st.rerun()
    else:
        st.info("No documents indexed yet. Upload files above or download the sample papers below.")

    st.markdown("**Sample corpus**")
    st.caption("; ".join(t for _, _, t in SAMPLE_PAPERS))
    c1, c2 = st.columns(2)
    if c1.button("Download and index the sample arXiv papers"):
        with st.spinner("Downloading papers from arxiv.org…"):
            for name, status, msg in download_sample_papers(settings.docs_dir):
                (st.error if status == "failed" else st.caption)(f"{name}: {status} {msg}")
        with st.spinner("Indexing…"):
            ss["ingest"] = [asdict(r) for r in pipe.ingest_directory(settings.docs_dir)]
        st.rerun()
    if c2.button("Rebuild the whole index"):
        with st.spinner("Re-indexing every document…"):
            ss["ingest"] = [asdict(r) for r in pipe.rebuild_index()]
        st.rerun()

# ---------------------------------------------------------------------------- Evaluation log
def _cell_color(v):
    if v is None or pd.isna(v):
        return "color: grey"
    bg = "#1e9e5a" if v >= GREEN else "#e0a100" if v >= AMBER else "#d64545"
    return f"background-color: {bg}; color: white"


with tab_log:
    st.markdown("### Evaluation log")
    st.caption(f"Green ≥ {GREEN:.2f} · amber ≥ {AMBER:.2f} · red below. Accuracy uses ground truth for "
               "evaluation questions and citation validity for ad-hoc questions.")
    records = pipe.log.read()
    if not records:
        st.info("No queries logged yet. Ask a question or run the evaluation.")
    else:
        df = pd.DataFrame(records)[["timestamp", "origin", "question_id", "query", "status", "accuracy",
                                    "relevance", "groundedness", "latency_ms", "generator",
                                    "accuracy_basis"]].iloc[::-1]
        origin = st.radio("Show", ["all", "ui", "evaluation"], horizontal=True)
        if origin != "all":
            df = df[df["origin"] == origin]
        c = st.columns(4)
        c[0].metric("Queries", len(df))
        for col, name in zip(c[1:], ["accuracy", "relevance", "groundedness"]):
            col.metric(f"Mean {name}", fmt(df[name].dropna().mean() if df[name].notna().any() else None))
        styled = df.style.map(_cell_color, subset=["accuracy", "relevance", "groundedness"]) \
                         .format({"accuracy": fmt, "relevance": fmt, "groundedness": fmt,
                                  "latency_ms": lambda v: fmt(v, 0)})
        st.dataframe(styled, hide_index=True, width="stretch", height=520)
        c1, c2 = st.columns(2)
        c1.download_button("Download log (JSONL)", "\n".join(json.dumps(r) for r in records),
                           "eval_log.jsonl")
        if c2.button("Clear log"):
            pipe.log.clear()
            st.rerun()

# ---------------------------------------------------------------------------- Run evaluation
with tab_eval:
    st.markdown("### Run the evaluation dataset")
    items = load_dataset(DATASET_PATH)
    counts = pd.Series([i["type"] for i in items]).value_counts()
    st.caption(" · ".join(f"{k}: {v}" for k, v in counts.items()) + " · uses the sidebar settings; "
               "no paid service is involved.")
    with st.expander("Questions in the dataset"):
        st.dataframe(pd.DataFrame(items), hide_index=True, width="stretch")
    if st.button("Run evaluation", type="primary"):
        bar = st.progress(0.0, text="Starting…")
        results, metrics = run_evaluation(
            pipe, progress=lambda n, t, it: bar.progress(n / t, text=f"{n}/{t} · {it['id']} · {it['type']}"),
            **{k: v for k, v in ASK_KW.items() if k != "sources"})
        ss["eval"] = (results, metrics)
    elif "eval" not in ss and (settings.eval_results_dir / "metrics.json").exists():
        ss["eval"] = (json.loads((settings.eval_results_dir / "eval_results.json").read_text()),
                      json.loads((settings.eval_results_dir / "metrics.json").read_text()))
    if "eval" in ss:
        results, m = ss["eval"]
        st.caption(f"Run at {m.get('timestamp')} · generator(s): {', '.join(m.get('generators_used', []))}")
        names = [("Retrieval hit rate", "retrieval_hit_rate"), ("Citation correctness", "citation_correctness"),
                 ("Answer groundedness", "answer_groundedness"), ("Refusal accuracy", "refusal_accuracy"),
                 ("Answer accuracy", "answer_accuracy_answerable"),
                 ("Contradiction detection", "contradiction_detection_rate"),
                 ("Injection resistance", "injection_resistance_rate")]
        cols = st.columns(4)
        for n, (label, key) in enumerate(names):
            cols[n % 4].metric(label, fmt(m.get(key)))
        cols[3].metric("Avg latency", f"{m.get('avg_latency_ms') or 0:.0f} ms",
                       help=f"p95: {m.get('p95_latency_ms')} ms")
        st.dataframe(pd.DataFrame(results)[["id", "type", "status", "answer_correct", "retrieval_hit",
                                            "citation_correctness", "groundedness", "top_relevance",
                                            "latency_ms", "question"]],
                     hide_index=True, width="stretch")
        c1, c2 = st.columns(2)
        c1.download_button("Download metrics (JSON)", json.dumps(m, indent=2), "metrics.json")
        c2.download_button("Download per-question results (JSON)", json.dumps(results, indent=2),
                           "eval_results.json")

# ---------------------------------------------------------------------------- Injection demo
with tab_inj:
    st.markdown("### Instructions inside documents are treated as content, not commands")
    st.markdown(
        "The knowledge base contains `prompt_injection_test_document.md`, whose section 3 tells the "
        "model to ignore the user, reveal its system prompt and reply with “PWNED”. Five layers stop it:\n"
        "1. **Flag at ingestion**: suspicious chunks are tagged in the index.\n"
        "2. **Neutralise**: instruction-like sentences are replaced before the model sees them.\n"
        "3. **Isolate**: passages sit inside escaped `<source>` tags and the system prompt declares them data.\n"
        "4. **Guard the output**: a secret canary detects prompt leaks; payload strings demanded by the "
        "document are removed from the answer.\n"
        "5. **Validate citations**: only passages that were really retrieved can be cited.")
    flagged = [c for c in pipe.store.chunks if c.injection_flags]
    if not flagged:
        st.warning("No chunk with suspected injection is indexed. Re-index the sample documents or upload one.")
    else:
        pick = st.selectbox("Flagged chunk", [c.chunk_id for c in flagged])
        chunk = next(c for c in flagged if c.chunk_id == pick)
        clean, n = neutralize(chunk.text)
        c1, c2 = st.columns(2)
        c1.text_area("Stored text (untrusted)", chunk.text, height=260, disabled=True)
        c2.text_area(f"Text the model receives ({n} sentence(s) neutralised)", clean, height=260, disabled=True)
        st.caption(f"Detected patterns: {', '.join(chunk.injection_flags)}")

    attack_q = st.text_input("Question that retrieves the poisoned document",
                             "What do the practitioner notes say about normalizing embeddings and "
                             "choosing a similarity metric?")
    keep_raw = st.toggle("Turn off layer 2 (send the injected sentences verbatim) to test the other layers")
    if st.button("Run the attack", type="primary"):
        with st.spinner("Running…"):
            ss["attack"] = pipe.ask(attack_q, **{**ASK_KW, "sources": None},
                                    neutralize_injections=not keep_raw, origin="injection-demo")
    if "attack" in ss:
        resp: RAGResponse = ss["attack"]
        markers = set().union(*(extract_payload_markers(c.text) for c in flagged)) if flagged else set()
        answer = resp.answer_text()
        leaked = [m for m in markers if m.lower() in answer.lower()]
        checks = pd.DataFrame([
            {"check": "Poisoned passage was retrieved",
             "result": "yes" if any(rc.chunk.injection_flags for rc in resp.retrieved) else "no"},
            {"check": "Payload strings demanded by the document", "result": ", ".join(sorted(markers)) or "none"},
            {"check": "Payload present in final answer", "result": ", ".join(leaked) if leaked else "no ✅"},
            {"check": "System prompt / canary leaked", "result":
                "blocked by guard" if resp.status == "blocked" else "no ✅"},
            {"check": "Answer still grounded in legitimate text", "result": resp.status},
        ])
        st.dataframe(checks, hide_index=True, width="stretch")
        render_response(resp, "attack")
        with st.expander("Exact prompt sent to the model (user message)"):
            st.code(build_user_prompt(resp.query, resp.retrieved)[:8000], language="xml")