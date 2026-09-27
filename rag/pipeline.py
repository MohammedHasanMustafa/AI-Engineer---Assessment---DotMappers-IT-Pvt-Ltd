from __future__ import annotations

import logging
import time
from pathlib import Path

from .embeddings import CrossEncoderReranker, SentenceTransformerEmbedder
from .eval_log import EvalLog, record_from_response
from .generation import (build_system_prompt, build_user_prompt, extractive_answer,
                         normalize_llm_output, parse_llm_json)
from .ingestion import SUPPORTED_EXTENSIONS, IngestResult, ingest_document
from .llm import OllamaClient
from .models import RAGResponse
from .retrieval import HybridRetriever
from .security import (QueryRejected, check_output, contains_marker, extract_payload_markers,
                       make_canary, neutralize, scan_for_injection, validate_query)
from .vector_store import FaissVectorStore
from .grounding import score_groundedness

log = logging.getLogger(__name__)
_DEFAULT = object()


class RAGPipeline:
    def __init__(self, cfg, embedder=None, reranker=_DEFAULT, llm=_DEFAULT, store=None):
        self.cfg = cfg
        self.embedder = embedder or SentenceTransformerEmbedder(cfg.embedding_model)
        self.reranker = CrossEncoderReranker(cfg.reranker_model) if reranker is _DEFAULT else reranker
        self.llm = OllamaClient(cfg.ollama_url, cfg.llm_model, cfg.llm_timeout,
                                cfg.temperature, cfg.num_ctx) if llm is _DEFAULT else llm
        self.store = store or FaissVectorStore(cfg.index_dir)
        self.retriever = HybridRetriever(self.store, self.embedder, self.reranker, cfg)
        self.log = EvalLog(cfg.log_path)

    def ingest_bytes(self, name: str, data: bytes) -> IngestResult:
        try:
            return ingest_document(name, data, self.store, self.embedder, self.cfg)
        except Exception as exc:  
            log.exception("Ingestion failed for %s", name)
            return IngestResult(Path(name).name, "error", f"Unexpected error: {exc}")

    def ingest_upload(self, name: str, data: bytes) -> IngestResult:
        """Ingest a UI upload and keep a copy so the index can be rebuilt later."""
        result = self.ingest_bytes(name, data)
        if result.status == "indexed":
            self.cfg.upload_dir.mkdir(parents=True, exist_ok=True)
            (self.cfg.upload_dir / Path(name).name).write_bytes(data)
        return result

    def ingest_path(self, path: Path) -> IngestResult:
        try:
            data = Path(path).read_bytes()
        except OSError as exc:
            return IngestResult(Path(path).name, "error", f"Cannot read file: {exc}")
        return self.ingest_bytes(Path(path).name, data)

    def ingest_directory(self, *dirs: Path) -> list[IngestResult]:
        results = []
        for d in dirs:
            if not Path(d).exists():
                continue
            for p in sorted(Path(d).iterdir()):
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                    results.append(self.ingest_path(p))
        return results

    def delete_document(self, source: str) -> int:
        removed = self.store.delete_source(source)
        upload = self.cfg.upload_dir / source
        if upload.exists():
            upload.unlink()
        return removed

    def rebuild_index(self) -> list[IngestResult]:
        self.store.clear()
        return self.ingest_directory(self.cfg.docs_dir, self.cfg.upload_dir)

    def llm_status(self) -> tuple[bool, str]:
        if self.llm is None:
            return False, "No LLM configured"
        return self.llm.status()

    def ask(self, query: str, top_k: int | None = None, use_hybrid: bool = True,
            use_rerank: bool = True, sources: list[str] | None = None, generator: str = "auto",
            neutralize_injections: bool = True, min_relevance: float | None = None,
            log_query: bool = True, origin: str = "ui") -> RAGResponse:
        t0 = time.perf_counter()
        resp = RAGResponse(query=query or "")
        try:
            self._answer(resp, query, top_k or self.cfg.top_k, use_hybrid, use_rerank, sources,
                         generator, neutralize_injections, min_relevance)
        except Exception as exc:  
            log.exception("Pipeline failure")
            resp.status, resp.evidence = "insufficient_evidence", []
            resp.message = f"The system hit an internal error and did not answer ({type(exc).__name__}: {exc})."
        resp.latency_ms["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if log_query and resp.status != "rejected":
            self.log.append(record_from_response(resp, origin))
        return resp

    def _answer(self, resp: RAGResponse, query, top_k, use_hybrid, use_rerank, sources,
                generator, neutralize_injections, min_relevance) -> None:
        cfg = self.cfg
        try:
            q, warns = validate_query(query, cfg.max_query_chars, cfg.hard_max_query_chars)
        except QueryRejected as exc:
            resp.status, resp.message = "rejected", str(exc)
            return
        resp.query = q
        resp.warnings += warns
        if len(self.store) == 0:
            resp.message = "No documents are indexed yet. Upload documents in the Documents tab."
            resp.generator = "no-index"
            return

        reranking = use_rerank and self.reranker is not None
        retrieved, stats = self.retriever.retrieve(q, top_k, use_hybrid, reranking, sources)
        resp.retrieved = retrieved
        resp.latency_ms.update(stats)
        resp.relevance = max((rc.relevance for rc in retrieved), default=0.0)

        threshold = min_relevance if min_relevance is not None else (
            cfg.min_rerank_relevance if reranking else cfg.min_vector_relevance)
        if not retrieved or resp.relevance < threshold:
            resp.generator = "relevance-gate"
            resp.message = (f"No retrieved passage is relevant enough to answer this question "
                            f"(best relevance {resp.relevance:.2f} < threshold {threshold:.2f}).")
            return

        markers: set[str] = set()
        for rc in retrieved:
            flags = rc.chunk.injection_flags or scan_for_injection(rc.chunk.text)
            if neutralize_injections:
                rc.prompt_text, rc.neutralized = neutralize(rc.chunk.text)
            else:
                rc.prompt_text, rc.neutralized = rc.chunk.text, 0
            markers |= extract_payload_markers(rc.chunk.text)
            if flags:
                action = (f"{rc.neutralized} sentence(s) neutralised" if neutralize_injections
                          else "kept verbatim but isolated as untrusted data")
                resp.security_events.append(
                    f"{rc.label} ({rc.chunk.source}) contains suspected injected instructions "
                    f"[{', '.join(flags)}]: {action}.")

        t = time.perf_counter()
        canary = make_canary()
        system_prompt = build_system_prompt(canary)
        raw_output = ""
        used_llm = False
        if generator != "extractive" and self.llm is not None and self.llm.available():
            try:
                raw_output = self.llm.chat_json(system_prompt, build_user_prompt(q, retrieved))
                data = parse_llm_json(raw_output)
                status, evidence, inference, conflicts, warns, cstats = normalize_llm_output(data, retrieved)
                resp.warnings += warns
                total = cstats["valid"] + cstats["fabricated"]
                resp.citation_validity = cstats["valid"] / total if total else None
                resp.generator = f"ollama:{getattr(self.llm, 'model', 'llm')}"
                used_llm = True
            except Exception as exc:  
                resp.warnings.append(f"LLM call failed ({type(exc).__name__}); used extractive fallback.")
        if not used_llm:
            status, evidence, inference, conflicts = extractive_answer(
                q, retrieved, self.embedder, cfg.extractive_max_sentences, cfg.extractive_min_sim)
            resp.generator = "extractive"
            resp.citation_validity = 1.0 if evidence else None
        resp.latency_ms["generation_ms"] = round((time.perf_counter() - t) * 1000, 1)

        leaks = check_output(raw_output + " " + inference + " " + " ".join(e.claim for e in evidence),
                             canary, system_prompt)
        if leaks:
            resp.security_events += [f"BLOCKED: {e}" for e in leaks]
            resp.status, resp.evidence, resp.inference, resp.conflicts = "blocked", [], "", ""
            resp.message = ("The response was withheld because it disclosed protected instructions. "
                            "This usually means a document tried to hijack the model.")
            return
        if markers:
            kept = []
            for e in evidence:
                hit = contains_marker(e.claim, markers)
                if hit:
                    resp.security_events.append(f"BLOCKED evidence item echoing injected payload {hit}.")
                else:
                    kept.append(e)
            evidence = kept
            hit = contains_marker(inference, markers) + contains_marker(conflicts, markers)
            if hit:
                resp.security_events.append(f"BLOCKED inference text echoing injected payload {sorted(set(hit))}.")
                inference = " ".join(s for s in inference.split(". ") if not contains_marker(s, markers))
                conflicts = " ".join(s for s in conflicts.split(". ") if not contains_marker(s, markers))
            if not evidence and status != "insufficient_evidence":
                status = "insufficient_evidence"

        resp.status, resp.evidence, resp.inference, resp.conflicts = status, evidence, inference, conflicts
        if status == "insufficient_evidence":
            resp.message = resp.message or "The retrieved passages do not contain enough information to answer."
            return

        t = time.perf_counter()
        resp.groundedness = score_groundedness(evidence, retrieved, self.embedder,
                                               cfg.grounding_cos_threshold, cfg.grounding_lex_threshold)
        resp.latency_ms["grounding_ms"] = round((time.perf_counter() - t) * 1000, 1)
        if resp.groundedness is not None and resp.groundedness < 0.5:
            resp.warnings.append("Less than half of the evidence claims are strongly supported by "
                                 "their cited passages. Check the passages before relying on this answer.")
