from __future__ import annotations

import hashlib
import math
import time

import numpy as np
from rank_bm25 import BM25Okapi

from .models import RetrievedChunk
from .text_utils import normalize_ws, tokenize


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class HybridRetriever:
    def __init__(self, store, embedder, reranker, cfg):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.cfg = cfg
        self._bm25 = None
        self._bm25_version = -1

    def _get_bm25(self):
        if self._bm25_version != self.store.version:
            corpus = [tokenize(c.text) or ["_"] for c in self.store.chunks]
            self._bm25 = BM25Okapi(corpus) if corpus else None
            self._bm25_version = self.store.version
        return self._bm25

    def retrieve(self, query: str, top_k: int, use_hybrid: bool = True, use_rerank: bool = True,
                 sources: list[str] | None = None) -> tuple[list[RetrievedChunk], dict]:
        stats: dict = {}
        cfg = self.cfg
        t = time.perf_counter()
        qvec = self.embedder.embed_query(query)
        vec_hits = self.store.search(qvec, cfg.candidate_k, sources)
        stats["vector_search_ms"] = round((time.perf_counter() - t) * 1000, 1)

        cands: dict[int, RetrievedChunk] = {}
        for rank, (i, score) in enumerate(vec_hits):
            rc = cands.setdefault(i, RetrievedChunk(chunk=self.store.chunks[i]))
            rc.vector_score = score
            rc.fused_score += 1.0 / (cfg.rrf_k + rank + 1)

        if use_hybrid:
            t = time.perf_counter()
            bm25, q_tokens = self._get_bm25(), tokenize(query)
            if bm25 is not None and q_tokens:
                scores = np.asarray(bm25.get_scores(q_tokens))
                if sources:
                    allowed = np.array([c.source in set(sources) for c in self.store.chunks])
                    scores = np.where(allowed, scores, 0.0)
                for rank, i in enumerate(np.argsort(-scores)[: cfg.candidate_k]):
                    if scores[i] <= 0:
                        break
                    rc = cands.setdefault(int(i), RetrievedChunk(chunk=self.store.chunks[int(i)]))
                    rc.bm25_score = float(scores[i])
                    rc.fused_score += 1.0 / (cfg.rrf_k + rank + 1)
            stats["bm25_ms"] = round((time.perf_counter() - t) * 1000, 1)

        for i, rc in cands.items():                     
            if rc.vector_score is None:
                rc.vector_score = float(self.store.vectors[i] @ qvec)

        ordered = sorted(cands.items(), key=lambda kv: -kv[1].fused_score)
        before = len(ordered)
        ordered = self._dedup(ordered)
        stats["dedup_removed"] = before - len(ordered)
        items = [rc for _, rc in ordered]

        if use_rerank and self.reranker is not None and items:
            t = time.perf_counter()
            logits = self.reranker.score(query, [rc.chunk.text for rc in items])
            for rc, logit in zip(items, logits):
                rc.rerank_score = float(logit)
                rc.relevance = _sigmoid(float(logit))
            items.sort(key=lambda r: -(r.rerank_score or 0.0))
            stats["rerank_ms"] = round((time.perf_counter() - t) * 1000, 1)
        else:
            for rc in items:
                rc.relevance = max(0.0, min(1.0, rc.vector_score or 0.0))
            if not use_hybrid:
                items.sort(key=lambda r: -(r.vector_score or 0.0))

        final = items[:top_k]
        for n, rc in enumerate(final, start=1):
            rc.rank, rc.label = n, f"S{n}"
        stats["retrieval_ms"] = round(sum(v for k, v in stats.items() if k.endswith("_ms")), 1)
        return final, stats

    def _dedup(self, ordered):
        """Drop exact duplicates and near-duplicates (cosine >= threshold) keeping the best-ranked."""
        kept, seen = [], set()
        for i, rc in ordered:
            h = hashlib.md5(normalize_ws(rc.chunk.text).lower().encode()).hexdigest()
            if h in seen:
                continue
            v = self.store.vectors[i]
            if any(float(v @ self.store.vectors[j]) >= self.cfg.dedup_threshold for j, _ in kept):
                continue
            seen.add(h)
            kept.append((i, rc))
        return kept