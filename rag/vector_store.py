from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

import faiss
import numpy as np

from .models import Chunk

log = logging.getLogger(__name__)


class FaissVectorStore:

    def __init__(self, index_dir: Path):
        self.dir = Path(index_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray | None = None
        self.doc_meta: dict[str, dict] = {}
        self.index = None
        self.version = 0                      
        self._lock = threading.RLock()
        self.load()

    @property
    def _chunks_path(self) -> Path:
        return self.dir / "chunks.json"

    @property
    def _vectors_path(self) -> Path:
        return self.dir / "vectors.npy"

    def load(self) -> None:
        with self._lock:
            if self._chunks_path.exists() and self._vectors_path.exists():
                try:
                    data = json.loads(self._chunks_path.read_text(encoding="utf-8"))
                    self.chunks = [Chunk.from_dict(c) for c in data["chunks"]]
                    self.doc_meta = data.get("documents", {})
                    self.vectors = np.load(self._vectors_path).astype("float32")
                    if len(self.chunks) != len(self.vectors):
                        raise ValueError("chunk/vector count mismatch")
                except Exception as exc:  
                    log.warning("Index could not be loaded (%s); starting empty.", exc)
                    self.chunks, self.vectors, self.doc_meta = [], None, {}
            self._rebuild()

    def save(self) -> None:
        with self._lock:
            tmp_c = self._chunks_path.with_suffix(".json.tmp")
            tmp_c.write_text(json.dumps({"documents": self.doc_meta,
                                         "chunks": [c.to_dict() for c in self.chunks]}),
                             encoding="utf-8")
            tmp_v = self.dir / "vectors.tmp.npy"
            np.save(tmp_v, self.vectors if self.vectors is not None else np.zeros((0, 1), "float32"))
            os.replace(tmp_c, self._chunks_path)
            os.replace(tmp_v, self._vectors_path)

    def _rebuild(self) -> None:
        if self.vectors is None or len(self.vectors) == 0:
            self.index = None
        else:
            self.index = faiss.IndexFlatIP(self.vectors.shape[1])
            self.index.add(self.vectors)
        self.version += 1

    def add(self, chunks: list[Chunk], vectors: np.ndarray, doc_info: dict) -> None:
        vectors = np.asarray(vectors, dtype="float32")
        with self._lock:
            if self.vectors is not None and len(self.vectors) and vectors.shape[1] != self.vectors.shape[1]:
                raise ValueError("Embedding dimension changed; reset the index and re-ingest.")
            self.vectors = vectors if self.vectors is None or len(self.vectors) == 0 \
                else np.vstack([self.vectors, vectors])
            self.chunks.extend(chunks)
            self.doc_meta[doc_info["source"]] = doc_info
            self._rebuild()
            self.save()

    def delete_source(self, source: str) -> int:
        with self._lock:
            keep = [i for i, c in enumerate(self.chunks) if c.source != source]
            removed = len(self.chunks) - len(keep)
            self.chunks = [self.chunks[i] for i in keep]
            self.vectors = self.vectors[keep] if (self.vectors is not None and keep) else None
            self.doc_meta.pop(source, None)
            self._rebuild()
            self.save()
            return removed

    def clear(self) -> None:
        with self._lock:
            self.chunks, self.vectors, self.doc_meta = [], None, {}
            self._rebuild()
            self.save()

    def __len__(self) -> int:
        return len(self.chunks)

    def has_hash(self, doc_hash: str) -> bool:
        return any(m.get("doc_hash") == doc_hash for m in self.doc_meta.values())

    def has_source(self, source: str) -> bool:
        return source in self.doc_meta

    def search(self, query_vec: np.ndarray, k: int, sources: list[str] | None = None) -> list[tuple[int, float]]:
        """Return [(chunk_position, cosine)] best first. `sources` = metadata filter."""
        with self._lock:
            if self.index is None or k <= 0:
                return []
            q = np.asarray(query_vec, dtype="float32").reshape(1, -1)
            if sources:
                idx = np.array([i for i, c in enumerate(self.chunks) if c.source in set(sources)], dtype=int)
                if len(idx) == 0:
                    return []
                scores = self.vectors[idx] @ q[0]
                order = np.argsort(-scores)[:k]
                return [(int(idx[o]), float(scores[o])) for o in order]
            scores, ids = self.index.search(q, min(k, len(self.chunks)))
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
